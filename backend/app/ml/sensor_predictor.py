import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from app.core.config import settings
from app.core.demo_routes import generate_route_coordinates

SENSOR_COLUMNS = [
    "accelerometer_x",
    "accelerometer_y",
    "accelerometer_z",
    "gyroscope_x",
    "gyroscope_y",
    "gyroscope_z",
]

MAX_INTERPOLATION_GAP_SECONDS = 5.0


def is_valid_coordinate(lat: Any, lon: Any) -> bool:
    """Validates real-world geographic coordinates ranges."""
    try:
        lat = float(lat)
        lon = float(lon)
        if np.isnan(lat) or np.isnan(lon):
            return False
        if abs(lat) < 1e-6 and abs(lon) < 1e-6:
            return False  # (0, 0) null island
        return -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0
    except (TypeError, ValueError):
        return False


class SensorPredictor:
    """Production ML Inference Engine for Smartphone Sensor Road Damage Detection."""

    _instance: Optional["SensorPredictor"] = None

    def __init__(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = settings.SENSOR_MODEL_PATH
            # Fallback to sibling project path if local copy not found
            if not Path(model_path).exists():
                fallback_path = Path("D:/RoadSense-SensorML/models/roadsense_model.joblib")
                if fallback_path.exists():
                    model_path = str(fallback_path)

        self.model_path = model_path
        self.model = None
        self.feature_columns = []
        self.classes = []
        self.is_loaded = False
        self._load_model()

    @classmethod
    def get_instance(cls) -> "SensorPredictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        """Loads the serialized scikit-learn model bundle and feature registry."""
        p = Path(self.model_path)
        if not p.exists():
            print(f"[SensorPredictor] Warning: Model file not found at {self.model_path}")
            self.is_loaded = False
            return

        try:
            bundle = joblib.load(self.model_path)
            self.model = bundle["model"]
            self.feature_columns = list(bundle["feature_columns"])
            self.classes = list(bundle.get("classes", getattr(self.model, "classes_", ["Bump", "Pothole"])))
            self.is_loaded = True
            print(f"[SensorPredictor] Successfully loaded sensor model with classes: {self.classes}")
        except Exception as e:
            print(f"[SensorPredictor] Failed to load model: {e}")
            self.is_loaded = False

    @staticmethod
    def standardize_sensor_dataframe(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Standardizes column names, checks required sensor streams,
        and computes seconds_elapsed if not directly provided.
        """
        metadata = {
            "has_gps": False,
            "has_gyro": False,
            "has_speed": False,
            "total_rows": len(df),
            "rejected_rows": 0,
            "gps_valid_rows": 0,
        }

        # Canonical target columns
        targets = {
            "accelerometer_x": ["accelerometer_x", "acc_x", "ax", "accel_x"],
            "accelerometer_y": ["accelerometer_y", "acc_y", "ay", "accel_y"],
            "accelerometer_z": ["accelerometer_z", "acc_z", "az", "accel_z"],
            "gyroscope_x": ["gyroscope_x", "gyro_x", "gx"],
            "gyroscope_y": ["gyroscope_y", "gyro_y", "gy"],
            "gyroscope_z": ["gyroscope_z", "gyro_z", "gz"],
            "seconds_elapsed": ["seconds_elapsed", "elapsed_seconds", "time_elapsed", "elapsed_time"],
            "latitude": ["location_latitude", "latitude", "lat"],
            "longitude": ["location_longitude", "longitude", "lon", "lng"],
            "speed": ["speed", "spd", "vehicle_speed"],
            "timestamp": ["timestamp", "time", "datetime", "date"],
        }

        col_mapping = {}
        assigned_targets = set()

        # Step 1: Check exact matches first
        lower_cols = {col: col.strip().lower().replace(" ", "_").replace("(", "").replace(")", "") for col in df.columns}
        for target, candidates in targets.items():
            for col, clean in lower_cols.items():
                if target in assigned_targets:
                    break
                if clean in candidates:
                    col_mapping[col] = target
                    assigned_targets.add(target)
                    break

        # Step 2: Fallback partial matches for missing targets
        for target, candidates in targets.items():
            if target in assigned_targets:
                continue
            for col, clean in lower_cols.items():
                if col in col_mapping:
                    continue
                # Exclude uncalibrated / total acceleration from primary acc mapping
                if "uncalibrated" in clean or "total" in clean:
                    continue
                if any(cand in clean for cand in candidates):
                    col_mapping[col] = target
                    assigned_targets.add(target)
                    break

        clean_df = df.rename(columns=col_mapping)

        # Validate required accelerometer columns
        for acc_col in ["accelerometer_x", "accelerometer_y", "accelerometer_z"]:
            if acc_col not in clean_df.columns:
                raise ValueError(
                    f"Required accelerometer signal '{acc_col}' is missing from CSV. "
                    f"Available columns: {list(df.columns)}"
                )

        # Gyroscope defaults to 0.0 if not present, but note it in metadata
        has_gyro = True
        for gyro_col in ["gyroscope_x", "gyroscope_y", "gyroscope_z"]:
            if gyro_col not in clean_df.columns:
                clean_df[gyro_col] = 0.0
                has_gyro = False
        metadata["has_gyro"] = has_gyro

        # Check seconds_elapsed or synthesize from timestamp
        if "seconds_elapsed" not in clean_df.columns:
            if "timestamp" in clean_df.columns:
                try:
                    ts_series = pd.to_datetime(clean_df["timestamp"], errors="coerce")
                    if ts_series.isna().all():
                        # numeric timestamps or relative
                        numeric_ts = pd.to_numeric(clean_df["timestamp"], errors="coerce")
                        clean_df["seconds_elapsed"] = numeric_ts - numeric_ts.min()
                    else:
                        t0 = ts_series.dropna().iloc[0]
                        clean_df["seconds_elapsed"] = (ts_series - t0).dt.total_seconds()
                except Exception:
                    clean_df["seconds_elapsed"] = np.arange(len(clean_df)) * 0.1
            else:
                # Default assume 10Hz sampling
                clean_df["seconds_elapsed"] = np.arange(len(clean_df)) * 0.1

        # Check GPS presence without generating fake coordinates
        if "latitude" in clean_df.columns and "longitude" in clean_df.columns:
            clean_df["latitude"] = pd.to_numeric(clean_df["latitude"], errors="coerce")
            clean_df["longitude"] = pd.to_numeric(clean_df["longitude"], errors="coerce")
            
            # Filter valid coordinate bounds (-90 to 90, -180 to 180, not (0,0))
            valid_mask = clean_df.apply(
                lambda r: is_valid_coordinate(r["latitude"], r["longitude"]), axis=1
            )
            clean_df.loc[~valid_mask, ["latitude", "longitude"]] = np.nan
            valid_count = int(valid_mask.sum())
            metadata["gps_valid_rows"] = valid_count
            if valid_count > 0:
                metadata["has_gps"] = True
        else:
            clean_df["latitude"] = np.nan
            clean_df["longitude"] = np.nan

        # Speed column
        if "speed" in clean_df.columns:
            clean_df["speed"] = pd.to_numeric(clean_df["speed"], errors="coerce")
            metadata["has_speed"] = True
        else:
            clean_df["speed"] = np.nan

        # Clean numerical values
        required_numeric = ["seconds_elapsed", *SENSOR_COLUMNS]
        for c in required_numeric:
            clean_df[c] = pd.to_numeric(clean_df[c], errors="coerce")

        before_drop = len(clean_df)
        clean_df = (
            clean_df.dropna(subset=required_numeric)
            .sort_values("seconds_elapsed")
            .reset_index(drop=True)
        )
        metadata["rejected_rows"] = before_drop - len(clean_df)

        if clean_df.empty:
            raise ValueError("No valid sensor readings found after removing invalid/null rows.")

        return clean_df, metadata

    def extract_features(
        self, df: pd.DataFrame, route_mode: str = "real_gps"
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Extracts exactly the 30 kinematic and shock features per 1-second window
        matching the exact training distribution of roadsense_model.joblib.
        Associates each window with valid GPS, interpolated GPS, or demo route positions.
        """
        df_copy = df.copy()
        df_copy["window"] = np.floor(
            df_copy["seconds_elapsed"] - df_copy["seconds_elapsed"].min()
        ).astype(int)

        # Pre-extract all valid GPS points with timestamp for linear interpolation if needed
        valid_gps_df = df_copy.dropna(subset=["latitude", "longitude"])[
            ["seconds_elapsed", "latitude", "longitude"]
        ].sort_values("seconds_elapsed")

        records = []
        for window_id, w in df_copy.groupby("window"):
            # Model was trained on windows of at least 50 samples
            if len(w) < 50:
                continue

            w_start = float(w["seconds_elapsed"].min())
            w_end = float(w["seconds_elapsed"].max())
            w_mid = (w_start + w_end) / 2.0

            # GPS representative position calculation
            lat = None
            lon = None
            provenance = "none"
            is_interpolated = False
            is_location_pending = True

            w_valid_gps = w.dropna(subset=["latitude", "longitude"])
            if not w_valid_gps.empty:
                # Use MEDIAN valid GPS position within the 1-second window
                lat = float(w_valid_gps["latitude"].median())
                lon = float(w_valid_gps["longitude"].median())
                provenance = "actual_gps"
                is_interpolated = False
                is_location_pending = False
            elif not valid_gps_df.empty:
                # Attempt interpolation if gap is within MAX_INTERPOLATION_GAP_SECONDS
                prev_points = valid_gps_df[valid_gps_df["seconds_elapsed"] <= w_start]
                next_points = valid_gps_df[valid_gps_df["seconds_elapsed"] >= w_end]

                if not prev_points.empty and not next_points.empty:
                    prev_p = prev_points.iloc[-1]
                    next_p = next_points.iloc[0]
                    gap = float(next_p["seconds_elapsed"] - prev_p["seconds_elapsed"])

                    if gap <= MAX_INTERPOLATION_GAP_SECONDS and gap > 0:
                        alpha = (w_mid - prev_p["seconds_elapsed"]) / gap
                        lat = float(prev_p["latitude"] + alpha * (next_p["latitude"] - prev_p["latitude"]))
                        lon = float(prev_p["longitude"] + alpha * (next_p["longitude"] - prev_p["longitude"]))
                        provenance = "interpolated_gps"
                        is_interpolated = True
                        is_location_pending = False

            spd = float(w["speed"].dropna().median()) if "speed" in w.columns and w["speed"].notna().any() else None

            features = {
                "window": int(window_id),
                "start_seconds": w_start,
                "end_seconds": w_end,
                "samples": len(w),
                "latitude": round(lat, 6) if lat is not None else None,
                "longitude": round(lon, 6) if lon is not None else None,
                "speed": round(spd, 2) if spd is not None else None,
                "coordinate_provenance": provenance,
                "is_interpolated": is_interpolated,
                "is_location_pending": is_location_pending,
            }

            for col in SENSOR_COLUMNS:
                x = w[col].to_numpy(dtype=float)
                features[f"{col}_std"] = float(np.std(x))
                features[f"{col}_range"] = float(np.ptp(x))
                features[f"{col}_max_abs"] = float(np.max(np.abs(x)))
                features[f"{col}_rms"] = float(np.sqrt(np.mean(x**2)))

            ax = w["accelerometer_x"].to_numpy(dtype=float)
            ay = w["accelerometer_y"].to_numpy(dtype=float)
            az = w["accelerometer_z"].to_numpy(dtype=float)

            magnitude = np.sqrt(ax**2 + ay**2 + az**2)
            times = w["seconds_elapsed"].to_numpy(dtype=float)
            dt = np.diff(times)
            dm = np.diff(magnitude)

            valid = dt > 0
            jerk = dm[valid] / dt[valid]

            features["magnitude_std"] = float(np.std(magnitude))
            features["magnitude_max"] = float(np.max(magnitude))
            features["magnitude_rms"] = float(np.sqrt(np.mean(magnitude**2)))
            features["jerk_std"] = float(np.std(jerk)) if len(jerk) else 0.0
            features["jerk_max_abs"] = float(np.max(np.abs(jerk))) if len(jerk) else 0.0

            prominence = max(np.std(magnitude) * 0.75, 0.01)
            peaks, _ = find_peaks(magnitude, prominence=prominence)
            features["peak_count"] = int(len(peaks))

            records.append(features)

        if not records:
            raise ValueError(
                "No usable one-second windows found (minimum 50 samples per window required). "
                "Ensure sampling frequency is >= 50Hz or input recording is long enough."
            )

        # Apply Demo Route Mode or fallback anchor to Chhatrapati Sambhajinagar corridor
        has_local_gps = any(
            rec.get("latitude") is not None
            and 19.70 <= rec["latitude"] <= 20.05
            and 75.10 <= rec["longitude"] <= 75.60
            for rec in records
        )
        if route_mode == "demo_aurangabad" or not has_local_gps:
            demo_coords = generate_route_coordinates(len(records))
            for i, rec in enumerate(records):
                rec["latitude"] = demo_coords[i]["latitude"]
                rec["longitude"] = demo_coords[i]["longitude"]
                rec["coordinate_provenance"] = "demo_route"
                rec["is_interpolated"] = False
                rec["is_location_pending"] = False

        windows_df = pd.DataFrame(records)

        # Match exact training feature columns and ordering
        X = windows_df.reindex(columns=self.feature_columns)
        if X.isna().any().any():
            X = X.fillna(0.0)

        return windows_df, X

    def predict_dataframe(self, df: pd.DataFrame, route_mode: str = "real_gps") -> Dict[str, Any]:
        """
        Runs the full standardized pipeline: validation -> feature extraction -> model inference.
        Supports both 'real_gps' and 'demo_aurangabad' presentation route modes.
        """
        if not self.is_loaded:
            self._load_model()

        if not self.is_loaded or self.model is None:
            raise RuntimeError(
                f"Trained sensor model could not be loaded from {self.model_path}. "
                "Please verify that 'roadsense_model.joblib' is present."
            )

        clean_df, metadata = self.standardize_sensor_dataframe(df)
        metadata["route_mode"] = route_mode

        windows_df, X = self.extract_features(clean_df, route_mode=route_mode)

        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)

        results = []
        class_list = list(self.model.classes_)

        for i, pred_label in enumerate(predictions):
            prob_idx = class_list.index(pred_label)
            confidence = round(float(probabilities[i, prob_idx]), 4)
            row = windows_df.iloc[i]

            feature_dict = {
                feat: round(float(X.iloc[i][feat]), 4)
                for feat in self.feature_columns
            }

            results.append({
                "window_index": int(row["window"]),
                "start_seconds": round(float(row["start_seconds"]), 2),
                "end_seconds": round(float(row["end_seconds"]), 2),
                "sample_count": int(row["samples"]),
                "predicted_class": str(pred_label),
                "confidence": confidence,
                "confidence_percent": round(confidence * 100.0, 1),
                "latitude": row["latitude"],
                "longitude": row["longitude"],
                "speed": row["speed"],
                "coordinate_provenance": row["coordinate_provenance"],
                "is_interpolated": bool(row["is_interpolated"]),
                "is_location_pending": bool(row["is_location_pending"]),
                "features": feature_dict,
            })

        # Summary counts strictly for supported classes ('Bump', 'Pothole')
        counts = {cls_name: 0 for cls_name in self.classes}
        for r in results:
            cls_name = r["predicted_class"]
            counts[cls_name] = counts.get(cls_name, 0) + 1

        geolocated_count = sum(1 for r in results if r["latitude"] is not None)
        metadata["geolocated_windows"] = geolocated_count

        return {
            "status": "success",
            "model_path": self.model_path,
            "supported_classes": self.classes,
            "total_windows": len(results),
            "counts": counts,
            "metadata": metadata,
            "windows": results,
        }
