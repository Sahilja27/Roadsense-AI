import pytest
import pandas as pd
import numpy as np
from app.ml.sensor_predictor import SensorPredictor


def test_sensor_predictor_loaded():
    predictor = SensorPredictor.get_instance()
    assert predictor.is_loaded is True
    assert "Bump" in predictor.classes
    assert "Pothole" in predictor.classes
    assert len(predictor.feature_columns) == 30


def test_sensor_dataframe_standardization():
    raw_df = pd.DataFrame({
        "time": [f"2026-10-15 12:00:0{i}" for i in range(10)],
        "acc_x": np.random.randn(10),
        "acc_y": np.random.randn(10),
        "acc_z": np.random.randn(10),
        "lat": [28.61] * 10,
        "lon": [77.20] * 10,
    })
    clean_df, meta = SensorPredictor.standardize_sensor_dataframe(raw_df)
    assert "accelerometer_x" in clean_df.columns
    assert "accelerometer_y" in clean_df.columns
    assert "accelerometer_z" in clean_df.columns
    assert "seconds_elapsed" in clean_df.columns
    assert meta["has_gps"] is True


def test_missing_required_accelerometer_raises():
    bad_df = pd.DataFrame({
        "seconds_elapsed": [1, 2, 3],
        "gyroscope_x": [0, 0, 0],
    })
    with pytest.raises(ValueError) as exc:
        SensorPredictor.standardize_sensor_dataframe(bad_df)
    assert "Required accelerometer signal" in str(exc.value)


def test_real_sensor_model_inference():
    predictor = SensorPredictor.get_instance()
    # Create synthetic 100-sample 1-second window
    data = {
        "seconds_elapsed": np.linspace(0, 1.0, 100),
        "accelerometer_x": np.random.normal(0, 0.5, 100),
        "accelerometer_y": np.random.normal(0, 0.5, 100),
        "accelerometer_z": np.random.normal(9.8, 2.5, 100),
        "gyroscope_x": np.random.normal(0, 0.1, 100),
        "gyroscope_y": np.random.normal(0, 0.1, 100),
        "gyroscope_z": np.random.normal(0, 0.1, 100),
        "latitude": [28.6139] * 100,
        "longitude": [77.2090] * 100,
    }
    df = pd.DataFrame(data)
    result = predictor.predict_dataframe(df)

    assert result["status"] == "success"
    assert result["total_windows"] == 1
    assert result["windows"][0]["predicted_class"] in ["Bump", "Pothole"]
    assert 0.0 <= result["windows"][0]["confidence"] <= 1.0
    assert len(result["windows"][0]["features"]) == 30
