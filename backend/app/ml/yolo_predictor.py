import os
import time
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple
import cv2
import numpy as np
from app.core.config import settings

# Color palette for RDD2022 classes (BGR format for OpenCV)
CLASS_COLORS = {
    0: (0, 165, 255),    # alligator crack: Orange
    1: (255, 191, 0),    # transverse crack: Deep Sky Blue
    2: (255, 105, 180),  # longitudinal crack: Hot Pink
    3: (128, 0, 128),    # other corruption: Purple
    4: (0, 0, 255),      # pothole: Bright Red
}

DEFAULT_NAMES = {
    0: "alligator crack",
    1: "transverse crack",
    2: "longitudinal crack",
    3: "other corruption",
    4: "pothole",
}


class YOLOPredictor:
    """Singleton production YOLO model runner for RDD2022 road defect detection."""

    _instance: Optional["YOLOPredictor"] = None

    def __init__(self, weights_path: Optional[str] = None):
        if weights_path is None:
            weights_path = settings.YOLO_MODEL_PATH
            if not Path(weights_path).exists():
                fallback_path = Path("D:/RoadSense-YOLO/runs/detect/train-7/weights/best.pt")
                if fallback_path.exists():
                    weights_path = str(fallback_path)
                else:
                    alt_path = Path("D:/roadsense-vision/models/best.pt")
                    if alt_path.exists():
                        weights_path = str(alt_path)

        self.weights_path = weights_path
        self.model = None
        self.names = DEFAULT_NAMES
        self.is_loaded = False
        self._load_model()

    @classmethod
    def get_instance(cls) -> "YOLOPredictor":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_model(self):
        """Initializes and caches the Ultralytics YOLO model."""
        p = Path(self.weights_path)
        if not p.exists():
            print(f"[YOLOPredictor] Warning: Trained weights not found at {self.weights_path}")
            self.is_loaded = False
            return

        try:
            from ultralytics import YOLO
            self.model = YOLO(self.weights_path)
            if hasattr(self.model, "names") and self.model.names:
                self.names = self.model.names
            self.is_loaded = True
            print(f"[YOLOPredictor] Successfully loaded YOLO weights: {self.weights_path}")
            print(f"[YOLOPredictor] Model classes: {self.names}")
        except Exception as e:
            print(f"[YOLOPredictor] Error initializing YOLO model: {e}")
            self.is_loaded = False

    def predict_image(
        self,
        image_input: Union[str, Path, bytes, np.ndarray],
        conf_threshold: float = 0.25,
        imgsz: int = 640,
        annotate: bool = True,
        save_annotated_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Runs inference on a single image.
        Returns structured detection metadata and optionally saves annotated output.
        """
        if not self.is_loaded:
            self._load_model()

        if not self.is_loaded or self.model is None:
            raise RuntimeError(
                f"YOLO model not loaded. Trained weights missing at: {self.weights_path}"
            )

        start_time = time.time()

        # Handle raw bytes
        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image from byte buffer.")
            source = img
        else:
            source = image_input

        results = self.model.predict(
            source=source,
            conf=conf_threshold,
            imgsz=imgsz,
            verbose=False,
        )
        inference_time_ms = round((time.time() - start_time) * 1000.0, 2)

        detections: List[Dict[str, Any]] = []
        max_conf = 0.0

        if results and len(results) > 0:
            res = results[0]
            boxes = res.boxes
            img_h, img_w = res.orig_shape

            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls_id = int(box.cls[0])
                    cls_name = self.names.get(cls_id, f"class_{cls_id}")
                    conf_score = round(float(box.conf[0]), 4)
                    xyxy = [round(float(c), 1) for c in box.xyxy[0].tolist()]

                    xmin, ymin, xmax, ymax = xyxy
                    norm_x = round(((xmin + xmax) / 2.0) / img_w, 4)
                    norm_y = round(((ymin + ymax) / 2.0) / img_h, 4)
                    norm_w = round((xmax - xmin) / img_w, 4)
                    norm_h = round((ymax - ymin) / img_h, 4)

                    if conf_score > max_conf:
                        max_conf = conf_score

                    detections.append({
                        "class_id": cls_id,
                        "class_name": cls_name,
                        "confidence": conf_score,
                        "bbox_xyxy": xyxy,
                        "bbox_normalized": [norm_x, norm_y, norm_w, norm_h],
                    })

            # Save annotated image if requested
            if annotate and save_annotated_path is not None:
                orig_img = res.orig_img
                annotated_img = self.draw_detections(orig_img, detections)
                Path(save_annotated_path).parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(save_annotated_path), annotated_img)

        # Class counts
        class_counts = {name: 0 for name in self.names.values()}
        for d in detections:
            class_counts[d["class_name"]] = class_counts.get(d["class_name"], 0) + 1

        return {
            "status": "success",
            "inference_time_ms": inference_time_ms,
            "total_detections": len(detections),
            "max_confidence": round(max_conf, 4),
            "class_counts": class_counts,
            "detections": detections,
            "annotated_path": str(save_annotated_path) if save_annotated_path else None,
        }

    def process_video_generator(
        self,
        video_path: Union[str, Path],
        conf_threshold: float = 0.25,
        sampling_fps: float = 2.0,
        output_annotated_path: Optional[Union[str, Path]] = None,
    ):
        """
        Memory-efficient generator that yields detection frames and progress.
        Avoids reading entire large videos into RAM.
        """
        if not self.is_loaded:
            self._load_model()

        if not self.is_loaded or self.model is None:
            raise RuntimeError(f"YOLO model not loaded from {self.weights_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Cannot open video stream at: {video_path}")

        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        frame_step = max(1, int(round(orig_fps / sampling_fps)))

        writer = None
        if output_annotated_path is not None:
            Path(output_annotated_path).parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(
                str(output_annotated_path),
                fourcc,
                sampling_fps,
                (width, height),
            )

        frame_idx = 0
        processed_count = 0
        accumulated_detections = []

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % frame_step == 0:
                    timestamp_s = round(frame_idx / orig_fps, 2)
                    res = self.predict_image(
                        frame,
                        conf_threshold=conf_threshold,
                        annotate=False,
                    )
                    frame_dets = res["detections"]
                    for d in frame_dets:
                        d["frame_number"] = frame_idx
                        d["timestamp_seconds"] = timestamp_s
                        accumulated_detections.append(d)

                    if writer is not None:
                        annotated_frame = self.draw_detections(frame, frame_dets)
                        writer.write(annotated_frame)

                    processed_count += 1
                    progress = min(100.0, round((frame_idx / total_frames) * 100.0, 1))

                    yield {
                        "frame_number": frame_idx,
                        "timestamp_seconds": timestamp_s,
                        "progress_percent": progress,
                        "frame_detections": frame_dets,
                        "total_detections_so_far": len(accumulated_detections),
                    }

                frame_idx += 1

        finally:
            cap.release()
            if writer is not None:
                writer.release()

    @staticmethod
    def draw_detections(
        image: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> np.ndarray:
        """Draws clean high-contrast bounding boxes, labels, and confidence overlays."""
        annotated = image.copy()
        for det in detections:
            bbox = det["bbox_xyxy"]
            xmin, ymin, xmax, ymax = [int(coord) for coord in bbox]
            cls_id = det.get("class_id", 0)
            cls_name = det.get("class_name", "defect")
            conf = det.get("confidence", 0.0)

            color = CLASS_COLORS.get(cls_id, (0, 255, 0))

            # Bounding Box
            cv2.rectangle(annotated, (xmin, ymin), (xmax, ymax), color, 2)

            # Label banner
            label = f"{cls_name.upper()} {conf:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                1,
            )
            cv2.rectangle(
                annotated,
                (xmin, max(0, ymin - text_h - 6)),
                (xmin + text_w + 6, max(text_h + 6, ymin)),
                color,
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (xmin + 3, max(text_h + 2, ymin - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        return annotated
