import numpy as np
from app.ml.yolo_predictor import YOLOPredictor


def test_yolo_model_loaded():
    predictor = YOLOPredictor.get_instance()
    assert predictor.is_loaded is True
    assert 4 in predictor.names
    assert predictor.names[4] == "pothole"
    assert "alligator crack" in predictor.names.values()


def test_yolo_image_inference():
    predictor = YOLOPredictor.get_instance()
    # Create synthetic test image (640x640x3)
    dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
    dummy_img[200:300, 200:300] = 120  # gray patch

    result = predictor.predict_image(dummy_img, conf_threshold=0.25, annotate=False)
    assert result["status"] == "success"
    assert "inference_time_ms" in result
    assert "detections" in result
    assert "class_counts" in result
