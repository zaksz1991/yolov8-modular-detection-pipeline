from ultralytics import YOLO

class YOLODetector:
    def __init__(self, model_name='yolov8n.pt'):
        self.model = YOLO(model_name)
        
    def detect(self, source, conf_threshold=0.5):
        return self.model.predict(source, conf=conf_threshold, verbose=False)