import json
import cv2
from ultralytics import YOLO

def export_to_json(results, output_filename='results.json'):
    structured_data = []
    for r in results:
        for box in r.boxes:
            structured_data.append({
                "class_id": int(box.cls),
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()[0]
            })
    with open(output_filename, 'w') as f:
        json.dump(structured_data, f, indent=4)

def process_video(model, video_path):
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        success, frame = cap.read()
        if not success: break
        results = model.predict(frame, verbose=False)
        # You could add logic here to save frames or export data per frame
    cap.release()
