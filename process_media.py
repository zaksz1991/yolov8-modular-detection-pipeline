import json
import cv2
from ultralytics import YOLO

def export_to_json(results, output_path='detections.json'):
    """Explicitly converts YOLO results to a structured JSON file."""
    data = []
    for r in results:
        for box in r.boxes:
            data.append({
                "class_id": int(box.cls),
                "confidence": float(box.conf),
                "bbox": box.xyxy.tolist()[0]
            })
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Successfully saved {len(data)} detections to {output_path}")

def process_media(model, source, is_video=False):
    """Handles both image detection and video processing loop."""
    if is_video:
        cap = cv2.VideoCapture(source)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            results = model.predict(frame, verbose=False)
            # Visualize detection
            cv2.imshow("Detection", results[0].plot())
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()
    else:
        results = model.predict(source, verbose=False)
        export_to_json(results)
        return results