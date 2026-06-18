import cv2
import json

def process_media(model, source_path, is_video=False):
    if is_video:
        cap = cv2.VideoCapture(source_path)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            results = model.predict(frame, verbose=False)
            # Visualize or save frame data here
            cv2.imshow("Detection", results[0].plot())
            if cv2.waitKey(1) & 0xFF == ord('q'): break
        cap.release()
        cv2.destroyAllWindows()
    else:
        results = model.predict(source_path)
        # Handle single image as before...