import matplotlib.pyplot as plt
import cv2

def display_results(image_path, results):
    # Ultralytics built-in plotting
    img = results[0].plot()
    # Convert BGR to RGB for matplotlib
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
import cv2

def draw_custom_boxes(image_path, results):
    img = cv2.imread(image_path)
    for box in results[0].boxes:
        # Get coordinates
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Put label
        label = f"{results[0].names[int(box.cls)]} {float(box.conf):.2f}"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imwrite('custom_viz.jpg', img)