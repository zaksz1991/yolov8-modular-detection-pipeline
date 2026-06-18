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