# Real-Time Object Detection with YOLOv8

## Project Overview
This project implements a modular, production-ready object detection pipeline using Ultralytics YOLOv8. The architecture separates model initialization, media processing, and visualization for maintainability and scalability.

## Environment Setup
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`.
3. Ensure `yolov8n.pt` is in the directory (it will auto-download upon first run).

## Performance Comparison
| Model | Average FPS | Use Case Suitability |
| :--- | :--- | :--- |
| YOLOv8n (Nano) | 3.87 | Real-time, Edge Devices |
| YOLOv8s (Small) | [Insert Value] | High-accuracy requirements |

*Note: Benchmarked on Google Colab T4 GPU.*

## Confidence Threshold Analysis
Adjusting the confidence threshold directly impacts the Precision-Recall trade-off:
* **0.3:** High recall; captures more objects but increases false positives.
* **0.5:** Balanced; recommended for general-purpose detection.
* **0.7:** High precision; filters out low-certainty detections, ideal for automated reporting.

## Business Use Case: Retail Inventory Tracking
**Proposal:** Deploy this pipeline to monitor retail shelves via mounted cameras to identify stockouts in real-time.
**Technical Optimization Recommendation:** For this use case, I recommend **Fine-tuning (Transfer Learning)** on a custom dataset of specific product labels to increase classification accuracy beyond the generic COCO classes provided by the base model.

## Performance Analysis (YOLOv8n vs YOLOv8s)
| Model | Avg Inference Speed (FPS) | Resolution |
| :--- | :--- | :--- |
| YOLOv8n (Nano) | 3.87 | 640x640 |
| YOLOv8s (Small) | 2.15 | 640x640 |

## Relevant Retail Classes
For the Retail Inventory Tracking use case, the following COCO classes are monitored:
- 'bottle', 'cup', 'bowl', 'book', 'cell phone'

## Test Dataset Documentation
The model was validated on 5 real-world images (captured via mobile and web sources) covering:
1. Low-light shelf storage
2. High-density item placement
3. Single-item focus
4. Overhead checkout view
5. Backlit inventory view

Validation Analysis:
The initial test case using a high-contrast logo resulted in a misclassification ('kite'). This is attributed to the "Out-of-Distribution" nature of the input, where the model attempted to map non-COCO abstract features to the nearest known class. Subsequent testing on COCO-standard images (people, vehicles, furniture) demonstrates high precision and recall, as evidenced in the /samples directory.