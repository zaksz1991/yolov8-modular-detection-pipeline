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