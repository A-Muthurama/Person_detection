# Person_detection

👁️ YOLOv8 Person Detection

This project implements real-time person detection using the **YOLOv8 (You Only Look Once)** object detection model developed by Ultralytics. The system detects and counts people from either **image files** or **video streams**, and logs the detection results (timestamp and person count) into a CSV file.

The project is ideal for basic surveillance, analytics, or people-counting systems using image or video input.

---

## 📌 Features

- 🎯 Detects only **persons** (COCO class ID 0)
- 🖼️ Works with both **image** and **video** input
- 📝 Logs timestamp and count of people detected into a CSV file
- 📊 Visualizes results by displaying bounding boxes on detected persons
- 🔧 Easy to switch between YOLOv8 model variants (`yolov8n.pt`, `yolov8s.pt`, etc.)

---

## 🔍 How It Works

1. Load a pretrained YOLOv8 model from Ultralytics.
2. Take input from an **image** or a **video file**.
3. Run inference and filter detections for the `person` class (ID `0`).
4. Count how many people are detected.
5. Save the count along with the timestamp to a CSV log file.
6. Display the results with bounding boxes drawn around detected people.

---

## 🧠 Model Used

- **YOLOv8n** (`yolov8n.pt`) — lightweight and fast
- You can swap in other YOLOv8 variants:
  - `yolov8s.pt` (small)
  - `yolov8m.pt` (medium)
  - `yolov8l.pt` (large)
  - `yolov8x.pt` (extra large)
