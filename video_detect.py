from ultralytics import YOLO
import cv2
import csv
from datetime import datetime
import os

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # You can change to yolov8s.pt, etc.

# Path to input video file
video_path = 'input.mp4'  # Replace with the path to your video file
cap = cv2.VideoCapture(video_path)

# Create a CSV file and write headers if not exists
csv_file = 'person_detections.csv'
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Person_Count'])

# Start reading frames from the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect only persons (class 0)
    results = model.predict(source=frame, classes=[0], conf=0.3, verbose=False)

    # Count number of detected persons
    detections = results[0].boxes
    person_count = len(detections) if detections is not None else 0

    # If any persons detected, log to CSV
    if person_count > 0:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, person_count])

    # Annotate and display the frame
    annotated_frame = results[0].plot()
    cv2.imshow('YOLOv8 Person Detection - Video', annotated_frame)

    # Press 'q' to stop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
