from ultralytics import YOLO
import cv2
import csv
from datetime import datetime
import os

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  

# Input image path
input_image_path = 'img1.jpg'  

# Read the input image
image = cv2.imread(input_image_path)

# Create a CSV file and write headers if not exists
csv_file = 'person_detections.csv'
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Timestamp', 'Person_Count'])

# Detect only persons (class 0)
results = model.predict(source=image, classes=[0], conf=0.3, verbose=False)

# Count number of detected persons
detections = results[0].boxes
person_count = len(detections) if detections is not None else 0

# If any persons detected, log to CSV
if person_count > 0:
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([timestamp, person_count])

# Annotate the image
annotated_image = results[0].plot()

# Display the annotated image
cv2.imshow('YOLOv8 Person Detection', annotated_image)

# Wait for a key press to close the window
cv2.waitKey(0)
cv2.destroyAllWindows()
