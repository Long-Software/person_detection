from pathlib import Path
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2
import os

# Load the YOLOv5 model
model = torch.hub.load('yolov5', 'custom', path='./yolov5/runs/train/exp2/weights/last.pt', force_reload=True, source='local') 

def load_image_into_numpy_array(image_path):
    return np.array(Image.open(image_path))

def detect_objects(image_np):
    results = model(image_np)
    return results.xyxy[0].numpy()

def draw_boxes(image_path, detections, threshold=0.5):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)


    font_path = "arial.ttf"
    font_size = 40
    font = ImageFont.truetype(font_path, font_size)
    
    for det in detections:
        if det[4] >= threshold:
            xmin, ymin, xmax, ymax, conf, cls = det
            class_name = model.names[int(cls)]
            
            draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="green", width=2)
            draw.text((xmin, ymin), f"{class_name} ({conf:.2f})", fill="green", font=font)


    boxed_image_path = image_path.replace('.jpg', '_boxed.jpg').replace('.png', '_boxed.png')
    image.save(boxed_image_path)
    return boxed_image_path

def process_video(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

    all_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        detections = detect_objects(frame_rgb)
        all_detections.extend(detections)

        frame_with_boxes = draw_boxes_cv2(frame, detections)
        out.write(frame_with_boxes)

    cap.release()
    out.release()

    detected_classes = [model.names[int(det[5])] for det in all_detections if det[4] > 0.5]
    return output_path, detected_classes

def draw_boxes_cv2(frame, detections, threshold=0.5):
    for det in detections:
        if det[4] >= threshold:
            xmin, ymin, xmax, ymax, conf, cls = det
            class_name = model.names[int(cls)]
            
            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2)
            cv2.putText(frame, f"{class_name} ({conf:.2f})", (int(xmin), int(ymin) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    return frame