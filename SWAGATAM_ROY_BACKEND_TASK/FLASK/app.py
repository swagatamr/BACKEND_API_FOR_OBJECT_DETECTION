# app.py
from flask import Flask, render_template, request, jsonify, send_file
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime
import io
import base64

app = Flask(__name__)

# Create directories for saving results
UPLOAD_FOLDER = 'static/uploads'
DETECTION_FOLDER = 'static/detections'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTION_FOLDER, exist_ok=True)

# Initialize YOLO model
model = YOLO('yolov8l.pt')

def process_image(image_path, confidence):
    """
    Process image with YOLOv8 model and return results
    """
    # Read image
    image = cv2.imread(image_path)
    
    # Run inference
    results = model(image, conf=confidence)
    
    # Get the annotated image
    annotated_image = results[0].plot()
    
    return annotated_image, results[0]

def create_detection_json(results, original_path, annotated_path):
    """
    Create JSON with detection results
    """
    detections = []
    
    # Get boxes, confidence scores, and class IDs
    boxes = results.boxes.xyxy.cpu().numpy()
    confidences = results.boxes.conf.cpu().numpy()
    class_ids = results.boxes.cls.cpu().numpy()
    
    for i in range(len(boxes)):
        detection = {
            "class": results.names[int(class_ids[i])],
            "confidence": float(confidences[i]),
            "bbox": {
                "x1": float(boxes[i][0]),
                "y1": float(boxes[i][1]),
                "x2": float(boxes[i][2]),
                "y2": float(boxes[i][3])
            }
        }
        detections.append(detection)
    
    return {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "original_image": original_path,
        "annotated_image": annotated_path,
        "detections": detections,
        "total_objects": len(detections)
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    try:
        # Check if image was uploaded
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No image selected'}), 400
        
        # Get confidence threshold
        confidence = float(request.form.get('confidence', 0.25))
        
        # Save uploaded image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = f"original_{timestamp}.jpg"
        annotated_filename = f"annotated_{timestamp}.jpg"
        json_filename = f"detections_{timestamp}.json"
        
        original_path = os.path.join(UPLOAD_FOLDER, original_filename)
        annotated_path = os.path.join(DETECTION_FOLDER, annotated_filename)
        json_path = os.path.join(DETECTION_FOLDER, json_filename)
        
        # Save original image
        file.save(original_path)
        
        # Process image
        annotated_image, results = process_image(original_path, confidence)
        
        # Save annotated image
        cv2.imwrite(annotated_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        
        # Create and save JSON results
        detection_data = create_detection_json(results, original_filename, annotated_filename)
        with open(json_path, 'w') as f:
            json.dump(detection_data, f, indent=4)
        
        # Convert images to base64 for display
        with open(original_path, 'rb') as f:
            original_base64 = base64.b64encode(f.read()).decode()
        
        with open(annotated_path, 'rb') as f:
            annotated_base64 = base64.b64encode(f.read()).decode()
        
        return jsonify({
            'original_image': original_base64,
            'annotated_image': annotated_base64,
            'detection_data': detection_data,
            'original_filename': original_filename,
            'annotated_filename': annotated_filename,
            'json_filename': json_filename
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download/<filename>')
def download_file(filename):
    """Download detection results"""
    try:
        if filename.startswith('original_'):
            return send_file(os.path.join(UPLOAD_FOLDER, filename))
        else:
            return send_file(os.path.join(DETECTION_FOLDER, filename))
    except Exception as e:
        return jsonify({'error': str(e)}), 404

if __name__ == '__main__':
    app.run(debug=True)