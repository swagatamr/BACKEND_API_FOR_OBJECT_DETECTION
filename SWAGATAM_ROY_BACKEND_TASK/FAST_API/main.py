# main.py
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime
import io
import base64
from starlette.requests import Request
from typing import Optional

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create directories for saving results
UPLOAD_FOLDER = 'static/uploads'
DETECTION_FOLDER = 'static/detections'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(DETECTION_FOLDER, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize templates
templates = Jinja2Templates(directory="templates")

# Initialize YOLO model
model = YOLO('yolov8l.pt')

def process_image(image_path: str, confidence: float):
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

def create_detection_json(results, original_path: str, annotated_path: str):
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

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Render the index page"""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    confidence: float = Form(0.25)
):
    try:
        # Save uploaded image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        original_filename = f"original_{timestamp}.jpg"
        annotated_filename = f"annotated_{timestamp}.jpg"
        json_filename = f"detections_{timestamp}.json"
        
        original_path = os.path.join(UPLOAD_FOLDER, original_filename)
        annotated_path = os.path.join(DETECTION_FOLDER, annotated_filename)
        json_path = os.path.join(DETECTION_FOLDER, json_filename)
        
        # Save original image
        with open(original_path, "wb") as buffer:
            buffer.write(await image.read())
        
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
        
        return JSONResponse({
            'original_image': original_base64,
            'annotated_image': annotated_base64,
            'detection_data': detection_data,
            'original_filename': original_filename,
            'annotated_filename': annotated_filename,
            'json_filename': json_filename
        })
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={'error': str(e)}
        )

@app.get("/download/{filename}")
async def download_file(filename: str):
    """Download detection results"""
    try:
        if filename.startswith('original_'):
            return FileResponse(
                os.path.join(UPLOAD_FOLDER, filename),
                media_type='image/jpeg',
                filename=filename
            )
        else:
            return FileResponse(
                os.path.join(DETECTION_FOLDER, filename),
                filename=filename
            )
    except Exception as e:
        return JSONResponse(
            status_code=404,
            content={'error': str(e)}
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)