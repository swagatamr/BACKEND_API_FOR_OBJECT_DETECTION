import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import json
from datetime import datetime
import imghdr

# Set page config
st.set_page_config(
    page_title="Object Detection",
    page_icon="🔍",
    layout="wide"
)

# Create output directory if it doesn't exist
OUTPUT_DIR = "detection_results"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Title and description
st.title("Object Detection")
st.write("Upload an image to detect objects using YOLOv8 model")

# Initialize YOLO model
@st.cache_resource
def load_model():
    return YOLO('yolov8l.pt')

model = load_model()

def get_file_extension(file):
    """
    Get the correct file extension based on the image type
    """
    if file is None:
        return None
    
    # First try to get the extension from the filename
    original_extension = os.path.splitext(file.name)[1].lower()
    if original_extension in ['.jpg', '.jpeg', '.png']:
        return original_extension
    
    # If no extension or unknown, detect the image type
    file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
    file.seek(0)  # Reset file pointer
    img_type = imghdr.what(None, file_bytes)
    
    extension_map = {
        'jpeg': '.jpg',
        'jpg': '.jpg',
        'png': '.png'
    }
    
    return extension_map.get(img_type, '.jpg')  # Default to jpg if unknown

# File uploader with multiple image formats
uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

# Confidence threshold slider
confidence = st.slider('Confidence threshold', min_value=0.0, max_value=1.0, value=0.25)

def process_image(image, conf):
    """
    Process image with YOLOv8 model and return annotated image and results
    """
    # Convert PIL Image to BGR format (which OpenCV uses)
    if isinstance(image, np.ndarray):
        # If image is already a numpy array, convert from RGB to BGR
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        # If image is a PIL Image, convert to numpy array and then to BGR
        image_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Perform prediction
    results = model(image_bgr)
    
    # Get the annotated image (it will be in BGR format)
    annotated_image = results[0].plot()
    
    # Convert back to RGB for displaying
    annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    
    return annotated_image_rgb, results[0]

def create_detection_json(results, original_image_path, annotated_image_path):
    """
    Create JSON file with detection results
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
    
    detection_data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "original_image": original_image_path,
        "annotated_image": annotated_image_path,
        "confidence_threshold": float(confidence),
        "detections": detections,
        "total_objects": len(detections)
    }
    
    return detection_data

def save_results(original_image, annotated_image, results, file_extension):
    """
    Save original image, annotated image, and JSON results with proper extension
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    
    # Create paths with proper extensions
    original_image_path = os.path.join(OUTPUT_DIR, f"original_{timestamp}{file_extension}")
    annotated_image_path = os.path.join(OUTPUT_DIR, f"annotated_{timestamp}{file_extension}")
    json_path = os.path.join(OUTPUT_DIR, f"detections_{timestamp}.json")
    
    # Save original image in its original format
    if isinstance(original_image, np.ndarray):
        Image.fromarray(original_image).save(original_image_path, quality=95, optimize=True)
    else:
        original_image.save(original_image_path, quality=95, optimize=True)
    
    # Save annotated image in the same format as original
    Image.fromarray(annotated_image).save(annotated_image_path, quality=95, optimize=True)
    
    # Create and save JSON
    detection_data = create_detection_json(results, 
                                        os.path.basename(original_image_path),
                                        os.path.basename(annotated_image_path))
    
    with open(json_path, 'w') as f:
        json.dump(detection_data, f, indent=4)
    
    return original_image_path, annotated_image_path, json_path

def get_mime_type(file_extension):
    """
    Get the correct MIME type based on file extension
    """
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png'
    }
    return mime_types.get(file_extension, 'image/jpeg')

def display_detection_results(results):
    """
    Display detection results in a formatted way
    """
    if len(results.boxes) > 0:
        st.subheader("Detection Results")
        
        # Create columns for different metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Objects Detected:**")
            # Count occurrences of each class
            classes = results.boxes.cls.cpu().numpy()
            names = results.names
            unique_classes, counts = np.unique(classes, return_counts=True)
            for cls, count in zip(unique_classes, counts):
                st.write(f"- {names[int(cls)]}: {count}")
        
        with col2:
            st.write("**Confidence Scores:**")
            # Display confidence scores
            confidences = results.boxes.conf.cpu().numpy()
            st.write(f"- Average: {confidences.mean():.2f}")
            st.write(f"- Maximum: {confidences.max():.2f}")
            st.write(f"- Minimum: {confidences.min():.2f}")
        
        with col3:
            st.write("**Bounding Boxes:**")
            st.write(f"- Total: {len(results.boxes)}")

if uploaded_file is not None:
    try:
        # Get file extension
        file_extension = get_file_extension(uploaded_file)
        
        # Read image
        image = Image.open(uploaded_file)
        image = image.convert('RGB')  # Ensure image is in RGB format
        
        # Convert PIL Image to numpy array
        image_np = np.array(image)
        
        # Process image
        processed_image, results = process_image(image_np, confidence)
        
        # Save results with proper extension
        original_path, annotated_path, json_path = save_results(
            image_np, 
            processed_image, 
            results,
            file_extension
        )
        
        # Get correct MIME type
        mime_type = get_mime_type(file_extension)
        
        # Display images side by side
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image_np, use_column_width=True)
            
            # Add download button for original image
            with open(original_path, 'rb') as file:
                st.download_button(
                    label="Download original image",
                    data=file,
                    file_name=os.path.basename(original_path),
                    mime=mime_type
                )
        
        with col2:
            st.subheader("Detected Objects")
            st.image(processed_image, use_column_width=True)
            
            # Add download button for processed image
            with open(annotated_path, 'rb') as file:
                st.download_button(
                    label="Download annotated image",
                    data=file,
                    file_name=os.path.basename(annotated_path),
                    mime=mime_type
                )
        
        # Display detection results
        display_detection_results(results)
        
        # Add download button for JSON results
        with open(json_path, 'r') as file:
            st.download_button(
                label="Download detection results (JSON)",
                data=file,
                file_name=os.path.basename(json_path),
                mime="application/json"
            )

        # Display JSON content in expandable section
        with st.expander("View Detection Results JSON"):
            with open(json_path, 'r') as file:
                st.json(json.load(file))

    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        st.write("Please try uploading a different image.")

else:
    st.info("Please upload an image to begin object detection")