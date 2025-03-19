import os
os.system('pip install ultralytics')
from ultralytics import YOLO

import streamlit as st
from PIL import Image
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the trained YOLO model from the models folder
MODEL_PATH = 'best.torchscript'
model = YOLO(MODEL_PATH)

# Streamlit App Title
st.title("BCCD Detection App")
st.write("Upload an image to detect blood cells")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Load the uploaded image using PIL
    image = Image.open(uploaded_file)
    
    # Display the uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL image to OpenCV format
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Perform inference using YOLO model
    results = model(image_cv)

    # Draw the bounding boxes on the image
    annotated_image = results[0].plot()

    # Convert OpenCV BGR to RGB format (for Streamlit display)
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # Display the result with bounding boxes
    st.image(annotated_image, caption="Detected Objects", use_column_width=True)

    # Show detailed prediction results
    st.write("### Prediction Results:")
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        st.write(f"- **Class:** {model.names[class_id]}")
        st.write(f"- **Confidence:** {confidence:.2f}")
        st.write(f"- **Bounding Box:** ({x1}, {y1}), ({x2}, {y2})")
        st.write("---")

# Footer
st.write("Developed by [Your Name]")
