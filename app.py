import streamlit as st
import torch
import numpy as np
import cv2

# Streamlit app title and description
st.title("Driver Drowsiness Detection App")
st.write("Use your webcam for real-time detection with your custom YOLOv5 model.")

# Load YOLOv5 Model
@st.cache_resource
def load_model():
    # Load custom-trained YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5/runs/train/exp4/weights/last.pt', force_reload=True)
    return model

model = load_model()

# Object Detection Function
def detect_objects(image):
    # Get detections
    results = model(image)

    # Render detections on the image
    results.render()

    # Return the rendered image as a NumPy array
    return np.array(results.ims[0])

# Sidebar for Webcam Detection
st.sidebar.header("Real-Time Detection (Webcam)")
start_webcam = st.sidebar.button("Start Webcam")

if start_webcam:
    st.write("Starting webcam... Press 'Stop Webcam' to stop.")
    
    # Placeholder for video feed
    video_placeholder = st.empty()

    # Button to stop the webcam
    stop_webcam = st.sidebar.button("Stop Webcam", key="stop_webcam")
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame.")
            break

        # Perform detection
        frame = detect_objects(frame)

        # Convert BGR to RGB for Streamlit display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Update the placeholder with the latest frame
        video_placeholder.image(frame_rgb, use_container_width=True)

        # Check if 'Stop Webcam' button is clicked
        if stop_webcam:
            st.write("Stopping webcam...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Footer
st.sidebar.markdown("---")
st.sidebar.write("Developed by Shubham Darkad")
