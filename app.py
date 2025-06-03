import streamlit as st
import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
from utils import analyze_face, draw_predictions

# Set page config
st.set_page_config(
    page_title="Face Age, Gender & Emotion Detection",
    page_icon="👤",
    layout="centered"
)


def process_image(image):
    # Convert PIL Image to OpenCV BGR format
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    # Detect faces using MTCNN (or change to 'retinaface', 'opencv', 'ssd', etc.)
    detections = DeepFace.extract_faces(
        img_path=image_bgr, detector_backend='opencv', enforce_detection=False)

    if not detections:
        st.warning("No faces detected in the image!")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Process each detected face
    for det in detections:
        region = det['facial_area']
        x, y, w, h = region['x'], region['y'], region['w'], region['h']
        face_img = image_bgr[y:y + h, x:x + w]

        # Analyze face
        age, gender, emotion = analyze_face(face_img)

        # Draw predictions
        face_info = {'box': [x, y, w, h]}
        image_bgr = draw_predictions(
            image_bgr, face_info, age, gender, emotion)

    # Convert back to RGB for Streamlit display
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def main():
    st.title("👤 Face Age, Gender & Emotion Detector")
    st.write("Upload an image and let AI analyze the faces!")

    uploaded_file = st.file_uploader(
        "Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=500)

        if st.button("Analyze Image"):
            result_image = process_image(image)
            st.image(result_image, caption="Analysis Result",
                     width=500)

    st.markdown("---")
    st.markdown("""
    ### About
    This app uses **DeepFace** to detect:
    - Age
    - Gender
    - Emotion

    ### How to Use
    1. Upload an image with visible faces.
    2. Click **Analyze Image**.
    3. See the detected age, gender, and dominant emotion for each face!

    **Note**: The first run might take longer as it loads models.
    """)


if __name__ == "__main__":
    main()
