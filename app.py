# here i'm trying to update the code
import streamlit as st
import cv2
from keras.models import load_model
import numpy as np

# Load pre-trained emotion detection model
model = load_model('emotion_model.h5')  # Ensure you have the correct path

# Emotion labels based on the model, including new emotions
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral', 'Crying', 'Laughing']

# Function to detect face and predict emotion
def predict_emotion(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load OpenCV face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=8, minSize=(50, 50))

    # If no faces are detected, return empty lists
    if len(faces) == 0:
        return [], []

    predicted_emotions = []
    face_coordinates = []

    for (x, y, w, h) in faces:
        # Filter out detections that are too small or too large
        if w < 30 or h < 30:
            continue

        face_region = gray_image[y:y + h, x:x + w]
        resized_face = cv2.resize(face_region, (48, 48))
        normalized_face = resized_face / 255.0
        reshaped_face = np.reshape(normalized_face, (1, 48, 48, 1))

        predictions = model.predict(reshaped_face)
        max_index = np.argmax(predictions[0])
        predicted_emotion = emotion_labels[max_index]

        predicted_emotions.append(predicted_emotion)
        face_coordinates.append((x, y, w, h))

    return predicted_emotions, face_coordinates

# Function to describe non-facial images
def describe_non_facial_image(image):
    if "logo" in image.lower() or "graphic" in image.lower():
        return "It looks like you've uploaded a logo or graphic. No human faces detected."
    else:
        return "This appears to be a non-facial image. Please upload an image with a face for emotion detection."

# Streamlit app
st.title("Facial Emotion Detection")

# Upload an image
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)

    # Predict emotion
    emotions, faces = predict_emotion(image)

    # Display the image
    st.image(image, channels="BGR")

    # Display emotions and draw bounding boxes if faces are detected
    if emotions:
        for i, (emotion, (x, y, w, h)) in enumerate(zip(emotions, faces)):
            st.write(f"Person {i+1}: Predicted Emotion: {emotion}")
            cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
        st.image(image, channels="BGR")
    else:
        st.write("No face detected in the image.")
        st.write(describe_non_facial_image(uploaded_file.name))
