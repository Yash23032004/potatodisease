import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import os

# Load the trained model
model_path = os.path.abspath("/Users/yashchaudhary/potatodisease/training/models/5.keras")  # Update to your actual model path
model = tf.keras.models.load_model(model_path)

# Define class names
class_names = ["Healthy", "Early Blight", "Late Blight"]

# Function to preprocess and predict the uploaded image
def preprocess_image(ima_path):
    img = image.load_img(ima_path, target_size=(256, 256))  # Resize the image
    img_array = image.img_to_array(img) / 255.0  # Normalize the image (same as during training)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

# Prediction function
def predict_image(model, ima_path):
    img_array = preprocess_image(ima_path)  # Preprocess the image
    prediction = model.predict(img_array)  # Predict the class
    predicted_class = np.argmax(prediction[0])  # Get the index of the highest probability
    confidence = round(100 * np.max(prediction[0]), 2)  # Get confidence in the prediction
    return class_names[predicted_class], confidence

# Streamlit UI
st.title("Potato Disease Classification")
st.write("Upload an image of a potato leaf to classify the disease.")

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# If the file is uploaded, process and predict
if uploaded_file is not None:
    img = Image.open(uploaded_file)  # Open the uploaded image
    st.image(img, caption="Uploaded Image", use_column_width=True)  # Display the image
    
    # Save the uploaded image temporarily to disk
    ima_path = "temp_image.jpg"
    img.save(ima_path)
    
    # Get prediction
    predicted_class, confidence = predict_image(model, ima_path)
    
    # Display the predicted class and confidence
    st.write(f"Predicted class: {predicted_class}")
    st.write(f"Confidence: {confidence}%")
