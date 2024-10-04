"""import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
st.title("✌️ CN LAB PROJECT ✌️ ")
st.write(
    "This was the cnn project for image classification of minst dataset"
)



# Load the saved model
#model = tf.keras.models.load_model('mnist_model.h5')
model = tf.keras.models.load_model('my_model.keras')
st.title("MNIST Digit Recognizer")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Preprocess the image
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to MNIST input size
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    # Make prediction
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)

    st.write(f"Predicted Digit: {predicted_digit}")
"""
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Set the page configuration
st.set_page_config(page_title="MNIST Digit Recognizer", layout="wide")

# Load the saved model
model = tf.keras.models.load_model('my_model.keras')

# Custom CSS for background and styling
st.markdown(
    """
    <style>
    .main {
        background-color: #99d6ff;
    }
    .sidebar .sidebar-content {
        background-color: #0099ff;
    }
    .uploaded-image {
        border: 2px solid #4CAF50;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and Description
st.title("✌️ CNN LAB PROJECT ✌️")
st.subheader("MNIST Digit Recognizer")
st.write("This project classifies handwritten digits from the MNIST dataset using a Convolutional Neural Network (CNN). Upload an image, and the model will predict the digit!")

# File uploader with enhanced styling
st.sidebar.title("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)  # Fixed here

    # Preprocess the image
    image = image.convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to MNIST input size
    image_array = np.array(image)
    image_array = image_array.astype('float32') / 255.0
    image_array = image_array.reshape(1, 28, 28, 1)

    # Make prediction
    prediction = model.predict(image_array)
    predicted_digit = np.argmax(prediction)

    # Display the predicted digit
    st.success(f"Predicted Digit: **{predicted_digit}**")

    # Optional: Display confidence scores
    confidence = np.max(prediction)
    st.write(f"Confidence: {confidence:.2f}")

    # Add a button to clear the input
    if st.button("Clear Image"):
        uploaded_file = None
        st.experimental_rerun()

# Add footer for more information
st.markdown("---")
st.write("Built with ❤️ using Streamlit and TensorFlow.")
