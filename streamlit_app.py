import streamlit as st
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
