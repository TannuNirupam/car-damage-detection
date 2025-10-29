import streamlit as st
from PIL import Image
from model_helper import predict

st.title("Car Damage Detection")

# Display your app screenshot
try:
    image = Image.open("app_screenshot.jpg")
    st.image(image, caption="App Screenshot", use_container_width=True)
except Exception as e:
    st.warning(f"Could not load app_screenshot.jpg — {e}")

# Upload user image for prediction
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
    st.write("Processing...")
    prediction = predict(uploaded_file)
    st.success(f"Prediction: {prediction}")

