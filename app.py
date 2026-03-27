import streamlit as st
from PIL import Image
from model_helper import predict

# Page config
st.set_page_config(page_title="Car Damage Detection", layout="wide")

# Title
st.title("🚗 Car Damage Detection")
st.markdown("Upload an image or use your webcam to detect car damage.")

st.markdown("---")

# Layout
col1, col2 = st.columns(2)

# LEFT: Upload
with col1:
    st.subheader("📂 Upload Image")

    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "png", "jpeg"]
    )

    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

        st.write("🔍 Processing...")
        prediction, confidence = predict(uploaded_file)

        st.success(f"Prediction: {prediction}")
        st.info(f"Confidence: {confidence:.2f}")
        st.progress(int(confidence * 100))


# RIGHT: Camera
with col2:
    st.subheader("📸 Camera Input")

    camera_on = st.toggle("Open Camera")

    if camera_on:
        st.success("📸 Camera is ON")

        camera_image = st.camera_input("Capture Image")

        if camera_image is not None:
            st.image(camera_image, caption="Captured Image", use_container_width=True)

            st.write("🔍 Processing...")
            prediction, confidence = predict(camera_image)

            st.success(f"Prediction: {prediction}")
            st.info(f"Confidence: {confidence:.2f}")
            st.progress(int(confidence * 100))

    else:
        st.warning("Camera is OFF")
