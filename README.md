🚗 Car Damage Detection App

A simple and interactive Streamlit web app that classifies car images into different damage categories using a trained PyTorch deep learning model (ResNet50).

🧠 Project Overview

This project uses a pretrained ResNet50 model fine-tuned to detect car damage from images.
The model analyzes uploaded or captured images and predicts one of several damage categories in real-time.

📁 Project Structure
streamlit-app/
│
├── app.py                # Streamlit web application
├── model_helper.py       # Model loading and prediction logic
├── saved_model.pth       # Trained PyTorch model (not included in repo)
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
⚙️ Setup Instructions
1️⃣ Clone the repository
git clone https://github.com/TannuNirupam/car-damage-detection.git
cd streamlit-app
2️⃣ Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
3️⃣ Install dependencies
pip install -r requirements.txt
🚀 Run the App
streamlit run app.py

Then open:

http://localhost:8501
🖼️ Features

✅ Image upload prediction
✅ Webcam capture prediction
✅ Real-time inference
✅ Confidence score display
✅ Clean UI using Streamlit

🧩 Model Details
Architecture: ResNet50 (Transfer Learning)
Framework: PyTorch
Input Size: 224 × 224 RGB image
Classes:
Front Breakage
Front Crushed
Front Normal
Rear Breakage
Rear Crushed
Rear Normal
🧪 Example

Example Output:

Prediction: Front Crushed
Confidence: 0.87
⚠️ Note
The trained model file (saved_model.pth) is not included due to GitHub file size limits.
To run the app, place the model file in the same directory as app.py.
🧰 Troubleshooting
Model not found: Ensure saved_model.pth is in the project folder
Module error: Activate virtual environment and reinstall dependencies
Streamlit issue: Restart using streamlit run app.py
✨ Future Improvements
Add Grad-CAM visualization (model explainability)
Improve dataset for better accuracy
Deploy app on Streamlit Cloud
Convert to mobile/web product
🧑‍💻 Author

Tannu Nirupam
🔗 https://github.com/TannuNirupam

