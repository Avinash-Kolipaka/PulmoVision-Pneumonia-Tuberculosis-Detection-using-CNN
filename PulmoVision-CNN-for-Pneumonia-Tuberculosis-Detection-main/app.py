import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os
import gdown

# Download model if not exists
MODEL_PATH = "xray_cnn.pth"
GOOGLE_DRIVE_URL = "https://drive.google.com/uc?id=1UjlesGtxpdLsdcLzjU84DCnUdiICJ3c1"

if not os.path.exists(MODEL_PATH):
    with st.spinner("Downloading model..."):
        gdown.download(GOOGLE_DRIVE_URL, MODEL_PATH, quiet=False)

# Define CNN model
class XRayCNN(nn.Module):
    def __init__(self, num_classes):
        super(XRayCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128*28*28, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classes = ["Normal", "Pneumonia", "Tuberculosis"]

model = XRayCNN(num_classes=len(classes))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

def predict_image(image):
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image_tensor)
        _, pred = torch.max(output, 1)
    return classes[pred.item()]

# Streamlit UI
st.set_page_config(page_title="PulmoVision - XRay Diagnosis", layout="wide")

st.title("🩺 PulmoVision: Tuberculosis & Pneumonia Detection")
st.markdown("Upload a chest X-ray image and let our CNN model predict whether it's **Normal, Pneumonia, or Tuberculosis**.")

uploaded_file = st.file_uploader("📂 Upload Chest X-Ray", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded X-Ray", use_container_width=True)

    if st.button("🔍 Predict"):
        prediction = predict_image(image)
        st.success(f"✅ The model predicts: **{prediction}**")
        st.markdown("---", unsafe_allow_html=True)
st.markdown(
   "<p style='text-align: center; font-size:14px;'> Made with 💡 by <b>Avinash</b></p>",
    unsafe_allow_html=True
)