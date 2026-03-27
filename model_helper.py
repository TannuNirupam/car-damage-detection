from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import streamlit as st

class_names = [
    'Front Breakage',
    'Front Crushed',
    'Front Normal',
    'Rear Breakage',
    'Rear Crushed',
    'Rear Normal'
]

class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=len(class_names), dropout_rate=0.5):
        super(CarClassifierResNet, self).__init__()

        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True

        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


@st.cache_resource
def load_model():
    model = CarClassifierResNet(num_classes=len(class_names))
    model.load_state_dict(torch.load("saved_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model


def predict(image_input):

    if isinstance(image_input, Image.Image):
        image = image_input.convert("RGB")
    else:
        image = Image.open(image_input).convert("RGB")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    image_tensor = transform(image).unsqueeze(0)

    model = load_model()

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, predicted_class = torch.max(probs, 1)

        return class_names[predicted_class.item()], confidence.item()