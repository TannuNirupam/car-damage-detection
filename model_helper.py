from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# Define all class names
class_names = [
    'Front Breakage',
    'Front Crushed',
    'Front Normal',
    'Rear Breakage',
    'Rear Crushed',
    'Rear Normal'
]

# Define the model first — before creating an instance
class CarClassifierResNet(nn.Module):
    def __init__(self, num_classes=len(class_names), dropout_rate=0.5):
        super(CarClassifierResNet, self).__init__()

        # Load pretrained ResNet50
        self.model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # Freeze all layers first
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze layer4 for fine-tuning
        for param in self.model.layer4.parameters():
            param.requires_grad = True

        # Replace the final FC layer
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(self.model.fc.in_features, num_classes)
        )

    def forward(self, x):
        return self.model(x)


# Define prediction function AFTER the model definition
def predict(image_path, model_path="/Users/tannunirupam/Downloads/Download files 2/model/saved_model.pth"):
    # Image preprocessing
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    image_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # Load trained model
    model = CarClassifierResNet(num_classes=len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted_class = torch.max(outputs, 1)
        return class_names[predicted_class.item()]
