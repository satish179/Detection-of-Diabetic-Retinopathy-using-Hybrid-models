import torch
import torchvision.transforms as transforms
from timm import create_model
import torch.nn.functional as F

# Human-friendly class names
class_names = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]

# Load SwinV2 model
def load_model():
    model = create_model('swinv2_small_window16_256', pretrained=False, num_classes=5)
    model.load_state_dict(torch.load("model/swinv2_small_window16_256_epoch45.pt", map_location="cpu"))
    model.eval()
    return model

# Load once globally
model = load_model()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Predict top class with confidence
def predict_image(image):
    image_tensor = transform(image).unsqueeze(0)  # shape: [1, 3, 256, 256]
    with torch.no_grad():
        outputs = model(image_tensor)  # shape: [1, 5]
        probs = F.softmax(outputs, dim=1)[0]  # shape: [5]
        top_class = torch.argmax(probs).item()
        confidence = probs[top_class].item() * 100

    return f"Prediction: {class_names[top_class]} ({confidence:.2f}% confidence)"
