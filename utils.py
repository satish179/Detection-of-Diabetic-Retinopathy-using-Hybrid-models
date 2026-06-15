import os
import torch
import torchvision.transforms as transforms
from timm import create_model
import torch.nn.functional as F

class_names = [
    "No Diabetic Retinopathy",
    "Mild Diabetic Retinopathy",
    "Moderate Diabetic Retinopathy",
    "Severe Diabetic Retinopathy",
    "Proliferative Diabetic Retinopathy"
]

def load_model():
    model_path = os.getenv("MODEL_PATH", "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found at '{model_path}'. Set MODEL_PATH or add model.pt."
        )
    model = create_model('swinv2_small_window16_256', pretrained=False, num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = None
model_error = None

def get_model():
    global model, model_error
    if model is None and model_error is None:
        try:
            model = load_model()
        except Exception as exc:
            model_error = str(exc)
    return model

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Predict top class with confidence
def predict_image(image):
    loaded_model = get_model()
    if loaded_model is None:
        return f"Prediction unavailable: {model_error}"

    image_tensor = transform(image).unsqueeze(0)  # shape: [1, 3, 256, 256]
    with torch.no_grad():
        outputs = loaded_model(image_tensor)  # shape: [1, 5]
        probs = F.softmax(outputs, dim=1)[0]  # shape: [5]
        top_class = torch.argmax(probs).item()
        confidence = probs[top_class].item() * 100

    return f"Prediction: {class_names[top_class]} ({confidence:.2f}% confidence)"
