from typing import Union

from fastapi import FastAPI, UploadFile, File
import torch
from torchvision import transforms
from PIL import Image
import io

app = FastAPI()

model = torch.jit.load("model_scripted.pt", map_location="cpu")
model.eval()
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Load uploaded image
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")

    # Apply transforms
    x = preprocess(img).unsqueeze(0)  # shape (1,3,H,W)

    # Inference
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]
        pred_class = int(probs.argmax().item())
        confidence = float(probs[pred_class].item())

    return {"class": pred_class, "confidence": confidence}