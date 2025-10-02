import os
import pathlib
from typing import Union
from fastapi import FastAPI, UploadFile, File, Form
import torch
from torchvision import transforms
from PIL import Image
import io
import timm
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import numpy as np
import cv2
from starlette.responses import StreamingResponse
import base64
from google import genai
from google.genai import types
from dotenv import load_dotenv
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import base64
import json

load_dotenv()

ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
ALLOWED_ORIGIN_REGEX = os.getenv("ALLOWED_ORIGIN_REGEX") 

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS if ALLOWED_ORIGIN_REGEX is None else [],
    allow_origin_regex=ALLOWED_ORIGIN_REGEX,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_URL = os.getenv("MODEL_URL", "https://github.com/LowisWano/rice-disease-detector-server/releases/download/v1.0.0/best_model.pth")
MODEL_PATH = "best_model.pth"

def ensure_model():
  try:
    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 10_000_000:
      print(f"Downloading model from {MODEL_URL}")
      tmp_path = MODEL_PATH + ".tmp"
      pathlib.Path(tmp_path).unlink(missing_ok=True)
      
      import requests
      response = requests.get(MODEL_URL, stream=True)
      response.raise_for_status()
      
      with open(tmp_path, 'wb') as f:
          for chunk in response.iter_content(chunk_size=8192):
              f.write(chunk)
      
      os.replace(tmp_path, MODEL_PATH)
      print(f"Model downloaded successfully from GitHub Releases")
    else:
      print(f"Model already exists: {MODEL_PATH}")
  except Exception as e:
    print(f"Error downloading model: {e}")
    raise e

ensure_model()

num_classes = 6 # change this to 4 during retraining

model = timm.create_model(
    'swin_tiny_patch4_window7_224.ms_in22k_ft_in1k',
    pretrained=False,
    num_classes=num_classes,
    drop_rate=0.1
)

model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=timm.data.IMAGENET_DEFAULT_MEAN,
        std=timm.data.IMAGENET_DEFAULT_STD
    )
])
CLASS_NAMES = ["Brown spot", "Leaf Blight", "Leaf Scald", "Leaf blast", "Narrow brown spot", "healthy"]

@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Predict rice disease from uploaded image."""
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")

    x = preprocess(img).unsqueeze(0)  # (1,3,H,W)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(probs.argmax().item())
    confidence = float(probs[pred_idx].item())
    pred_class = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)

    return {
        "class_index": pred_idx,
        "class_name": pred_class,
        "confidence": confidence
    }

def reshape_transform(tensor, height=7, width=7):
    result = tensor.reshape(tensor.size(0), height, width, tensor.size(-1))
    result = result.permute(0, 3, 1, 2)
    return result


@app.post("/gradcam")
async def gradcam(file: UploadFile = File(...)):
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    x = preprocess(img).unsqueeze(0)

    target_layer = model.layers[-1].blocks[-1]

    cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)

    grayscale_cam = cam(input_tensor=x)[0]

    img_np = np.array(img.resize((224, 224))) / 255.0
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    _, buffer = cv2.imencode('.jpg', cam_image[:, :, ::-1])
    cam_bytes = buffer.tobytes()

    return StreamingResponse(io.BytesIO(cam_bytes), media_type="image/jpeg")

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    """Predict rice disease and return Grad-CAM heatmap."""
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")

    x = preprocess(img).unsqueeze(0)

    # --- Forward pass for prediction ---
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(probs.argmax().item())
    confidence = float(probs[pred_idx].item())
    pred_class = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)

    # --- Grad-CAM on same forward pass ---
    target_layer = model.layers[-1].blocks[-1]
    cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)
    grayscale_cam = cam(input_tensor=x)[0]

    img_np = np.array(img.resize((224, 224))) / 255.0
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    _, buffer = cv2.imencode('.jpg', cam_image[:, :, ::-1])
    cam_base64 = base64.b64encode(buffer).decode("utf-8")

    return {
        "class_index": pred_idx,
        "class_name": pred_class,
        "confidence": confidence,
        "gradcam_heatmap": cam_base64
    }


client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

class DiseaseIdentificationResponse(BaseModel):
    disease_identified: str
    symptoms: str
    explanation: str
    confidence: float
    gradcam_heatmap: str | None = None

# @app.post("/llm")
# async def llm(file: UploadFile = File(...)):
#     prompt = """
#       You are a rice leaf disease expert. You will be given an image of a rice leaf as input. 
#       Analyze the image and return your findings in valid JSON format only.

#       Your JSON response must include:
#       - "disease_identified": The most likely rice disease (or "Healthy" if no disease is present).
#       - "symptoms": A concise list of visible symptoms (e.g., irregular lesions, clustered dark spots, yellowing, etc.).
#       - "explanation": A detailed explanation that describes the observed symptoms and explains why they indicate the identified disease.

#       Guidelines:
#       - Always respond with valid JSON.
#       - Keep "symptoms" short and comma-separated.
#       - Ensure "explanation" is comprehensive but focused on connecting the observed symptoms to the disease.

#       Example response:
#       {
#         "disease_identified": "Bacterial Leaf Blight",
#         "symptoms": "water-soaked streaks, yellowing, wavy lesions, bacterial ooze, wilting",
#         "explanation": "The leaf shows elongated water-soaked lesions that later turn yellow and wavy. These are typical of bacterial blight caused by Xanthomonas oryzae, which disrupts water transport and leads to wilting."
#       }
#     """
#     img_bytes = await file.read()

#     response = client.models.generate_content(
#       model="gemini-2.5-flash",
#       contents=[
#           types.Part.from_text(text=prompt),
#           types.Part.from_bytes(data=img_bytes, mime_type=file.content_type or "image/png")
#       ],
#       config=types.GenerateContentConfig(
#           response_mime_type="application/json",
#           response_schema=DiseaseIdentificationResponse,
#       )
#     )


#     return response

class ImageRequest(BaseModel):
    rice_image_base64: str

@app.post("/explain-disease")
async def explain_disease(request: ImageRequest):
    # Decode base64 image
    rice_image_bytes = base64.b64decode(request.rice_image_base64)
    rice_image_rgb = Image.open(io.BytesIO(rice_image_bytes)).convert("RGB")

    x = preprocess(rice_image_rgb).unsqueeze(0)

    # --- Forward pass for prediction ---
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(probs.argmax().item())
    confidence = float(probs[pred_idx].item())
    pred_class = CLASS_NAMES[pred_idx] if pred_idx < len(CLASS_NAMES) else str(pred_idx)

    # --- Grad-CAM on same forward pass ---
    target_layer = model.layers[-1].blocks[-1]
    cam = GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform)
    grayscale_cam = cam(input_tensor=x)[0]

    img_np = np.array(rice_image_rgb.resize((224, 224))) / 255.0
    cam_image = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    _, buffer = cv2.imencode('.jpg', cam_image[:, :, ::-1])
    cam_base64 = base64.b64encode(buffer).decode("utf-8")
    cam_bytes = buffer.tobytes()

    prompt = f"""
      You are a rice leaf disease expert. You will be given an image of a rice leaf and its identified class is {pred_class}.
      Analyze the image, correlate it with the gradcam heatmap image, and return your findings in valid JSON format only.

      Your JSON response must include:
      - "disease_identified": "{pred_class}".
      - "symptoms": A concise list of visible symptoms (e.g., irregular lesions, clustered dark spots, yellowing, etc.).
      - "explanation": A detailed explanation that describes the observed symptoms and explains why they indicate the identified disease.
      - "confidence" : {confidence}

      Guidelines:
      - Always respond with valid JSON.
      - Keep "symptoms" short and comma-separated.
      - Ensure "explanation" is comprehensive but focused on connecting the observed symptoms to the disease.

      Example response:
      {{
        "disease_identified": "Bacterial Leaf Blight",
        "symptoms": "water-soaked streaks, yellowing, wavy lesions, bacterial ooze, wilting",
        "explanation": "The leaf shows elongated water-soaked lesions that later turn yellow and wavy. These are typical of bacterial blight caused by Xanthomonas oryzae, which disrupts water transport and leads to wilting.",
        "confidence": "{confidence}"
      }}
      """

    response = client.models.generate_content(
      model="gemini-2.5-flash",
      contents=[
          types.Part.from_text(text=prompt),
          types.Part.from_bytes(data=rice_image_bytes, mime_type="image/png"),
          types.Part.from_bytes(data=cam_bytes, mime_type="image/jpeg")
      ],
      config=types.GenerateContentConfig(
          response_mime_type="application/json",
          response_schema=DiseaseIdentificationResponse,
      )
    )

    if hasattr(response, "parsed") and response.parsed:
      parsed = response.parsed
      if isinstance(parsed, BaseModel):
        data = parsed.model_dump()
      else:
        data = dict(parsed)
    elif hasattr(response, "text") and response.text:
      data = json.loads(response.text)
    else:
      data = {}

    data["gradcam_heatmap"] = cam_base64

    return DiseaseIdentificationResponse(**data)
