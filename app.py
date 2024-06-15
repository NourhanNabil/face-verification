from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import base64
import requests

app = FastAPI()

# Load the ONNX model
onnx_model_path = 'siamese_model.onnx'
ort_session = ort.InferenceSession(onnx_model_path)

class ImageInput(BaseModel):
    image: str

def preprocess_image(image: Image.Image, target_size=(128, 128)) -> np.ndarray:
    image = image.resize(target_size)
    image = np.array(image).astype(np.float32) / 255.0
    if image.shape[2] == 1:  # Convert grayscale to RGB
        image = np.concatenate([image] * 3, axis=2)
    return image

def load_image_from_url(url: str) -> Image.Image:
    response = requests.get(url)
    image = Image.open(io.BytesIO(response.content))
    return image

def load_image_from_base64(base64_string: str) -> Image.Image:
    decoded_image = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(decoded_image))
    return image

def predict(image1: np.ndarray, image2: np.ndarray) -> float:
    inputs = {ort_session.get_inputs()[0].name: image1, ort_session.get_inputs()[1].name: image2}
    outputs = ort_session.run(None, inputs)
    return outputs[0][0][0]

@app.post("/predict")
async def predict_images(images: List[ImageInput]):
    if len(images) != 2:
        return {"error": "Please provide exactly two images."}

    image1_type = "url" if images[0].image.startswith("http") else "base64"
    image2_type = "url" if images[1].image.startswith("http") else "base64"

    if image1_type == "url":
        image1 = load_image_from_url(images[0].image)
    else:
        image1 = load_image_from_base64(images[0].image)

    if image2_type == "url":
        image2 = load_image_from_url(images[1].image)
    else:
        image2 = load_image_from_base64(images[1].image)

    image1 = preprocess_image(image1)
    image2 = preprocess_image(image2)

    image1 = np.expand_dims(image1, axis=0)  # Add batch dimension
    image2 = np.expand_dims(image2, axis=0)  # Add batch dimension

    score = predict(image1, image2)
    
    # Convert numpy bool_ to native Python bool
    match = bool(score > 0.5)
    
    return {"score": float(score), "match": match}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
