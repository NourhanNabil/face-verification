from fastapi import FastAPI , HTTPException
import uvicorn
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
    image = np.array(image)  
    if image.shape != (128, 128, 3): 
        image = np.stack([image] * 3, axis=-1)   # Convert grayscale to RGB by duplicating channels
    image = image.astype(np.float32) / 255.0     # Normalize pixel values 
    return image


def load_image_from_url(url: str) -> Image.Image:
    try:
        response = requests.get(url)
        image = Image.open(io.BytesIO(response.content))
        # Check if image format is supported
        if image.format.lower() not in ['jpg', 'jpeg', 'png', 'jfif']:
            raise HTTPException(status_code=415, detail="Unsupported image format. Please provide JPG, JPEG, PNG, or JFIF.")
        return image
    except IOError as e:
        raise HTTPException(status_code=400, detail="Error loading image from URL.")

def predict(image1: np.ndarray, image2: np.ndarray) -> float:
    # Create inputs dictionary for the ONNX model
    inputs = {
        ort_session.get_inputs()[0].name: image1,
        ort_session.get_inputs()[1].name: image2   
    }
    outputs = ort_session.run(None, inputs)  # None  means it will return all outputs produced by the model.
        # output looks like this [array([[0.99657184]], dtype=float32)]
    return float(outputs[0][0][0])   # Convert return numpy float32 to Python float


@app.get("/")
async def root():
    return {
        "message": "Welcome to the Face Verification API",
        "endpoints": {
            "/predict": {
                "description": "Endpoint for face verification prediction",
                "method": "POST",
                "payload_example": 
                     [
                        {"image": "url_to_image1"},
                        {"image": "url_to_image2"}
                    ]
                    ,
                "expected_response": {
                    "score": "float (0 to 1)",
                    "match": "bool"
                },
                "purpose": "Verifies if two faces in the provided images are the same person."
            }
        }
    }


@app.post("/predict")
async def predict_images(images: List[ImageInput]):
    if len(images) != 2:
        raise HTTPException(status_code=400, detail="Please provide exactly two images.")

    image1_url = images[0].image
    image2_url = images[1].image

    # Check if image1_url and image2_url are valid URLs
    if not image1_url.startswith("http"):
        raise HTTPException(status_code=400, detail="Image 1 must be a valid URL.")
    if not image2_url.startswith("http"):
        raise HTTPException(status_code=400, detail="Image 2 must be a valid URL.")

    image1 = load_image_from_url(images[0].image)
    image2 = load_image_from_url(images[1].image)


    image1 = preprocess_image(image1)
    image2 = preprocess_image(image2)

    # Add batch dimension model expects input images in the shape (batch_size, height, width, channels)
    image1 = np.expand_dims(image1, axis=0)  
    image2 = np.expand_dims(image2, axis=0)

    score = predict(image1, image2)
    
    match = score > 0.5
    
    return {"score": round(score,2), "match": match}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
