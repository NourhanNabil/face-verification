# Face Verification System

This project implements a face verification system using Siamese Networks with TensorFlow/Keras. It allows verifying whether two given images belong to the same person or not.

## Architecture
The architecture of the system involves:
- Siamese Neural Network with VGG16 base model for feature extraction.
- Conversion of the trained model to ONNX format.
- Optimization and deployment using TensorRT (TRT).
- Exposing an API for inference using FastAPI.
- Documentation with Swagger UI.

## Dataset
The LFW (Labeled Faces in the Wild) dataset was used for training and testing the model. It contains face images of various individuals collected from the internet. You can find the dataset [here](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset/data).

## Clone the repository
```bash
git clone https://github.com/NourhanNabil/face-verification.git
cd face-verification
```

## Model Architecture and Evaluation Metric
- Base Model: VGG16
- Evaluation Metric: Binary Crossentropy
- Accuracy: 59.30%

### Face Verification API
- **Endpoint**: `/predict`
- **Method**: POST
- **Request Body**: JSON containing paths to two images to be verified, where each image can be either a URL or a base64 encoded image.
- **Example paylod**: 
``` bash[
    {
        "image": "https://m.media-amazon.com/images/M/MV5BMTQzMjkwNTQ2OF5BMl5BanBnXkFtZTgwNTQ4MTQ4MTE@._V1_.jpg"
    },
    {
        "image": "https://media-cldnry.s-nbcnews.com/image/upload/t_fit-1500w,f_auto,q_auto:best/streams/2013/January/130115/1B5537501-120215-ryan-gosling.jpg"
    }
]
```
- **Response**: JSON with the verification result (matched or not matched).

### Swagger UI
- Access the API documentation and test the endpoints using the Swagger UI at `/docs`.

## ONNX and TRT
The trained model is converted to ONNX format for interoperability and then optimized and deployed using TensorRT (TRT) for faster inference.

## Docker
1. Build the Docker image:
```
bash docker build -t face-verification  
```

2. Run the Docker container:
``` bash 
docker run -d -p 8000:8000 face-verification 
```

3. Access the API documentation at `http://localhost:8000/docs` in your web browser.

## Requirements
- TensorFlow
- Keras
- FastAPI
- ONNX Runtime
- TensorRT
- Docker





