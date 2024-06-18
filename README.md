# Face Verification System

This project implements a face verification system using Siamese Networks with TensorFlow/Keras. It allows verifying whether two given images belong to the same person or not.

## Architecture
The architecture of the system involves:
- Siamese Neural Network with VGG16 base model for feature extraction.
- Conversion of the trained model to ONNX format.
- Exposing an API for inference using FastAPI.


## Dataset
The LFW (Labeled Faces in the Wild) dataset was used for training and testing the model. It contains face images of various individuals collected from the internet. You can find the dataset [here](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset/data).

## Clone the repository
```bash
git clone https://github.com/NourhanNabil/face-verification.git
cd face-verification
```

## Docker
1. Build the Docker image:
```bash
docker build -t face-verification .
```

2. Run the Docker container:
``` bash 
docker run -d -p 8000:8000 face-verification 
```

### Face Verification API
- **Endpoint**: `/predict`
- **Method**: POST
- **Request Body**: JSON containing paths to two images to be verified, where each image should be URL.
- **Paylod Example**: 
``` bash
   [
     { "image": "url_to_image1"},
     { "image": "url_to_image2"}
    ]
```
- **Response**: JSON with the verification result (similarity score and matched or not matched).

### Swagger UI
- Access the API documentation and test the endpoints using the Swagger UI at `http://localhost:8000/docs`. 








