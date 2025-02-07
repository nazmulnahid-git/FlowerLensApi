from fastapi import FastAPI, HTTPException
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import tensorflow as tf  # For TFLite
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the TFLite model
MODEL_PATH = "flowerLens.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Ensure model input shape is correct
expected_shape = input_details[0]['shape']
IMG_SIZE = expected_shape[1]  # Assuming shape is (1, 150, 150, 3)

# Flower labels
flower_labels = {0: "Daisy", 1: "Dandelion", 2: "Rose", 3: "Sunflower", 4: "Tulip"}

# Function to preprocess the image
def preprocess_image(image):
    image = image.convert("RGB")  # Ensure 3 channels
    image = image.resize((IMG_SIZE, IMG_SIZE))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to predict the flower
def predict_flower(image):
    processed_image = preprocess_image(image)
    interpreter.set_tensor(input_details[0]['index'], processed_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    probabilities = output_data[0]  # Get probabilities for all classes
    flower_class = np.argmax(probabilities)  # Get predicted class
    probability = float(probabilities[flower_class])  # Get probability of predicted class
    
    return int(flower_class), probability

# Define the API endpoint
@app.get("/predict/")
async def predict(image_url: str):
    try:
        # Download the image
        response = requests.get(image_url)
        response.raise_for_status()
        
        # Open and process the image
        try:
            image = Image.open(BytesIO(response.content))
            image = image.convert("RGB")  # Ensure correct format
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image format")

        # Predict the flower
        flower_class, probability = predict_flower(image)
        flower_name = flower_labels.get(flower_class, "Unknown")

        return {
            "flower_class": flower_class,
            "flower_name": flower_name,
            "probability": round(probability, 4) * 100  # Rounded for better readability
        }
    
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch image: {str(e)}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Welcome to the flowerLens API!"}
