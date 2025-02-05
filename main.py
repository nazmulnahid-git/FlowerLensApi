from fastapi import FastAPI, HTTPException
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import tensorflow as tf  # For TFLite

# Initialize FastAPI app
app = FastAPI()

# Load the TFLite model
MODEL_PATH = "flowerLens.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess the image
def preprocess_image(image):
    image = image.resize((150, 150))
    image_array = np.array(image, dtype=np.float32) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

# Function to predict the flower
def predict_flower(image):
    # Preprocess the image
    processed_image = preprocess_image(image)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], processed_image)

    # Run inference
    interpreter.invoke()

    # Get the output tensor
    output_data = interpreter.get_tensor(output_details[0]['index'])
    flower_class = np.argmax(output_data, axis=1)[0]  # Assuming output is a probability distribution

    return int(flower_class)

# Define the API endpoint
@app.get("/predict/")
async def predict(image_url: str):
    try:
        # Download the image from the URL
        response = requests.get(image_url)
        response.raise_for_status()

        # Open the image using PIL
        image = Image.open(BytesIO(response.content))

        # Predict the flower
        flower_class = predict_flower(image)

        # Return the result
        # We will return the flower name directly later.
        return {"flower_class": flower_class}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "Welcome to the flowerLens API!"}
# Run the app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)