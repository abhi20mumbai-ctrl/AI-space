
import io
import numpy as np
import tensorflow as tf

from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException


# ----------------------------------
# 1. Initialize FastAPI
# ----------------------------------

app = FastAPI(
    title="Digit Classification API",
    description="Handwritten digit recognition using TensorFlow ANN",
    version="1.0.0"
)


# ----------------------------------
# 2. Load trained model
# ----------------------------------

model = tf.keras.models.load_model(
    "models/digit_classifier.keras"
)

print("Digit classification model loaded successfully!")


# ----------------------------------
# 3. Health check API
# ----------------------------------

@app.get("/")
def home():
    return {
        "message": "Digit Classification API is running"
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# ----------------------------------
# 4. Prediction API
# ----------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # Read uploaded image
        image_bytes = await file.read()

        # Convert image to grayscale
        img = Image.open(
            io.BytesIO(image_bytes)
        ).convert("L")

        # Resize image to 28x28
        img = img.resize(
            (28, 28),
            Image.Resampling.LANCZOS
        )

        # Convert image to NumPy array
        img_array = np.array(
            img,
            dtype=np.float32
        )

        # Normalize pixel values (0-255) to (0-1)
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # Model inference
        predictions = model.predict(
            img_array,
            verbose=0
        )

        # Get predicted digit
        predicted_digit = int(
            np.argmax(predictions[0])
        )

        # Get confidence
        confidence = float(
            np.max(predictions[0])
        )

        return {
            "success": True,
            "predicted_digit": predicted_digit,
            "confidence": round(confidence * 100, 2)
        }

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
