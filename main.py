
import io
import numpy as np
import tensorflow as tf

from PIL import Image, UnidentifiedImageError
from scipy import ndimage

from fastapi import FastAPI, UploadFile, File, HTTPException


# ----------------------------------
# 1. Initialize FastAPI
# ----------------------------------

app = FastAPI(
    title="Digit Classification API",
    description="Handwritten digit recognition using TensorFlow ANN",
    version="2.0.0"
)


# ----------------------------------
# 2. Load trained model
# ----------------------------------

model = tf.keras.models.load_model(
    "models/digit_classifier.keras"
)

print("Digit classification model loaded successfully!")


# ----------------------------------
# 3. Image preprocessing
# ----------------------------------

def preprocess_image(image: Image.Image):

    # STEP 1: Convert image to RGB
    img = image.convert("RGB")

    # STEP 2: Convert RGB to HSV
    hsv = np.array(img.convert("HSV"))

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # STEP 3: Detect blue ink
    # PIL HSV Hue range: 0-255

    blue_mask = (
        (hue >= 130) &
        (hue <= 190) &
        (saturation > 40)
    )

    # STUDY: RGB to HSV conversion
    # STUDY: Boolean masking and image segmentation

    if not np.any(blue_mask):
        raise ValueError("No blue ink detected in the image")

    # STEP 4: Connected Component Analysis

    labels, number_of_components = ndimage.label(blue_mask)

    component_sizes = np.bincount(labels.ravel())

    # Ignore background
    component_sizes[0] = 0

    if component_sizes.max() == 0:
        raise ValueError("No connected digit component detected")

    # Find largest connected component
    largest_component = np.argmax(component_sizes)

    digit_mask = labels == largest_component

    # STUDY: Connected Component Analysis

    # STEP 5: Find digit boundaries

    rows, cols = np.where(digit_mask)

    top, bottom = rows.min(), rows.max() + 1
    left, right = cols.min(), cols.max() + 1

    # STEP 6: Create white digit on black background

    clean_binary = (digit_mask * 255).astype(np.uint8)

    cropped = Image.fromarray(clean_binary).crop(
        (left, top, right, bottom)
    )

    # STEP 7: Resize while maintaining aspect ratio

    cropped.thumbnail(
        (20, 20),
        Image.Resampling.LANCZOS
    )

    # STEP 8: Create black 28x28 canvas

    canvas = Image.new("L", (28, 28), 0)

    # STEP 9: Center digit

    x = (28 - cropped.width) // 2
    y = (28 - cropped.height) // 2

    canvas.paste(cropped, (x, y))

    # STEP 10: Normalize pixel values (0-1)

    img_array = np.array(
        canvas,
        dtype=np.float32
    ) / 255.0

    # STEP 11: Add batch dimension
    # Shape: (28, 28) -> (1, 28, 28)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    return img_array


# ----------------------------------
# 4. Health check API
# ----------------------------------

@app.get("/")
def home():

    return {
        "message": "Digit Classification API is running",
        "version": "2.0.0"
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": model is not None
    }


# ----------------------------------
# 5. Prediction API
# ----------------------------------

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:

        # STEP 1: Read uploaded image
        image_bytes = await file.read()

        if not image_bytes:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty"
            )

        # STEP 2: Open uploaded image
        try:
            img = Image.open(
                io.BytesIO(image_bytes)
            )

        except UnidentifiedImageError:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file"
            )

        # STEP 3: Preprocess image
        img_array = preprocess_image(img)

        # STEP 4: Model inference
        predictions = model.predict(
            img_array,
            verbose=0
        )

        # STEP 5: Get predicted digit
        predicted_digit = int(
            np.argmax(predictions[0])
        )

        # STEP 6: Get confidence
        confidence = float(
            np.max(predictions[0])
        )

        # STEP 7: Get probabilities for all digits
        probabilities = {
            str(digit): round(float(probability) * 100, 2)
            for digit, probability in enumerate(predictions[0])
        }

        return {
            "success": True,
            "predicted_digit": predicted_digit,
            "confidence": round(confidence * 100, 2),
            "probabilities": probabilities
        }

    except HTTPException:
        raise

    except ValueError as e:

        raise HTTPException(
            status_code=422,
            detail=str(e)
        )

    except Exception as e:

        print(f"Prediction error: {str(e)}")

        raise HTTPException(
            status_code=500,
            detail="Internal prediction error"
        )
