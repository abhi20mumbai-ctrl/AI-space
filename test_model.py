
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from pathlib import Path
from scipy import ndimage


# ----------------------------------
# 1. Load trained model
# ----------------------------------

model = tf.keras.models.load_model(
    "models/digit_classifier.keras"
)

print("Model loaded successfully!")

model.summary()


# ----------------------------------
# 2. Locate test images
# ----------------------------------

test_dir = Path("test_data")

image_paths = (
    list(test_dir.glob("*.jpg")) +
    list(test_dir.glob("*.png")) +
    list(test_dir.glob("*.jpeg"))
)

print(f"Found {len(image_paths)} images")


# ----------------------------------
# 3. Image preprocessing
# ----------------------------------


def preprocess_image(image_path):

    # STEP 1: Load original RGB image
    img = Image.open(image_path).convert("RGB")

    # STEP 2: Convert RGB to HSV
    hsv = np.array(img.convert("HSV"))

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]

    # STEP 3: Detect blue ink
    # PIL Hue range: 0-255
    # Blue generally falls approximately between 130-190

    blue_mask = (
        (hue >= 130) &
        (hue <= 190) &
        (saturation > 40)
    )

    # STUDY: Boolean masking and HSV colour space

    if not np.any(blue_mask):
        raise ValueError("No blue ink detected")

    # STEP 4: Connected component analysis
    labels, number_of_components = ndimage.label(blue_mask)

    component_sizes = np.bincount(labels.ravel())
    component_sizes[0] = 0

    largest_component = np.argmax(component_sizes)

    digit_mask = labels == largest_component

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

    # STEP 8: Create 28x28 black canvas
    canvas = Image.new("L", (28, 28), 0)

    # STEP 9: Center digit
    x = (28 - cropped.width) // 2
    y = (28 - cropped.height) // 2

    canvas.paste(cropped, (x, y))

    # STEP 10: Normalize
    img_array = np.array(
        canvas,
        dtype=np.float32
    ) / 255.0

    # STEP 11: Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    return img_array, canvas


# ----------------------------------
# 4. Run predictions
# ----------------------------------

for image_path in image_paths:

    print("\nProcessing:", image_path.name)

    img_array, processed_image = preprocess_image(image_path)

    # Model inference
    predictions = model.predict(img_array, verbose=0)

    # Predicted digit
    predicted_digit = np.argmax(predictions[0])

    # Confidence
    confidence = np.max(predictions[0]) * 100

    print("Predicted digit:", predicted_digit)
    print("Confidence:", round(confidence, 2), "%")

    
# Display original and processed images

fig, axes = plt.subplots(1, 2, figsize=(8, 4))

# Original photograph
original = Image.open(image_path)

axes[0].imshow(original)
axes[0].set_title("Original Image")
axes[0].axis("off")

# Processed 28x28 image
axes[1].imshow(processed_image, cmap="gray", vmin=0, vmax=255)
axes[1].set_title(
    f"Processed 28x28 | Predicted: {predicted_digit}"
)
axes[1].axis("off")

plt.tight_layout()
plt.show()
