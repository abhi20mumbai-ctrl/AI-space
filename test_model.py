
import tensorflow as tf
import numpy as np

# 1. Load the exported model
model = tf.keras.models.load_model(
    "models/digit_classifier.keras"
)

print("Model loaded successfully!")

# 2. Display model architecture
model.summary()

# 3. Create a dummy image
# Shape: (1, 28, 28)
dummy_image = np.random.random((1, 28, 28)).astype(np.float32)

# 4. Run inference
predictions = model.predict(dummy_image)

# 5. Get predicted digit
predicted_digit = np.argmax(predictions[0])

# 6. Get confidence
confidence = np.max(predictions[0]) * 100

print("\nPrediction:", predicted_digit)
print("Confidence:", confidence, "%")
print("Output shape:", predictions.shape)
