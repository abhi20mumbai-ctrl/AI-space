
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Load original image
img = Image.open("test_data/nin.jpg").convert("L")

# Convert to NumPy array
img_array = np.array(img)

print("Image shape:", img_array.shape)
print("Minimum pixel:", img_array.min())
print("Maximum pixel:", img_array.max())
print("Mean pixel:", img_array.mean())

# Inspect pixel intensity distribution
plt.figure(figsize=(8, 4))
plt.hist(img_array.ravel(), bins=256)
plt.title("Pixel Intensity Distribution")
plt.xlabel("Pixel intensity (0-255)")
plt.ylabel("Frequency")
plt.show()

# Try multiple thresholds
thresholds = [100, 130, 150, 170, 190, 210]

fig, axes = plt.subplots(2, 3, figsize=(12, 7))

for ax, threshold in zip(axes.flat, thresholds):

    binary = (img_array < threshold).astype(np.uint8) * 255

    ax.imshow(binary, cmap="gray")
    ax.set_title(f"Threshold: {threshold}")
    ax.axis("off")

plt.tight_layout()
plt.show()
