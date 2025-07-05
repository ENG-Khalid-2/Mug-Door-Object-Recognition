# Mug & Door Object Recognition Project

This project uses a **Keras model** to recognize objects (Mug or Door) in images.

---

## 📌 Collected Samples

I collected sample images of mug and door for training:

- **Mug Sample**
  <img src="Mug_img.png" width="300">

- **Door Sample**
  <img src="door_img.png" width="300">

---

## 📌 Uploaded Samples to Teachable Machine

I uploaded the collected images to the training website:

<img src="1.png" width="500">

---

## 📌 Generated Code and Tested

I used the generated code from the Teachable Machine, and tested it on Google Colab:

- First code test:
  <img src="code_pic 1.png" width="500">

- Final code test:
  <img src="code_pic 2.png" width="500">
---

## 📌 Final Code Used

```python
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("/content/keras_model.h5", compile=False)

# Load the labels
class_names = open("/content/labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Load your image
image = Image.open("/content/mug.jpg").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print(" Confidence Score:", confidence_score)
