import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image


async def preprocessing(file_content):
    image_bytes = BytesIO(file_content)
    input_image = Image.open(image_bytes).convert("RGB")
    pixels = np.array(input_image)
    pixels = pixels.astype("uint8")

    return pixels
