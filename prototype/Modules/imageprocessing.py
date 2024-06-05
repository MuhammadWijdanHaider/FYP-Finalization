import numpy as np
import cv2
import base64
from io import BytesIO
from PIL import Image
from Modules import contemp

# for test purposes, will remove if errors occured and will go back to the normal flow of prcessing
from fastapi import HTTPException


async def face_detection_and_cropping(file_content):
    image_bytes = BytesIO(file_content)
    input_image = Image.open(image_bytes).convert("RGB")
    # this conversion is for MTCNN
    pixels = np.array(input_image)
    pixels = pixels.astype("uint8")
    faces = await contemp.detect_faces(pixels)
    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No faces detected in the image.")
        # return -1
    else:
        faces = await contemp.cropping(faces, input_image)
    return faces


async def complete_preprocessing(file_content):
    image = face_detection_and_cropping(file_content)
    if type(image) == dict and image == -1:
        return -1
    else:
        if not len(image) == 0:

            pass
        else:
            raise HTTPException(
                status_code=400, detail="No faces detected in the image."
            )
        return image

