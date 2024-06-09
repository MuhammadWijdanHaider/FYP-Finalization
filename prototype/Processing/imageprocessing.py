import numpy as np
from io import BytesIO
from PIL import Image

from . import contemp

# for test purposes, will remove if errors occured and will go back to the normal flow of prcessing
from fastapi import HTTPException


async def face_detection_and_cropping(file_content) -> list:
    image_bytes = BytesIO(file_content)
    input_image = Image.open(image_bytes).convert("RGB")
    # this conversion is for MTCNN
    pixels = np.array(input_image)
    pixels = pixels.astype("uint8")
    faces = await contemp.detect_faces(pixels)
    if len(faces) == 0:
        raise HTTPException(status_code=400, detail="No faces detected in the image.")
    else:
        faces: list = await contemp.cropping(faces, input_image)
    return faces


async def complete_preprocessing(file_content):
    image: list = await face_detection_and_cropping(file_content)
    # check if the length is zero, and if it is, then raise error that nothing was detected
    if not (len(image) == 0):
        final_payload_return: dict = {"final_tensors": [],
                                      "cropped_images": []}
        for i in image:
            tensor = await contemp.preprocessing_f(i)
            final_payload_return["final_tensors"].append(tensor)
            final_payload_return["cropped_images"].append(i)
        return final_payload_return
    else:
        raise HTTPException(status_code=400, detail="No faces detected in the image.")
