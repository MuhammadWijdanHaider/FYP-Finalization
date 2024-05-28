from io import BytesIO
from fastapi import FastAPI, File, Response, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse, StreamingResponse
from mtcnn import MTCNN
from PIL import Image
import numpy as np
import base64
detector = MTCNN()
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all origins (replace with specific origins if needed)
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Allow GET and POST requests (add other methods as needed)
    allow_headers=["*"],  # Allow all headers (you can customize this based on your requirements)
)
def generate_images(images):
    l = []
    for image in images:
        img_byte_array = BytesIO()
        image.save(img_byte_array, format="jpeg")
        img_byte_array.seek(0)
        encoded_image = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')
        l.append(encoded_image)
    return l 

@app.post("/images", response_class=Response)
async def upload_file_and_json(file: UploadFile = File(...)):
    file_content = await file.read()
    image_bytes = BytesIO(file_content)
    input_image = Image.open(image_bytes).convert("RGB")
    
    pixels = np.array(input_image)
    pixels = pixels.astype('uint8')
    
    faces = detector.detect_faces(pixels)
    
    t_faces = []
    for face in faces:
        if face['confidence'] >= 0.9:
            x, y, w, h = face['box']
            x, y, w, h = max(x, 0), max(y, 0), max(w, 0), max(h, 0)
            extracted_face = input_image.crop((x, y, x+w, y+h))
            new_image = Image.new("RGB", extracted_face.size, (0, 0, 0, 0))
            new_image.paste(extracted_face, (0, 0))
            t_faces.append(new_image)
    return JSONResponse(content={"images": [generate_images(t_faces)]})
    #return StreamingResponse(
    #    generate_images(t_faces),
    #    status_code=status.HTTP_200_OK,
    #    media_type='image/jpeg'
    #)