from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from mtcnn import MTCNN

from tempfile import NamedTemporaryFile

import asyncio

# test section

from moviepy.editor import VideoFileClip

ALLOWED_EXTENSIONS = {"mp4", "avi"}

def allowed(filename:str):
    return "." in filename and filename.rsplit(".", 1)[1].lower()

async def extract_frames(file_content: bytes):
    
    with NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name


    clip = VideoFileClip(temp_file_path)
    frames = []
    for frame in clip.iter_frames(fps=1):
        frames.append(frame)
    clip.close()
    detector = MTCNN()
    detector.detect_faces()
    return 1



# , info: str = Form(...)
app = FastAPI()
@app.post("/upload/")
async def upload_file_and_json(file: UploadFile = File(...)):
    if not allowed(file.filename):
        raise HTTPException(status_code=400, detail="File extension not allowed")
    # Read the JSON data


    # json_data = json.loads(info)
    
    # Process the uploaded file
    file_content = await file.read()
    extracted_frames = await extract_frames(file_content)
    # Do something with the file and JSON data
    # For demonstration, let's just print them
    print("Uploaded file:", file.filename)
    # print("JSON data:", json_data)
    
    # Here, you can manipulate the file_content or json_data as needed
    
    # You can also return a response or perform any other actions
    
    return {"file_uploaded": file.filename}