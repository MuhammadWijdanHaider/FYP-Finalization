from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from io import BytesIO
import os
import json
import ffmpeg

import asyncio

# test section

from asynccpu import ProcessTaskPoolExecutor
from asyncffmpeg import FFmpegCoroutineFactory, StreamSpec

ALLOWED_EXTENSIONS = {"mp4", "avi"}

def allowed(filename:str):
    return "." in filename and filename.rsplit(".", 1)[1].lower()

def extract_frames(file):
    
    pass

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
    
    return {"file_uploaded": file.filename, "Type": type(extract_frames)}