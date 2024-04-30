from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from mtcnn import MTCNN

from tempfile import NamedTemporaryFile
from moviepy.editor import VideoFileClip

import asyncio

# test section



ALLOWED_EXTENSIONS = {"mp4", "avi"}

def allowed(filename:str):
    return "." in filename and filename.rsplit(".", 1)[1].lower()

async def extract_frames(file_content: bytes):
    '''Function to extract frames and detection of faces and selection'''
    with NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name

    detector = MTCNN()
    clip = VideoFileClip(temp_file_path)
    frames = []
    m_face_d = {"face_data": []}
    for frame in clip.iter_frames(fps=1):
        frames.append(frame) # this is depricated, and can be removed
        face = detector.detect_faces(frame)
        if len(face) > len(m_face_d["face_data"]):
                m_face_d["face_data"] = face
                m_face_d["frame"] = frame
    clip.close()
    # for the cropping
    if len(m_face_d["face_data"]) > 0:
         pass
    elif len(m_face_d["face_data"]) == 0:
         # return error handle
         pass
    else:
         # do for the single picture
         face_data = m_face_d["face_data"]
         if not(face_data['confidence'] < 0.9):
              x, y, w, h = face_data['box']
              x, y, w, h = max(x, 0), max(y, 0), max(w, 0), max(h, 0)
              cropped_face = m_face_d["frame"][y:y+h, x:x+w]
              
              pass
         else:
              raise ValueError("Face confidence is below 0.9")
         pass
         
    
    return 1

# first compare the len of all the frames, and the one with the greates number
# of faces will be returned for further selection, and if there is only one
# then proceed



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