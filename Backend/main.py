import PIL
import PIL.Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import JSONResponse
from mtcnn import MTCNN

from tempfile import NamedTemporaryFile
from moviepy.editor import VideoFileClip

import asyncio

# test section
from PIL import Image
import torchvision.transforms as transforms
from torchvision import transforms
import torch
import librosa


# Model loading bay, it is very resource heavy
#model_path = r"Models\\celebdf_final_model.pth"
#model_path_audio = r"Models\\audio_model_epoch5.pth"
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#model_audio = torch.load(model_path_audio, map_location=device)
#model = torch.load(model_path, map_location=device)
#model.eval()
#model_audio.eval()
detector = MTCNN()

ALLOWED_EXTENSIONS = {"mp4", "avi"}

def allowed(filename:str):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def audio_gen():
     dim = (224, 224)
     x, sr = librosa.load
     pass


def extract_faces(frames: list):
    
    pass

async def videoProcessing(file_content: bytes, information: dict):
     '''ths first step is to find whether the video's length is appropriate or not, but before that'''
     with NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name
     clip:VideoFileClip = VideoFileClip(temp_file_path)
     dur = clip.duration
     new: VideoFileClip
     if dur <= 10:
        providedTime = information["END"] - information["START"]
        # trimming parameters
        if providedTime > 0 and providedTime <= 10:
            new = clip.subclip(information["START"], information["END"])
        else:
            raise HTTPException(status_code=400, detail="The provided video and timestamps exceed the limit, which is 10 seconds. Please set the time stamps accordingly")
     else:
        providedTime = information["END"] - information["START"]
        # trimming parameters
        if providedTime > 0 and providedTime <= 10:
            pass
            new = clip.subclip(information["START"], information["END"])
        else:
            raise HTTPException(status_code=400, detail="The provided video and timestamps exceed the limit, which is 10 seconds. Please set the time stamps accordingly")
     
     # Frame extraction starts here, while we also start the extraction of the audio {later}
     print(new.duration)
     frames = []
     m_face_d = {"face_data": []}
     for frame in new.iter_frames(fps=1):
        print(type(frame))
        face = detector.detect_faces(frame)

        if len(face) > len(m_face_d["face_data"]):
                m_face_d["face_data"] = face
                m_face_d["frame"] = frame
     new.close()
     # very dependent on the FrontEnd architecture
     new_image: PIL.Image.Image
     if len(m_face_d["face_data"]) > 0:
         # the feature will be implemented when the frontend is finished because
         # now it is not viable
         pass
     elif len(m_face_d["face_data"]) == 0:
         raise HTTPException(status_code=422, detail="No face detected in the provided media.")

     else:
         # do for the single picture
         face_data = m_face_d["face_data"]
         if not(face_data['confidence'] < 0.9):
              x, y, w, h = face_data[0]['box']
              x, y, w, h = max(x, 0), max(y, 0), max(w, 0), max(h, 0)
              cropped_face = m_face_d["frame"][y:y+h, x:x+w]
              cropped_face_shape = cropped_face.shape
              cropped_face_size = (cropped_face_shape[1], cropped_face_shape[0])
              new_image = Image.new("RGB", cropped_face_size, (0, 0, 0, 0))
              new_image.paste(Image.fromarray(cropped_face), (0, 0))
         else:
              raise HTTPException(status_code=422, detail="No face detected in the provided media.")

     return new_image
     
     
     
     pass

async def extraction_bay(file_content: bytes, required):
     '''Function to extract various things from the Video Files'''
     with NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name
     clip = VideoFileClip(temp_file_path)
     pass

async def extract_frames(file_content: bytes):
    '''Function to extract frames and detection of faces and selection'''
    with NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name

    
    clip = VideoFileClip(temp_file_path)
    frames = []
    m_face_d = {"face_data": []}
    for frame in clip.iter_frames(fps=1):
        face = detector.detect_faces(frame)
        if len(face) > len(m_face_d["face_data"]):
                m_face_d["face_data"] = face
                m_face_d["frame"] = frame
    clip.close()
    # for the cropping
    new_image: PIL.Image.Image
    if len(m_face_d["face_data"]) > 0:
         # the feature will be implemented when the frontend is finished because
         # now it is not viable
         pass
    elif len(m_face_d["face_data"]) == 0:
         # return error handle
         pass
    else:
         # do for the single picture
         face_data = m_face_d["face_data"]
         if not(face_data['confidence'] < 0.9):
              x, y, w, h = face_data[0]['box']
              x, y, w, h = max(x, 0), max(y, 0), max(w, 0), max(h, 0)
              cropped_face = m_face_d["frame"][y:y+h, x:x+w]
              cropped_face_shape = cropped_face.shape
              cropped_face_size = (cropped_face_shape[1], cropped_face_shape[0])
              new_image = Image.new("RGB", cropped_face_size, (0, 0, 0, 0))
              new_image.paste(Image.fromarray(cropped_face), (0, 0))

         else:
              raise ValueError("Face confidence is below 0.9")
         pass
         
    
    return new_image

async def audio_spectogram_generation():
     
     pass

async def prediction(image: PIL.Image.Image):
      
     transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True)
        ])
     img = transform(image)
     img = img.unsqueeze(0)
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
    extracted_frames = await videoProcessing(file_content, information={"END": 30, "START": 17})
    # Do something with the file and JSON data
    # For demonstration, let's just print them
    print("Uploaded file:", file.filename)
    # print("JSON data:", json_data)
    
    # Here, you can manipulate the file_content or json_data as needed
    
    # You can also return a response or perform any other actions
    
    return {"file_uploaded": extracted_frames}