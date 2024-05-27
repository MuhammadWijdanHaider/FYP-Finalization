from io import BytesIO
import PIL
import PIL.Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from mtcnn import MTCNN
from tempfile import NamedTemporaryFile
from moviepy.editor import VideoFileClip
import asyncio
import PIL
import PIL.Image
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np
import librosa
import cv2
import soundfile as sf

# Model loading
model_path = r"Models\celebdf_final_model.pth"
model_path_audio = r"Models\audio_model_epoch5.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_audio = torch.load(model_path_audio, map_location=device)
model = torch.load(model_path, map_location=device)
model.eval()
model_audio.eval()
detector = MTCNN()

ALLOWED_EXTENSIONS = ["mp3", "flac", "wav", "mp4", "mov", "mkv", "jpg", "jpeg", "png"]
AUDIO_EXTENSIONS = ["mp3", "flac", "wav"]
VIDEO_EXTENSIONS = ["mp4", "mov", "mkv"]
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]
AUDIO_DIM = (224,224)

def allowed(filename: str):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

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

async def preprocessing(file_content):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    img = transform(file_content)
    img = img.unsqueeze(0)        
    return img


async def audio_processing(file_content: bytes, required):
    aud_file = file_content
    if required["mint"] == True:
        aud_file = await retTempFile(file_content=file_content, suffixg=required["filename"])
        y, sr = librosa.load(aud_file, sr=None)
    else:
        audio_buffer = BytesIO()
        file_content.write_audiofile(audio_buffer, codec='pcm_s16le')
        audio_buffer.seek(0)
        y, sr = sf.read(audio_buffer)
    
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=175)
    S_dB = librosa.power_to_db(S)
    resized = cv2.resize(S_dB, AUDIO_DIM)
    normalized_image = (resized - np.min(resized)) * (255.0 / (np.max(resized) - np.min(resized)))
    normalized_image = normalized_image.astype('uint8')
    image = Image.fromarray(normalized_image)
    return image

async def retTempFile(file_content, suffixg):
    with NamedTemporaryFile(suffix=suffixg, delete=False) as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name
    return temp_file_path

async def image_processing(file_content):
    image_bytes = BytesIO(file_content)
    input_image = Image.open(image_bytes).convert("RGB")
    pixels = np.array(input_image)
    pixels = pixels.astype('uint8')
    temp_img = None
    faces = detector.detect_faces(pixels)
    
    if len(faces) > 1:
        pass  # further implementation after the frontend is done
    elif len(faces) == 0:
        raise HTTPException(status_code=400, detail="No faces detected in the image.")
    else:
        face = faces[0]
        print(faces)
        print(faces[0])
        if not (face['confidence'] < 0.9):
            x, y, w, h = face['box']
            x, y, w, h = max(x, 0), max(y, 0), max(w, 0), max(h, 0)
            extracted_face = input_image.crop((x, y, x+w, y+h))
            new_image = Image.new("RGB", extracted_face.size, (0, 0, 0, 0))
            new_image.paste(extracted_face, (0, 0))
            temp_img = new_image
        else:
            raise HTTPException(status_code=400, detail="No faces detected in the image.")
    return temp_img


async def make_predictions(img, type):
    prediction: int
    if type == "audio":
        with torch.no_grad():
            output = model_audio(pixel_values=img)
        logits = output.logits
        prediction = torch.argmax(logits).item()
    elif type == "image":
        with torch.no_grad():
            output = model(pixel_values=img)
        logits = output.logits
        prediction = torch.argmax(logits).item()
    else:
        raise HTTPException(status_code=500, detail="Internal Error")
    return prediction

app = FastAPI()

@app.post("/upload/")
async def upload_file_and_json(file: UploadFile = File(...)):
    file_content = await file.read()
    filename = file.filename
    extension = "." in filename and filename.rsplit(".", 1)[1].lower()
    if not extension in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File extension not allowed")
    prediction:int
    p_type: str
    if extension in AUDIO_EXTENSIONS:
        mel_spectrograms = await audio_processing(file_content, required={"filename":extension, "mint": True})
        mel_spectrograms = mel_spectrograms.convert('RGB')
        img = await preprocessing(mel_spectrograms)       
        prediction = await make_predictions(img, "audio")
        p_type = "Audio"
    elif extension in IMAGE_EXTENSIONS:
        image = await image_processing(file_content)
        img = await preprocessing(image) 
        prediction = await make_predictions(img, "image")
        p_type = "Image"
    return {"file_uploaded": prediction, "Type": p_type}
