from tempfile import NamedTemporaryFile
import warnings
import torch
from mtcnn import MTCNN
import json
from fastapi import FastAPI, Response, File, Form, UploadFile, HTTPException
import librosa
import cv2
from PIL.Image import Image
from PIL import Image
import numpy as np
from moviepy.editor import VideoClip, VideoFileClip, AudioClip, AudioFileClip
warnings.filterwarnings("ignore")



AUDIO_EXTENSIONS = ["mp3", "flac", "wav"]
VIDEO_EXTENSIONS = ["mp4", "mov", "mkv"]
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]
AUDIO_DIM = (175, 175)
SILENCY_LAYER = 'model.block12.rep.4.conv1'


# interpretability import section
from torchray.attribution.grad_cam import grad_cam
from fastapi.middleware.cors import CORSMiddleware
# Model loading
model_path = r"Models\celebdf_final_model.pth"
model_alter = r"Models\final_model_epoch2.pth"
model_path_audio = r"Models\audio_model_epoch5.pth"
model_inter_path = r"Models\xception.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_audio = torch.load(model_path_audio, map_location=device)
model = torch.load(model_path, map_location=device)
model_inter = torch.load(model_inter_path, map_location = device)
model.eval()
model_audio.eval()
#model_inter.eval()
detector = MTCNN()

async def makeTempFile(file_content, suffixg):
     
     with NamedTemporaryFile(suffix=suffixg, delete=False) as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name
     return temp_file_path

async def extract_audio(file_content, required: dict):
     # this checks whether the uploaded audio file is uploaded by the user or is extracted from the uploaded video file
     if required["mint"]:
          aud_file = await makeTempFile(file_content=file_content, suffixg=required["filename"])
          # to check the volume, if it is zero, then we return an error message, and if it is above a certain
          # threshold, we proceed to further analysis
          aud_file_chc = AudioFileClip(aud_file)
          if aud_file_chc.max_volume() < 0.4:
               raise HTTPException(status_code=422, detail="Volume is too low for detection")
          else:
            y, sr = librosa.load(aud_file)
     else:
          
          # this is for processing audio extracted from video, which is a tab complex, because we are not saving it anywhere
          pass
     S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=175)
     S_dB = librosa.power_to_db(S)
     resized = cv2.resize(S_dB, AUDIO_DIM)
     normalized_image = (resized - np.min(resized)) * (255.0 / (np.max(resized) - np.min(resized)))
     normalized_image = normalized_image.astype('uint8')
     image = Image.fromarray(normalized_image)
     rgb_image = image.convert('RGB')
     return rgb_image




class VideoFrame:
    def __init__(self, frame, frame_id, time, size):
        self.frame = frame
        self.frame_id = frame_id
        self.time = time
        self.size = size

    def assign_frame_id(self, frame_id):
        self.frame_id = frame_id



def video_processing(file_content, data):
    clip: VideoFileClip
    # return the temp file using the extension
    cli = makeTempFile(file_content=file_content, suffixg=data["filename"])
    # loading the video in the memory in RAM without using harddrive, saving in the harddrive will come later
    
    clip = VideoFileClip(cli)
    pass



# Form(...)
app = FastAPI()
@app.post("/upload/")
async def main(file: UploadFile = File(...), json_data:str = {"nm": 23}):
     
     file_content = await file.read()
     data = json.loads(json_data)
     filename = file.filename
     extension = "." in filename and filename.rsplit(".", 1)[1].lower()
     # This part is for Video Processing
     if extension in VIDEO_EXTENSIONS:
          pass
     # This part is for Image Processing
     elif extension in IMAGE_EXTENSIONS:
          pass
     # This part is for Audio Processing
     elif extension in AUDIO_EXTENSIONS:
          
          pass
     else:
          raise HTTPException(status_code=422, detail="Unsupported File has been uploaded")
     return {"Working": 1}
     
