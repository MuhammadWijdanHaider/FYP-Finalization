import torch
from mtcnn import MTCNN
import json
from fastapi import FastAPI, Response, File, Form, UploadFile, HTTPException



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


app = FastAPI()
@app.post("/upload/")
async def main(file: UploadFile = File(...), json_data:str = Form(...)):
     
     file_content = await file.read()
     data = json.loads(json_data)
     filename = file_content.filename
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
     
