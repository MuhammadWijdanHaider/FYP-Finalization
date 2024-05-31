from io import BytesIO
import PIL
import PIL.Image
from fastapi import Body, FastAPI, File, Response, UploadFile, HTTPException, Form
from pydantic import BaseModel
from typing import Dict, Any

from fastapi.responses import JSONResponse, StreamingResponse
from mtcnn import MTCNN
from tempfile import NamedTemporaryFile
from moviepy.editor import VideoFileClip, AudioFileClip
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
from scipy.io.wavfile import write
import base64
# interpretability import section
from torchray.attribution.grad_cam import grad_cam
from fastapi.middleware.cors import CORSMiddleware

class Payload(BaseModel):
    data: Dict[str, Any]




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

ALLOWED_EXTENSIONS = ["mp3", "flac", "wav", "mp4", "mov", "mkv", "jpg", "jpeg", "png"]
AUDIO_EXTENSIONS = ["mp3", "flac", "wav"]
VIDEO_EXTENSIONS = ["mp4", "mov", "mkv"]
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]
AUDIO_DIM = (175,175)
SILENCY_LAYER = 'model.block12.rep.4.conv1'


def ndarray_to_image(ndarray: np.ndarray) -> BytesIO:
    image = Image.fromarray(ndarray.astype('uint8'), 'RGB')
    img_io = BytesIO()
    image.save(img_io, format='jpeg')
    img_io.seek(0)
    encoded_image = base64.b64encode(img_io.getvalue()).decode('utf-8')
    return encoded_image
    pass

def allowed(filename: str):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


async def preprocessing(file_content):
    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    img = transform(file_content)
    img = img.unsqueeze(0)        
    print(type(img))
    return img


async def audio_processing(file_content, required):
    aud_file = file_content
    if required["mint"] == True:
        aud_file = await retTempFile(file_content=file_content, suffixg=required["filename"])
        aud_file_chc = AudioFileClip(aud_file)
        if aud_file_chc.max_volume() < 0.4:
               raise HTTPException(status_code=422, detail="Volume is too low for detection")
        else:
            y, sr = librosa.load(aud_file)

    else:
        try:
            audio_clip:AudioFileClip = file_content
            print(type(audio_clip))
            audio_fps = audio_clip.fps
            audio_chunks = []
            
            for chunk in audio_clip.iter_chunks(fps=audio_fps, chunksize=4096):
                audio_chunks.append(chunk)
            audio_data = np.vstack(audio_chunks)
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)
            buffer = BytesIO()
            write(buffer, audio_fps, (audio_data * 32767).astype(np.int16))
            buffer.seek(0)
            y, sr = librosa.load(buffer, sr=audio_fps)
        except Exception as e:
            print(e) 
    try:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=175)
        S_dB = librosa.power_to_db(S)
        resized = cv2.resize(S_dB, AUDIO_DIM)
        normalized_image = (resized - np.min(resized)) * (255.0 / (np.max(resized) - np.min(resized)))
        normalized_image = normalized_image.astype('uint8')
        image = Image.fromarray(normalized_image)
        return image
    # mel_spectrograms = mel_spectrograms.convert('RGB')
    except Exception as e:
        print(e)

async def retTempFile(file_content, suffixg):
    with NamedTemporaryFile(suffix=suffixg, delete=False) as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name
    return temp_file_path

async def image_processing(file_content):
    image_bytes = BytesIO(file_content)
    input_image = Image.open(image_bytes).convert("RGB")
    print(type(input_image))
    pixels = np.array(input_image)

    pixels = pixels.astype('uint8')
    temp_img = None
    faces = detector.detect_faces(pixels)
    
    if len(faces) > 1:
        t_faces = []
        for face in faces:
            if face['confidence'] >= 0.9:
                x, y, w, h = face['box']
                x, y, w, h = max(x, 0), max(y, 0), max(w, 0), max(h, 0)
                extracted_face = input_image.crop((x, y, x+w, y+h))
                new_image = Image.new("RGB", extracted_face.size, (0, 0, 0, 0))
                new_image.paste(extracted_face, (0, 0))
                t_faces.append(new_image)
        return t_faces
    elif len(faces) == 0:
        raise HTTPException(status_code=400, detail="No faces detected in the image.")
    else:
        face = faces[0]
        #print(faces)
        #print(faces[0])
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


async def make_predictions(img, types):
    prediction: int
    if types == "audio":
        with torch.no_grad():
            output = model_audio(pixel_values=img)
        logits = output.logits
        prediction = torch.argmax(logits).item()
    elif types == "image":
        with torch.no_grad():
            output = model(pixel_values=img)
        logits = output.logits
        prediction = torch.argmax(logits).item()
    else:
        raise HTTPException(status_code=500, detail="Internal Error")
    print(prediction)
    return prediction

async def video_processing(file_content, data):
    clip: VideoFileClip
    # tem = retTempFile(file_content, data["ext"])
    with NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name
    clip = VideoFileClip(temp_file_path)
    # print("TEST")
    dur = clip.duration
    new: VideoFileClip

    if dur <= 10:
        providedTime = data["END"] - data["START"]
        if providedTime > 0 and providedTime <= 10:
            new = clip.subclip(data["START"], data["END"])
        else:
            raise HTTPException(status_code=400, detail="The provided video and timestamps exceed the limit, which is 10 seconds. Please set the time stamps accordingly")
        pass
    else:
        providedTime = data["END"] - data["START"]
        if providedTime > 0 and providedTime <= 10:
            new = clip.subclip(data["START"], data["END"])
        else:
            raise HTTPException(status_code=400,
                                detail="The provided video and timestamps exceed the limit, which is 10 seconds. Please set the time stamps accordingly")
    frames = []
    m_face_d = {"face_data": []}
    for frame in new.iter_frames(fps=1):
        print(type(frame))
        face = detector.detect_faces(frame)

        if len(face) > len(m_face_d["face_data"]):
                m_face_d["face_data"] = face
                m_face_d["frame"] = frame
    audio = new.audio
    # free up memories
    # new.close()
    # clip.close()

    new_image: PIL.Image.Image
    if len(m_face_d["face_data"]) > 1:
         t_faces = []
         actual_face_data:list = m_face_d["face_data"]
         actual_frame = m_face_d["frame"]
         for face in actual_face_data:
             x, y, w, h = face['box']
             x, y, w, h = max(x, 0), max(y, 0), max(w, 0), max(h, 0)
             extracted_face = actual_frame.crop((x, y, x+w, y+h))
             new_image = Image.new("RGB", extracted_face.size, (0, 0, 0, 0))
             new_image.paste(extracted_face, (0, 0))
             t_faces.append(new_image)
         return {"total_faces": t_faces, "completed": True}
         

        #  for face in m_face_d:
        #      pass
         # the feature will be implemented when the frontend is finished because
         # now it is not viable
    elif len(m_face_d["face_data"]) == 0:
         raise HTTPException(status_code=422, detail="No face detected in the provided media.")
    else:
        face_data = m_face_d["face_data"]
        if not(face_data[0]['confidence'] < 0.9):
              x, y, w, h = face_data[0]['box']
              x, y, w, h = max(x, 0), max(y, 0), max(w, 0), max(h, 0)
              cropped_face = m_face_d["frame"][y:y+h, x:x+w]
              cropped_face_shape = cropped_face.shape
              cropped_face_size = (cropped_face_shape[1], cropped_face_shape[0])
              new_image = Image.new("RGB", cropped_face_size, (0, 0, 0, 0))
              new_image.paste(Image.fromarray(cropped_face), (0, 0))
        else:
              raise HTTPException(status_code=422, detail="No face detected in the provided media.")

    transformed = await preprocessing(new_image) # this is a tensor
    image_prediction = await make_predictions(transformed, types="image") # that is the prediction
    
    mel = await audio_processing(audio, required={"mint": False})
    mels = mel.convert('RGB')
    transformed_audio = await preprocessing(mels)
    audio_prediction = await make_predictions(transformed_audio, types="audio")
    return {"Image": image_prediction, "audio": audio_prediction, "PIL_IMAGE": new_image, "TENSORS": transformed, "completed": True}


async def interpretability(image, prediction, tensor):
    grad_cam_result = grad_cam(model_inter, tensor, saliency_layer='model.block12.rep.4.conv1', target=prediction)
    grad_cam_heatmap = grad_cam_result[0].detach().numpy().sum(axis=0)
    resized_heatmap = cv2.resize(grad_cam_heatmap, (224, 224))
    # change this if error occurs
    nd = np.array(image, dtype=np.uint8)
    input_image = cv2.cvtColor(nd, cv2.COLOR_RGBA2RGB)
    input_image = cv2.resize(input_image, (224,224))
    resized_heatmap = (resized_heatmap - resized_heatmap.min()) / (resized_heatmap.max() - resized_heatmap.min() + 1e-8)
    heatmap_rgb = cv2.applyColorMap(np.uint8(255 * resized_heatmap), cv2.COLORMAP_JET)
    overlayed_image = cv2.addWeighted(input_image, 0.5, heatmap_rgb, 0.5, 0)
    p = cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)
    print(type(p))
    return p
    

async def generate_images(images):
    l = []
    print("COME")
    for image in images:
        img_byte_array = BytesIO()
        image.save(img_byte_array, format="jpeg")
        img_byte_array.seek(0)
        encoded_image = base64.b64encode(img_byte_array.getvalue()).decode('utf-8')
        l.append(encoded_image)
    return l 



app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all origins (replace with specific origins if needed)
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Allow GET and POST requests (add other methods as needed)
    allow_headers=["*"],  # Allow all headers (you can customize this based on your requirements)
)
@app.post("/upload/")
async def upload_file_and_json(file: UploadFile = File(...)):
    file_content = await file.read()
    filename = file.filename
    extension = "." in filename and filename.rsplit(".", 1)[1].lower()
    if not extension in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="File extension not allowed")
    prediction:int
    p_type: str
    # audio
    if extension in AUDIO_EXTENSIONS:
        print(extension)
        mel_spectrograms = await audio_processing(file_content, required={"filename":extension, "mint": True})
        mel_spectrograms = mel_spectrograms.convert('RGB')
        img = await preprocessing(mel_spectrograms)       
        prediction = await make_predictions(img, "audio")
        p_type = "Audio"
        return {"prediction": prediction, "key": "audio"}
    # images
    elif extension in IMAGE_EXTENSIONS:
        image = await image_processing(file_content)
        # to check for multiple files
        if type(image) == list:
            # for multiple faces
            imagee = image
            final_result = []
            predictions = []
            for i in range(len(imagee)):
                dumdum = await preprocessing(imagee[i])
                prediction = await make_predictions(dumdum, "image")        
                predictions.append(prediction)
                interpretability_result:np.ndarray = await interpretability(imagee[i], prediction, dumdum)
                img_io = ndarray_to_image(interpretability_result)
                final_result.append(img_io)

            return JSONResponse(content={"interpret": final_result, "multiple": "yes", "key": "image", "pred": predictions})

        img = await preprocessing(image) 
        prediction = await make_predictions(img, "image")
        interpretability_result:np.ndarray = await interpretability(image, prediction, img)
        img_io = ndarray_to_image(interpretability_result)
        print(type(img_io))
        payload = {"Prediction": str(prediction)}
        # file_response = StreamingResponse(img_io, media_type="image/png")
        p_type = "Image"
        # for singular face
        return JSONResponse(content={"result": img_io, "prediction": prediction, "key": "image"})
    # video
    else:
        HTTPException(status_code=422, detail="Unsupported File Format.")
           

    return {"file_uploaded": prediction, "Type": p_type}

import json
@app.post("/upload_video/")
async def upload_file_and_json(file: UploadFile = File(...), json_data:str = Form(...)):
    file_content = await file.read()
    data = json.loads(json_data)
    filename = file.filename
    extension = "." in filename and filename.rsplit(".", 1)[1].lower()
    if extension in VIDEO_EXTENSIONS:
        p = await video_processing(file_content, data={"ext": ".mp4", "END": int(data["END"]), "START": int(data["START"])})
        if p["completed"]:
            interpretability_result:np.ndarray = await interpretability(p["PIL_IMAGE"], p["Image"], p["TENSORS"])
            img_io = ndarray_to_image(interpretability_result)
            # this is for sigular face in video
            return JSONResponse(content={"result": img_io, "image_prediction": p["Image"], "audio_prediction": p["audio"], "key": "video"})
            payload = {"Image_Pred": str(p["Image"]), "Audio_Pred": str(p["audio"])}

            return Response(content = img_io.getvalue(), headers = payload, media_type="image/png")
        else:
            n = await generate_images(p["total_faces"])
            # this is for multiple faces in video
            return JSONResponse(content={"images": n, "process": "multiple faces", "key": "video"})

    

#added code lines 
@app.post("/process-selected-image/")
async def process_selected_image(data: dict = Body(...)):
    file_content = await data.read()
    print(file_content)
    selected_image = "dwdfwf"
    image_data = base64.b64decode(selected_image)
    image = Image.open(BytesIO(image_data))
    image = image.convert('RGB')
    img = await preprocessing(image)
    prediction = await make_predictions(img, "image")
    interpretability_result: np.ndarray = await interpretability(image, prediction, img)
    img_io = ndarray_to_image(interpretability_result)
    return JSONResponse(content={"result": img_io, "prediction": prediction})