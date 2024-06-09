import base64
from tempfile import NamedTemporaryFile
from PIL import Image
import torchvision.transforms as transforms
import torch
from fastapi import HTTPException
from torchray.attribution.grad_cam import grad_cam
import cv2
import numpy as np
from io import BytesIO
# this is from the __init__
from . import MINIMUM_CONFIDENCE, model, model_inter, model_audio, detector


async def retTempFile(file_content, suffixg):
    with NamedTemporaryFile(suffix=suffixg, delete=False) as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name
    return temp_file_path


async def detect_faces(frame):
    faces = detector.detect_faces(frame)
    return faces


async def cropping(data, input_image):
    """Provide the face data as well as the original image"""
    face_data = data
    t_faces = []
    for face in face_data:
        if not (face["confidence"] <= MINIMUM_CONFIDENCE):
            x, y, w, h = face["box"]
            x, y, w, h = max(x, 0), max(y, 0), max(w, 0), max(h, 0)
            extracted_face = input_image.crop((x, y, x + w, y + h))
            new_image = Image.new("RGB", extracted_face.size, (0, 0, 0, 0))
            new_image.paste(extracted_face, (0, 0))
            t_faces.append(new_image)
        else:
            new_image = None
    return t_faces


async def preprocessing_f(file_content):
    # this is for transforming the image into tensor, will work on audio mel spectrograms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = transform(file_content)
    img = img.unsqueeze(0)
    return img


async def make_predictions(img, types):
    prediction: int
    if types == "audio":
        with torch.no_grad():
            output = model_audio(pixel_values=img)
        logits = output.logits
        prediction = torch.argmax(logits).item()
    elif types == "image":
        # with torch.no_grad():
        #     outputs = model(img)
        # logits = outputs.logits
        # prediction = torch.argmax(logits).item()
        with torch.no_grad():
            outputs = model(img)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
        probabilities = torch.sigmoid(logits)
        prediction = (probabilities > 0.5).float().cpu().numpy()
    else:
        raise HTTPException(status_code=500, detail="Internal Error")
    return prediction


async def ndarray_embed(nda: np.ndarray) -> str:
    image = Image.fromarray(nda.astype('uint8'), 'RGB')
    img_io = BytesIO()
    image.save(img_io, format='jpeg')
    img_io.seek(0)
    encoded_image = base64.b64encode(img_io.getvalue()).decode('utf-8')
    return encoded_image


async def interpretability(cropped_image, final_tensor, predicted_class):
    grad_cam_result = grad_cam(model_inter, final_tensor, saliency_layer='model.block12.rep.4.conv1', target=predicted_class)
    grad_cam_heatmap = grad_cam_result[0].detach().numpy().sum(axis=0)
    resized_heatmap = cv2.resize(grad_cam_heatmap, (224, 224))
    # change this if error occurs
    nd = np.array(cropped_image, dtype=np.uint8)
    input_image = cv2.cvtColor(nd, cv2.COLOR_RGBA2RGB)
    input_image = cv2.resize(input_image, (224, 224))
    resized_heatmap = (resized_heatmap - resized_heatmap.min()) / (resized_heatmap.max() - resized_heatmap.min() + 1e-8)
    heatmap_rgb = cv2.applyColorMap(np.uint8(255 * resized_heatmap), cv2.COLORMAP_JET)
    overlayed_image = cv2.addWeighted(input_image, 0.5, heatmap_rgb, 0.5, 0)
    p = cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)
    return p
