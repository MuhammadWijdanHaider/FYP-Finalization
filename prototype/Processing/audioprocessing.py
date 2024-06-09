import librosa
from .contemp import retTempFile, preprocessing_f, make_predictions
import cv2
from . import AUDIO_DIM
from PIL import Image
import numpy as np
from librosa import feature
from fastapi import HTTPException


async def audio_processing(file_content, required):
    aud_file = file_content
    if required["mint"]:
        aud_file_r = await retTempFile(aud_file, suffixg=required["filename"])
        y, sr = librosa.load(aud_file_r)
    else:
        pass
    try:
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=175)
        S_dB = librosa.power_to_db(S)
        resized = cv2.resize(S_dB, AUDIO_DIM)
        normalized_image = (resized - np.min(resized)) * (255.0 / (np.max(resized) - np.min(resized)))
        normalized_image = normalized_image.astype('uint8')
        image = Image.fromarray(normalized_image)
        image = image.convert('RGB')
        return image
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def complete_audio_processing(file_content, required):
    result = await audio_processing(file_content, required)
    preprocessed_result = await preprocessing_f(result)
    pred = await make_predictions(preprocessed_result, types="audio")
    return pred
