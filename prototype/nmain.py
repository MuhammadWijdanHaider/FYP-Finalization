import numpy as np
import transformers

from Modules import videoprocessing, imageprocessing, contemp
from fastapi import FastAPI, HTTPException, Form, File, UploadFile
import json


ALLOWED_EXTENSIONS = ["mp3", "flac", "wav", "mp4", "mov", "mkv", "jpg", "jpeg", "png"]
AUDIO_EXTENSIONS = ["mp3", "flac", "wav"]
VIDEO_EXTENSIONS = ["mp4", "mov", "mkv"]
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]
app = FastAPI()


@app.post("/upload/")
async def main(file: UploadFile = File(...), json_data: str = Form("{}")):
    final_ret_result: dict = {}
    file_content = await file.read()
    data = json.loads(json_data)
    filename = file.filename
    extension = "." in filename and filename.rsplit(".", 1)[1].lower()
    data["ext"] = extension
    if extension in VIDEO_EXTENSIONS:
        result = videoprocessing.video_processing(file_content, data)
        t_result = type(result)
        if t_result == list:
            return {"result": result}
        else:
            raise HTTPException(
                status_code=422, detail="Unsupported File has been uploaded"
            )
        pass

    elif extension in IMAGE_EXTENSIONS:
        result: dict = await imageprocessing.complete_preprocessing(file_content)
        result_tensor: list = result["final_tensors"]
        result_cimages: list = result["cropped_images"]
        pred: list
        final_payload: list = []
        final_embed: list = []
        for i in range(len(result_tensor)):
            pred = await contemp.make_predictions(result_tensor[i], types="image")
            processed_prediction = int(pred[0][0])
            final_payload.append(processed_prediction)
            interpretability_result: np.ndarray = await contemp.interpretability(cropped_image=result_cimages[i],
                                                                                 predicted_class=processed_prediction,
                                                                                 final_tensor=result_tensor[i])
            image_io = await contemp.ndarray_embed(interpretability_result)
            final_embed.append(image_io)
        print(final_payload)
        final_ret_result = {"Embeded_images": final_embed, "predicitions": final_payload}
    elif extension in AUDIO_EXTENSIONS:
        pass
    else:
        raise HTTPException(
            status_code=422, detail="Unsupported File has been uploaded"
        )
    # print(file_content)
    return final_ret_result
