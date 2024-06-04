from Modules import videoprocessing, imageprocessing, contemp
from fastapi import FastAPI, HTTPException, Form, File, UploadFile
import json


ALLOWED_EXTENSIONS = ["mp3", "flac", "wav", "mp4", "mov", "mkv", "jpg", "jpeg", "png"]
AUDIO_EXTENSIONS = ["mp3", "flac", "wav"]
VIDEO_EXTENSIONS = ["mp4", "mov", "mkv"]
IMAGE_EXTENSIONS = ["jpg", "jpeg", "png"]
app = FastAPI()


@app.post("/upload/")
async def main(file: UploadFile = File(...), json_data: str = Form(...)):
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
        pass
    elif extension in AUDIO_EXTENSIONS:
        pass
    else:
        raise HTTPException(
            status_code=422, detail="Unsupported File has been uploaded"
        )
    # print(file_content)
    return {"Working": 1}
