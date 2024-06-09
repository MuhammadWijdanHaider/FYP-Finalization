from moviepy.editor import VideoFileClip, AudioFileClip
from .contemp import retTempFile, detect_faces, cropping
from fastapi import HTTPException
from . import MAXIMUM_LENGHT
import asyncio


async def video_processing(file_content, required):
    clip: VideoFileClip
    new: VideoFileClip
    tem = await retTempFile(file_content=file_content, suffixg=required["ext"])
    clip = VideoFileClip(tem)
    duration = clip.duration
    # requires some work after talking with the frontend guy, but I am keeping it this way for the foreseeable future
    providedTime = required["END"] - required["START"]
    if 0 < providedTime <= MAXIMUM_LENGHT:
        new = clip.subclip(required["START"], required["END"])
        return new
    else:
        raise HTTPException(status_code=400,
                            detail="The provided video and timestamps exceed the limit, which is 10 seconds. "
                                   "Please set the time stamps accordingly")


async def frame_extraction_and_detection(file):
    m_face_data: dict = {"face_data": [], "image": [], "cropped_images": []}
    final_return = {"cropped_images": [], "final_tensors": []}

    # frame extraction using moviepy at the FPS of 1, can be changed based on hardware
    for frame in file.iter_frames(fps=1):
        print(type(frame))
        face = await detect_faces(frame)
        # this make sure that only the frames with images be selected
        if len(face) != 0:
            m_face_data["face_data"].append(face)
            m_face_data["image"].append(frame)

    # cropping the face
    if len(m_face_data["face_data"]) == 0:
        raise HTTPException(status_code=400, detail="No face detected in the video.")
    for i in range(len(m_face_data["face_data"])):
        f: list = await cropping(m_face_data["face_data"][i], m_face_data["image"][i])
        if len(f) == 0:
            pass
        else:
            final_return["cropped_images"].append(f)
    return m_face_data


async def complete_video_processing(file_content, required):
    result: VideoFileClip = await video_processing(file_content, required)
    audio: AudioFileClip = result.audio
    # start the first ayncio task
    frame_extraction = asyncio.create_task(frame_extraction_and_detection(result))

