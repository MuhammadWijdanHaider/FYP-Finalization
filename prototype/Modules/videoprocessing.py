from Modules import contemp
from moviepy.editor import VideoFileClip, AudioFileClip

ALLOWED_LENGHT = 10


async def video_processing(file_content, data):
    clip: VideoFileClip
    # return the temp file using the extension
    cli = contemp.retTempFile(file_content, data["ext"])
    # loading the video in the memory in RAM without using harddrive, saving in the harddrive will come later
    clip = VideoFileClip(cli)
    dur = clip.duration
    new: VideoFileClip
    if dur <= ALLOWED_LENGHT:
        providedTime = data["END"] - data["START"]
        if providedTime > 0 and providedTime <= ALLOWED_LENGHT:
            new = clip.subclip(data["START"], data["END"])
        else:
            # pass out some distinct error codes to the main for raising errors
            return -1
    else:
        providedTime = data["END"] - data["START"]
        if providedTime > 0 and providedTime <= ALLOWED_LENGHT:
            new = clip.subclip(data["START"], data["END"])
        else:
            return -1
    frames = []
    m_face_d = {"face_data": []}
    # frame extraction using moviepy at the FPS of 1, can be changed based on hardware
    for frame in new.iter_frames(fps=1):
        face = contemp.detect_faces(frame)
