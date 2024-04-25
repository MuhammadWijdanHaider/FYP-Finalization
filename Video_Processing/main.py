import os
import imageio
from moviepy.editor import VideoFileClip

# for face detection
from mtcnn import MTCNN
import cv2


def frames_extraction(st_time, en_time, file, intervals=1):
    frames = []
    try:
        clip = file
    except Exception as e:
        print(e)
    pass


def extract_frames(path, output_dir, intervals=1):
    frames = []
    try:
        clip = VideoFileClip(path)
        print(type(clip))
        duration = clip.duration
        fps = clip.fps
        frame_indices = [int(fps * i) for i in range(0, int(duration), intervals)]
        # print(frame_indices)
        # return
        for idx in frame_indices:
            frame = clip.get_frame(idx / fps)
            frames.append(frame)
            # break
        clip.close()
        print("Frames extracted successfully.")

        detector = MTCNN()
        print(detector.detect_faces(frames[0]))

    except Exception as e:
        print(e)
    pass

def face_detection(frame):
    # img = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)
    print("Name")
    pass

extract_frames(r"testing_data\\videos\\test.mp4", r"testing_data\\videos\\frames")