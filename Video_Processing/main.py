import os
import imageio
from moviepy.editor import VideoFileClip

# for face detection
from mtcnn import MTCNN
import cv2
import pprint

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
        for frame in clip.iter_frames(fps=1):
            frames.append(frame)
        frame_indices = [int(fps * i) for i in range(0, int(duration), intervals)]
        print(len(frame_indices))
        print(len(frames))
        print(type(frames[0]))
        # print(frame_indices)
        # return
        frames = []
        for idx in frame_indices:
            frame = clip.get_frame(idx / fps)
            frames.append(frame)
        mt = MTCNN()
        ioi = []
        for i in range(len(frames)):
            p = mt.detect_faces(frames[i])
            print(p)
            ioi.append(p)
        # print(ioi)
        pprint.pprint(ioi)
        
        # print(type(frames[0]))
    #     clip.close()
    #     print("Frames extracted successfully.")

    #     detector = MTCNN()
    #     print(detector.detect_faces(frames[0]))

    except Exception as e:
        print(e)
    pass

def face_detection(frame):
    # img = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)
    print("Name")
    pass

extract_frames(r"testing_data\\videos\\test.mp4", r"testing_data\\videos\\frames")