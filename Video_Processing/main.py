import os
import imageio
from moviepy.editor import VideoFileClip

# for face detection
from mtcnn import MTCNN
import cv2
import pprint

import numpy as np

from PIL import Image
import torchvision.transforms as transforms
from torchvision import transforms
def frames_extraction(st_time, en_time, file, intervals=1):
    frames = []
    try:
        clip = file
    except Exception as e:
        print(e)
    pass

def extract_frames_rd(path, output, intervals = 1):
    frames = []
    detector = MTCNN()    
    m_face = []
    m_face_d = {"face_data": []}
    ff = 0
    try:
        clip = VideoFileClip(path)
        for frame in clip.iter_frames(fps=1):
            frames.append(frame)
            # do some checks to make sure that the face is in good condition, (later)
            f = detector.detect_faces(frame)
            # pprint.pprint(len(f))
            if len(f) > len(m_face_d["face_data"]):
                m_face_d["face_data"] = f
                m_face_d["frame"] = frame
                # for the sake of testing
            # face extraction testing without saving image for further processing and detection
        clip.close()
        face_data = m_face_d["face_data"]
        # print("HERE")
        # print(face_data)
        x, y, w, h = face_data[0]['box']
        x, y, w, h = max(x, 0), max(y, 0), max(w, 0), max(h, 0)
        cropped_face = m_face_d["frame"][y:y+h, x:x+w]
        imageio.imwrite("extracted_frame.png", m_face_d["frame"])
        print(type(m_face_d["frame"]))
        cropped_face_shape = cropped_face.shape
        cropped_face_size = (cropped_face_shape[1], cropped_face_shape[0])
        print(cropped_face_size)
        new_image = Image.new("RGB", cropped_face_size, (0, 0, 0, 0))
        new_image.paste(Image.fromarray(cropped_face), (0, 0))
        new_image.save("FACE.jpg")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224), antialias=True)
        ])
        img = transform(new_image)
        img = img.unsqueeze(0)
        # img.save("FACE_N.jpg")
    except Exception as e:
        print(e)


extract_frames_rd(r"testing_data\videos\two_face.mp4", "testing_data\videos")

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
            print(f"Type: {type(p)}\nContent: {p}")
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
    img = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)
    detector = MTCNN()
    p = detector.detect_faces(img)
    # img = cv2.cvtColor(cv2.imread(frame), cv2.COLOR_BGR2RGB)
    print(type(p[0]))
    

# extract_frames(r"testing_data\\videos\\test.mp4", r"testing_data\\videos\\frames")
# face_detection(r"testing_data\\t2.jpg")

