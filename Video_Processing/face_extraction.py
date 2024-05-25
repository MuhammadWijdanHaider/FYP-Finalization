from mtcnn import MTCNN
import pprint
import cv2
from PIL import Image
n = MTCNN()

# def cropping(data):
#     face_data = data
#     # face_data = m_face_d["face_data"]
#     if not(face_data['confidence'] < 0.9):
#         x, y, w, h = face_data[0]['box']
#         x, y, w, h = max(x, 0), max(y, 0), max(w, 0), max(h, 0)
#         cropped_face = m_face_d["frame"][y:y+h, x:x+w]
#         cropped_face_shape = cropped_face.shape
#         cropped_face_size = (cropped_face_shape[1], cropped_face_shape[0])
#         new_image = Image.new("RGB", cropped_face_size, (0, 0, 0, 0))
#         new_image.paste(Image.fromarray(cropped_face), (0, 0))
#     pass

def croppingV2(data):
     # get the required data from the dictionary 'data'
    x, y, w, h = data['box']
    x, y, w, h = max(x, 0), max(y, 0), max(w, 0), max(h, 0)
    face_image = data['image'][y:y+h, x:x+w]
    cropped_face_shape = face_image.shape
    cropped_face_size = (cropped_face_shape[1], cropped_face_shape[0])
    new_image = Image.new("RGB", cropped_face_size, (0, 0, 0, 0))
    new_image.paste(Image.fromarray(face_image), (0, 0))
    face_image_pil = Image.fromarray(face_image)
    face_image_pil.show()
    new_image.save("m1.jpg")

path = r"frame_16.jpg"
img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
p = n.detect_faces(img)
pprint.pprint(p)
# l = n.detect_faces(img)
for i in range(len(p)):
    p[i]['image'] = img
    croppingV2(p[i])
    

