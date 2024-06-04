from tempfile import NamedTemporaryFile
from mtcnn import MTCNN
from PIL import Image


MINIMUM_CONFIDENCE = 0.95

detector = MTCNN()


async def retTempFile(file_content, suffixg):
    with NamedTemporaryFile(suffix=suffixg, delete=False) as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name
    return temp_file_path


def detect_faces(frame):
    faces = detector.detect_faces(frame)
    return faces


def cropping(data):
    face_data = data
    # face_data = m_face_d["face_data"]
    if not (face_data["confidence"] < MINIMUM_CONFIDENCE):
        x, y, w, h = face_data[0]["box"]
        x, y, w, h = max(x, 0), max(y, 0), max(w, 0), max(h, 0)
        cropped_face = face_data["frame"][y : y + h, x : x + w]
        cropped_face_shape = cropped_face.shape
        cropped_face_size = (cropped_face_shape[1], cropped_face_shape[0])
        new_image = Image.new("RGB", cropped_face_size, (0, 0, 0, 0))
        new_image.paste(Image.fromarray(cropped_face), (0, 0))
    else:
        new_image = None
    return new_image
