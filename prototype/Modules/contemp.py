from tempfile import NamedTemporaryFile
from mtcnn import MTCNN
from PIL import Image
import torchvision.transforms as transforms

MINIMUM_CONFIDENCE = 0.95

detector = MTCNN()


async def retTempFile(file_content, suffixg):
    with NamedTemporaryFile(suffix=suffixg, delete=False) as temp_file:
        temp_file.write(file_content)
        temp_file_path = temp_file.name
    return temp_file_path


async def detect_faces(frame):
    faces = detector.detect_faces(frame)
    return faces


async def cropping(data, input_image):
    face_data = data
    t_faces = []
    for face in face_data:
        if not (face["confidence"] <= MINIMUM_CONFIDENCE):
            x, y, w, h = face["box"]
            x, y, w, h = max(x, 0), max(y, 0), max(w, 0), max(h, 0)
            extracted_face = input_image.crop((x, y, x + w, y + h))
            new_image = Image.new("RGB", extracted_face.size, (0, 0, 0, 0))
            new_image.paste(extracted_face, (0, 0))
            t_faces.append(new_image)
        else:
            new_image = None
    return t_faces


async def preprocessing_f(file_content):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    img = transform(file_content)
    img = img.unsqueeze(0)
    return img
