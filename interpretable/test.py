from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from PIL import Image
import io
from fastapi.responses import Response, JSONResponse
from io import BytesIO
app = FastAPI()

@app.post("/detect")
async def detect_and_return_image(image_file: UploadFile = File(...)):
    """
    Handler of /detect POST endpoint
    Receives uploaded file with a name "image_file",
    passes it through YOLOv8 object detection
    network and returns an array of bounding boxes.
    :return: a JSON array of objects bounding
    boxes in format
    [[x1,y1,x2,y2,object_type,probability],..]
    """
    buf = await image_file.read()
    boxes, class_prob = detect_objects_on_image(Image.open(BytesIO(buf)))
    print(f'class proba {class_prob}')
    annotated_image = annotate_image(Image.open(BytesIO(buf)), boxes)
    return {
        "annotated_image": image_to_base64(annotated_image),
        "class_prob": class_prob
    }


def detect_objects_on_image(image):
    """
    Function receives an image,
    passes it through YOLOv8 neural network
    and returns an array of detected objects
    and their bounding boxes
    :param image: Input image
    :return: Array of bounding boxes in format
    [[x1,y1,x2,y2,object_type,probability],..]
    """
    model = YOLO('best.pt')
    results = model.predict(image)
    result = results[0]
    output = []
    class_prob = []
    for box in result.boxes:
        x1, y1, x2, y2 = [round(x) for x in box.xyxy[0].tolist()]
        class_id = box.cls[0].item()
        prob = round(box.conf[0].item(), 2)
        output.append([x1, y1, x2, y2, result.names[class_id], prob])
        class_prob.append([result.names[class_id], prob])
    return output, class_prob


def annotate_image(image, boxes):
    """
    Function annotates the image with bounding boxes.
    :param image: Input image
    :param boxes: Array of bounding boxes in format
    [[x1,y1,x2,y2,object_type,probability],..]
    :return: Annotated image
    """
    # Draw bounding boxes on the image
    draw = ImageDraw.Draw(image)
    for box in boxes:
        x1, y1, x2, y2, object_type, probability = box
        draw.rectangle([(x1, y1), (x2, y2)], outline="red", width=3)
        draw.text((x1, y1 - 10), f"{object_type} ({probability})", fill="red")

    return image


def save_annotated_image(image):
    """
    Function saves the annotated image and returns
    the image as a response.
    :param image: Annotated image
    :return: StreamingResponse with the image
    """
    output_buffer = BytesIO()
    image.save(output_buffer, format="PNG")
    output_buffer.seek(0)
    return StreamingResponse(output_buffer, media_type="image/png")

def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')