from fastapi import FastAPI, Response, File, Form, UploadFile, websockets
from mtcnn import MTCNN
from moviepy.editor import VideoFileClip


from fastapi.middleware.cors import CORSMiddleware

class frame:
    def __init__(self, frame, id):
        self.frame = frame
        self.id = id

    def check_area(self):
        clip = VideoFileClip(self.frame)
        frame = clip.get_frame(0)
        detector = MTCNN()
        faces = detector.detect_faces(frame)
        if len(faces) > 0:
            x, y, w, h = faces[0]['box']
            return x, y, w, h
        else:
            return None
        clip.close()

        pass

app = FastAPI()



app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from all origins (replace with specific origins if needed)
    allow_credentials=True,
    allow_methods=["GET", "POST"],  # Allow GET and POST requests (add other methods as needed)
    allow_headers=["*"],  # Allow all headers (you can customize this based on your requirements)
)

# WebSocket route
@app.websocket("/ws/{scenario}")
async def websocket_endpoint(websocket: websockets.WebSocket, scenario: str, file: UploadFile = File(...), json_data:str = Form(default='{"END": "0", "START": "0"}')):
    await websocket.accept()

    # Send a message to the client
    if scenario == "final":
        await websocket.send_text("Final request")
    else:
        while True:
            data = await websocket.receive_text()
            if data == "Something":
                await websocket.send_text("Response from the server")
                break
            else:
                await websocket.send_text("Invalid request")
                await websocket.close()
