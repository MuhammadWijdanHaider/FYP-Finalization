import torch
from mtcnn import MTCNN


MINIMUM_CONFIDENCE = 0.95
AUDIO_DIM = (175, 175)
model_path = r"Models\bce_final_model_epoch3.pth"
model_path_audio = r"Models\audio_model_epoch5.pth"
model_inter_path = r"Models\xception.pth"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_audio = torch.load(model_path_audio, map_location=device)
model = torch.load(model_path, map_location=device)
model_inter = torch.load(model_inter_path, map_location=device)
model.eval()
model_audio.eval()
detector = MTCNN()
