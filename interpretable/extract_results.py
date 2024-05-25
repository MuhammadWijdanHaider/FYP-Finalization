import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from torchray.attribution.grad_cam import grad_cam
import cv2
from torchray.attribution.guided_backprop import guided_backprop
from torchray.attribution.gradient import gradient
from torchray.attribution.linear_approx import linear_approx
model_path = r"Models\xception_based_Celebdf-10-regular.pth"

model = torch.load(model_path, map_location = 'cpu')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_eval = torch.load(r"Models\celebdf_final_model.pth", map_location=device)
model_eval.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
import os, random
# randomly selecting files
fake_path = r"images\fake"
all_files = os.listdir(fake_path)

#file_paths = [os.path.join(all_files, file) for file in all_files if os.path.isfile(os.path.join(all_files, file))]
random_files = file_paths = [
    r"images\fake\000_id5_id59_0007.mp4.jpg",
    r"images\fake\039_id6_id0_0009.mp4.jpg",
    r"images\fake\039_id5_id60_0004.mp4.jpg",
    r"images\fake\038_id6_id0_0005.mp4.jpg",
    r"images\fake\038_id4_id37_0007.mp4.jpg"
]
saliency_layer = 'model.block12.rep.4.conv1'
image_for_processing = []
for i in range(len(random_files)):
    image = Image.open(random_files[i]).convert("RGB")
    input_image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model_eval(pixel_values=input_image)  # Specify pixel_values
    
    logits = output.logits
    prediction = torch.argmax(logits).item()
    grad_cam_result = grad_cam(model, input_image, saliency_layer=saliency_layer, target=prediction)
    grad_cam_heatmap = grad_cam_result[0].detach().numpy().sum(axis=0)

    resized_heatmap = cv2.resize(grad_cam_heatmap, (224, 224))

    input_image = plt.imread(random_files[i])
    input_image = cv2.cvtColor(input_image, cv2.COLOR_RGBA2RGB)
    input_image = cv2.resize(input_image, (224,224))
    resized_heatmap = (resized_heatmap - resized_heatmap.min()) / (resized_heatmap.max() - resized_heatmap.min() + 1e-8)
    heatmap_rgb = cv2.applyColorMap(np.uint8(255 * resized_heatmap), cv2.COLORMAP_JET)

    # Overlay the heatmap onto the input image
    overlayed_image = cv2.addWeighted(input_image, 0.5, heatmap_rgb, 0.5, 0)

    # Visualize the overlayed image
    plt.imshow(cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    p = cv2.cvtColor(overlayed_image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(p)
    image.save(f'image{i}.png')









