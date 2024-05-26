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

def print_accessible_layers(model, prefix=''):
    for name, module in model.named_children():
        module_name = f"{prefix}.{name}" if prefix else name
        print(f"Module name: {module_name}")
        print(f"Module type: {type(module)}")
        print()
        if isinstance(module, torch.nn.Module):
            print_accessible_layers(module, module_name)

def is_layer_available(model, layer_name):
    for name, module in model.named_modules():
        if name == layer_name:
            return True
    return False

# Example usage
# layer_name = 'model.conv4.conv1'
# if is_layer_available(model, layer_name):
#     print(f"Layer '{layer_name}' is available in the model.")
# else:
#     print(f"Layer '{layer_name}' is not available in the model.")

# Example usage
# print_accessible_layers(model)


# print(model)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

img_path = r"m4.jpg"
image = Image.open(img_path).convert("RGB")
input_image = transform(image).unsqueeze(0)

with torch.no_grad():
        output = model_eval(pixel_values=input_image)  # Specify pixel_values
    
logits = output.logits
prediction = torch.argmax(logits).item()
print(prediction)


with torch.no_grad():
    output = model(input_image)
predicted_class = output.argmax(dim=1).item()

print(predicted_class)

saliency_layer = 'model.block12.rep.4.conv1'

#predicted_class = output.argmax(dim=1)

# Perform Grad-CAM attribution
grad_cam_result = grad_cam(model, input_image, saliency_layer=saliency_layer, target=prediction)
grad_cam_heatmap = grad_cam_result[0].detach().numpy().sum(axis=0)



# Function to display the heatmap
def display_heatmap(heatmap, title):
    plt.imshow(heatmap, cmap='jet')
    plt.title(title)
    plt.axis('off')
    plt.show()

# Display the heatmaps
# display_heatmap(grad_cam_heatmap, 'Grad-CAM Heatmap')

resized_heatmap = cv2.resize(grad_cam_heatmap, (224, 224))

input_image = plt.imread(img_path)
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
image.save('m4_grad.png')
# Visualize the resized heatmap
# plt.imshow(resized_heatmap, cmap='jet')
# plt.axis('off')
# plt.show()

'''This is my territory'''






