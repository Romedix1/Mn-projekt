import torch
from torchvision import models, transforms
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from DataSet import dataset, label_map

num_classes = len(label_map) + 1

model = models.detection.fasterrcnn_resnet50_fpn(weights=None)

in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

model.load_state_dict(torch.load('epochs2.pth'))
model.eval()

random_index = np.random.randint(0, len(os.listdir('./zdjecia')))
image_path = f'./zdjecia/zdjecie{random_index}.png'
image = Image.open(image_path)

transform = transforms.Compose([transforms.ToTensor()])
image_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    prediction = model(image_tensor)

boxes, labels, scores = prediction[0]['boxes'], prediction[0]['labels'], prediction[0]['scores']

fig, x = plt.subplots(1, figsize=(8, 6))
x.imshow(image)

for box, label, score in zip(boxes, labels, scores):
    if score > 0.2:
        xmin, ymin, xmax, ymax = box.tolist()
        rect = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,linewidth=2, edgecolor='r', facecolor='none')
        x.add_patch(rect)
        x.text(xmin, ymin, f'{label.item()} ({score:.2f})', color='r', fontsize=10, bbox=dict(facecolor='white', alpha=0.5))

plt.show()
