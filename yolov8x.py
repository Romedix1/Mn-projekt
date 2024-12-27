from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

model = YOLO("yolov8x.pt")

random_index = np.random.randint(0, len(os.listdir('./zdjecia')))

results = model(f'./zdjecia/zdjecie{random_index}.png')

img = results[0].plot()

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()