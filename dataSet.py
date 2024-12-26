import os
import json

import numpy as np
import matplotlib.pyplot as plt 

from PIL import Image

import torch
import torchvision.transforms as transforms

from matplotlib.patches import Rectangle

class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, img_path, label_path, transform=None):
        self.img_path = img_path
        self.label_path = label_path
        self.images = os.listdir(img_path)
        self.transform = transform

    #wczytanie pliku json z etykietami
    def load_annotations(self, image_name):
        label_file = os.path.join(self.label_path, f"{os.path.splitext(image_name)[0]}.json")
        if os.path.exists(label_file):
            with open(label_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            for image_data in data.get('images', []):
                if image_data.get('image_id') == image_name:
                    return image_data.get('objects', [])
        return []

    #ilosc zdjec w folderze
    def __len__(self):
        return len(self.images)

    #pobiera obraz o podanym indexie i zwraca obraz, etykiety i nazwe
    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.img_path, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        annotations = self.load_annotations(image_name)

        return image, annotations, image_name

#zaznacza na obrazie etykiety i wyswietla je z podpisem
def draw(image, annotations):
    fig, x = plt.subplots(1, figsize=(8, 6))

    image = image.permute(1, 2, 0).numpy()

    x.imshow(image)

    for annotation in annotations:
        label = annotation['label']
        bbox = annotation['bbox']
        x_min, y_min, x_max, y_max = bbox

        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
        x.add_patch(rect)
        x.text(x_min, y_min, label, color='red', fontsize=7, bbox=dict(facecolor='yellow', alpha=0.4))

    plt.show()


transform = transforms.Compose([
    transforms.Resize((1080, 1920)),
    transforms.ToTensor()
])

img_path = './zdjecia'
label_path = './etykiety'

dataset = ImageDataset(img_path, label_path, transform=transform)

print('Liczba obrazow w zbiorze:', len(dataset))

random_index = np.random.randint(0, len(dataset))
image, annotations, image_name = dataset[random_index]

print("Obraz nr:", random_index)
print("Etykiety dla obrazu:", annotations)

draw(image, annotations)
