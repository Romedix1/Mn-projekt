import os
import json
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision
from Draw import draw

class ImageDataset(torch.utils.data.Dataset):
    #konstruktor
    def __init__(self, img_path, label_path, transform=None):
        self.img_path = img_path
        self.label_path = label_path
        self.images = os.listdir(img_path)
        self.transform = transform
        self.label_map = None

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

    #pobiera obraz o podanym indexie i zwraca obraz, etykiety
    def __getitem__(self, index):
        image_name = self.images[index]
        image_path = os.path.join(self.img_path, image_name)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        annotations = self.load_annotations(image_name)

        boxes = []
        labels = []

        for annotation in annotations:
            bbox = annotation['bbox']
            label = annotation['label']
            if label in self.label_map:
                boxes.append(bbox)
                labels.append(self.label_map[label])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {
            'boxes': boxes,
            'labels': labels,
        }

        return image, target

#tworzy liste unikatowych etykiet
def create_label_map(label_path):
    label_set = set()

    for label_file in os.listdir(label_path):
        if label_file.endswith(".json"):
            with open(os.path.join(label_path, label_file), 'r', encoding='utf-8') as f:
                data = json.load(f)
                for image_data in data.get('images', []):
                    for obj in image_data.get('objects', []):
                        label_set.add(obj['label'])

    label_map = {label: idx + 1 for idx, label in enumerate(sorted(label_set))}
    # print("Label map:", label_map)
    return label_map

transform = transforms.Compose([
    transforms.Resize((1080, 1920)),
    transforms.ToTensor()
])

img_path = './zdjecia'
label_path = './etykiety'

label_map = create_label_map(label_path)

dataset = ImageDataset(img_path=img_path, label_path=label_path, transform=transform)

dataset.label_map = label_map

print('Liczba obrazow w zbiorze:', len(dataset))

random_index = np.random.randint(0, len(dataset))
image, annotations = dataset[4]

# print("Obraz nr:", random_index)

# draw(image, annotations, label_map)