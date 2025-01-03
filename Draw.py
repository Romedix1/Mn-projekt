import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

#zaznacza na obrazie etykiety i wyswietla je z podpisem (model przetrenowany)
def draw(image, annotations, label_map):
    fig, x = plt.subplots(1, figsize=(8, 6))

    image = image.permute(1, 2, 0).numpy()

    x.imshow(image)

    label_map_reverse = {v: k for k, v in label_map.items()}

    for i, box in enumerate(annotations['boxes']):
        x_min, y_min, x_max, y_max = box.tolist()
        label_index = annotations['labels'][i].item()
        label = label_map_reverse[label_index]

        rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, linewidth=1, edgecolor='r', facecolor='none')
        x.add_patch(rect)
        x.text(x_min, y_min, label, color='red', fontsize=7, bbox=dict(facecolor='yellow', alpha=0.4))

    plt.show()