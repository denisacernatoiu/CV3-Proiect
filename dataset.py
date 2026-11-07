import os
import random
import cv2
import numpy as np
import torch
import albumentations as A # util deoarece vrem sa fie ajustate si bbox urile
import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader, Subset

SIZE = 640 # dimensiunea necesara pentru yolo
CLASS_NAMES = ["longitudinal", "transverse", "alligator", "pothole"]  


def get_transforms(mean, std):
    resize = [A.Resize(SIZE, SIZE)]
    augmentation = [A.HorizontalFlip(p=0.5), 
                    A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0, p=0.5)]
    normalize = [A.Normalize(mean=mean, std=std), ToTensorV2()]
    bbox_params = A.BboxParams(format="albumentations", label_fields=["class_labels"])

    train_transform = A.Compose(resize + augmentation + normalize, bbox_params=bbox_params)
    eval_transform = A.Compose(resize + normalize, bbox_params=bbox_params)

    return train_transform, eval_transform


class RoadDamageDataset(Dataset):

    def __init__(self, folder, transform=None):
        self.images_path = folder + "/images"
        self.labels_path = folder + "/labels"
        self.images = sorted(f for f in os.listdir(self.images_path) if f.lower().endswith((".jpg", ".jpeg", ".png")))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        image = cv2.cvtColor(cv2.imread(self.images_path + "/" + name), cv2.COLOR_BGR2RGB)
        label_path = self.labels_path + "/" + name.rsplit(".", 1)[0] + ".txt"

        rows = []
        if os.path.exists(label_path):
            with open(label_path) as file:
                rows = [[float(x) for x in line.split()] for line in file if line.strip()]

        # albumentations cere colturi intre 0 si 1, iar unele etichete au valori negative, deci le limitam
        boxes = np.clip([[x - w / 2, y - h / 2, x + w / 2, y + h / 2] for _, x, y, w, h in rows], 0, 1).tolist()
        result = self.transform(image=image, bboxes=boxes, class_labels=[int(r[0]) for r in rows])
        labels = torch.tensor([[c, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1] for c, (x1, y1, x2, y2) in zip(result["class_labels"], result["bboxes"])], dtype=torch.float32).reshape(-1, 5)

        return result["image"], labels


class RoadDamageDataLoader:

    def __init__(self, source_path, batch_size=8, augment=True, random_seed=42, sizes=(7000, 1500, 1500), num_workers=4):

        random.seed(random_seed)

        # initializam obiectele
        train_data, val_data, test_data = [RoadDamageDataset(source_path + "/" + split) for split in ("train", "val", "test")]

        # cream subseturi
        self.train_dataset, self.val_dataset, self.test_dataset = [Subset(d, random.sample(range(len(d)), min(n, len(d)))) for d, n in zip((train_data, val_data, test_data), sizes)]

        # calculam statisticile pe setul de antrenare
        train_data.transform = get_transforms((0, 0, 0), (1, 1, 1))[1]
        stats_loader = DataLoader(self.train_dataset, batch_size=32, num_workers=num_workers, collate_fn=self.collate_fn)

        total = torch.zeros(3, dtype=torch.float64)
        total_sq = torch.zeros(3, dtype=torch.float64)
        count = 0

        for images, _ in stats_loader:
            total += images.sum(dim=(0, 2, 3))
            total_sq += (images ** 2).sum(dim=(0, 2, 3))
            count += images.shape[0] * images.shape[2] * images.shape[3]

        mean = total / count
        std = torch.sqrt(total_sq / count - mean ** 2)

        self.mean, self.std = mean.tolist(), std.tolist()
        print('Normalizare calculata pe {} imagini: mean = {}, std = {}'.format(len(self.train_dataset), self.mean, self.std))

    
        train_transform, eval_transform = get_transforms(self.mean, self.std)
        train_data.transform = train_transform if augment else eval_transform
        val_data.transform = eval_transform
        test_data.transform = eval_transform

        # optimizari pentru incarcarea datelor
        loader_args = dict(batch_size=batch_size,
                           num_workers=num_workers,
                           persistent_workers=num_workers > 0,
                           pin_memory=torch.cuda.is_available(),
                           collate_fn=self.collate_fn)

        self.train_loader = DataLoader(self.train_dataset, shuffle=True, **loader_args)
        self.val_loader = DataLoader(self.val_dataset, shuffle=False, **loader_args)
        self.test_loader = DataLoader(self.test_dataset, shuffle=False, **loader_args)

    @staticmethod
    def collate_fn(batch):
        images, labels = zip(*batch)
        return torch.stack(images), list(labels)


def plot_examples(dataset, mean, std, n=3):
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

    for ax, i in zip(axes, random.sample(range(len(dataset)), n)):
        image, labels = dataset[i]

        ax.imshow((image.permute(1, 2, 0).numpy() * std + mean).clip(0, 1))
        ax.axis("off")

        for c, x, y, w, h in labels.tolist():
            left, top = (x - w / 2) * SIZE, (y - h / 2) * SIZE
            ax.add_patch(plt.Rectangle((left, top), w * SIZE, h * SIZE, fill=False, color="red"))
            ax.text(left, top, CLASS_NAMES[int(c)], color="red")

    plt.show()


if __name__ == "__main__":

    data = RoadDamageDataLoader("/kaggle/input/datasets/sreekaraditya/rdd2022-yolo-crackscan-v2")

    print("Train:", len(data.train_dataset), "| Val:", len(data.val_dataset), "| Test:", len(data.test_dataset))

    images, labels = next(iter(data.train_loader))

    plot_examples(data.train_dataset, data.mean, data.std, 3)