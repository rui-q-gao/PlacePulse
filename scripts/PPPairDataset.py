"""
A module for the PPPairDataset, used in a simple MNIST siamese nn
implementation.

Author: Rui Gao
Date: Jan 22, 2020
"""
import csv
import numpy as np
import pandas as pd
import os
from torchvision import transforms
from PIL import Image
import torch
from time import time


class PPPairDataset(torch.utils.data.Dataset):
    image_folder = "download"
    data_folder = "../data"
    votes = "safety_votes_overfit.csv"

    def __init__(self, transform=None, target_transform=None, lazy=True):
        self.comparisons = pd.read_csv(os.path.join(self.data_folder, self.votes))
        self.transforms = transform
        self.target_transforms = target_transform
        self.trans = transforms.ToTensor()
        self.lazy = lazy
        if not self.lazy:
            self.data = []
            self.labels = []
            count = 0
            with open(os.path.join(self.data_folder, self.votes)) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    count += 1
                    left = Image.open(
                        os.path.join(self.image_folder, row['left_id']) + ".jpg")
                    left = self.trans(left)
                    right = Image.open(
                        os.path.join(self.image_folder, row['right_id']) + ".jpg")
                    right = self.trans(right)
                    self.data.append(torch.stack([left, right]))
                    self.labels.append(int(row["winner"]))
                    if count % 200 == 0:
                        print("Progress: ", count)



    def __getitem__(self, index):
        if not self.lazy:
            imgs, target = self.data[index], self.labels[index]
            img_ar = []
            trans = transforms.ToTensor()
            for i in range(len(imgs)):
                im = (imgs[i].numpy() * 255).astype('uint8')
                img = Image.fromarray(np.moveaxis(im, 0, -1))

                if self.transforms is not None:
                    img = self.transforms(img)
                else:
                    img = trans(img)
                img_ar.append(img)

            if self.target_transforms is not None:
                target = self.target_transforms(target)
        else:
            left_path = os.path.join(self.image_folder, self.comparisons['left_id'].iloc[index]) + ".jpg"
            left = Image.open(left_path)
            right_path = os.path.join(self.image_folder, self.comparisons['right_id'].iloc[index]) + ".jpg"
            right = Image.open(right_path)
            if self.transforms is not None:
                left_f = self.transforms(left)
                right_f = self.transforms(right)
            else:
                left_f = self.trans(left)
                right_f = self.trans(right)
            img_ar = [left_f, right_f]
            target = self.comparisons['winner'].iloc[index]

            if self.target_transforms is not None:
                target = self.target_transforms(target)

        return img_ar, target

    def __len__(self):
        return len(self.comparisons)

# trans = transforms.Compose(
#     [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],
#                              [0.229, 0.224, 0.225])])
# train_set = PPPairDataset(transform=trans, target_transform=None, lazy=True)

# print(train_set[0][0][0].shape)
# train_loader = torch.utils.data.DataLoader(
#             train_set, batch_size=64,
#             shuffle=True)
#
# print(type(train_set[0]), len(train_set[0]))
# print(type(train_set[0][0]), len(train_set[0][0]))
# print(type(train_set[0][1]), train_set[0][1])
#
# data = []
# data.extend(next(iter(train_loader)))
# target = data[1]
# data = data[0]
# print(target)
# print(data[0].shape)
# print(data[1].shape)

