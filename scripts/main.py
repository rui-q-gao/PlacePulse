"""
Script to run a training and testing loop for the PlacePulse SNN.

Author: Rui Gao
Date: Jan 22, 2020
"""

import argparse
import torch
import numpy as np
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from model import SNNet
from PPPairDataset import PPPairDataset
from torch.utils.data.sampler import SubsetRandomSampler
import matplotlib.pyplot as plt
import torchvision.models as models
from torch import nn
from time import time


def train(model, device, train_loader, epoch, optimizer):
    model.train()
    print("Hi I'm training!")

    for batch_idx, (data, target) in enumerate(train_loader):
        print(1)
        for i in range(len(data)):
            data[i] = data[i].to(device)

        optimizer.zero_grad()
        target = target.type(torch.LongTensor).to(device)
        output = model(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        print(2)
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * args.batch_size, len(train_loader.dataset),
                       100. * batch_idx * args.batch_size / len(
                           train_loader.dataset), loss.item()))


def test(model, device, test_loader):
    print("Hi I'm testing!")
    model.eval()
    with torch.no_grad():
        accurate_labels = 0
        all_labels = 0
        for batch_idx, (data, target) in enumerate(test_loader):
            print(batch_idx)
            for i in range(len(data)):
                data[i] = data[i].to(device)

            target = target.type(torch.LongTensor).to(device)
            output = model(data)
            loss = F.cross_entropy(output, target)
            accurate_labels = torch.sum(
                torch.argmax(output, dim=1) == target).cpu()
            all_labels = len(target)
            accuracy = 100. * accurate_labels / all_labels
            print('Test accuracy: {}/{} ({:.3f}%)\tLoss: {:.6f}'.format(
                accurate_labels, all_labels, accuracy, loss))


def oneshot(model, device, data):
    model.eval()

    with torch.no_grad():
        for i in range(len(data)):
            data[i] = data[i].to(device)

        output = model(data)
        return torch.squeeze(torch.argmax(output, dim=1)).cpu().item()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trans = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])])
    model = SNNet()

    model.load_state_dict(models.vgg16_bn(pretrained=True).state_dict())
    for param in model.parameters():
        param.requires_grad = False

    model.classifier[6] = nn.Linear(4096, 2)
    model = model.to(device)

    if args.do_learn == 1:

        dataset = PPPairDataset(transform=trans, target_transform=None)
        indices = list(range(len(dataset)))
        split = int(round(0.8 * len(dataset)))

        np.random.seed(args.seed)
        np.random.shuffle(indices)

        train_idx, test_idx = indices[:split], indices[split:]
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)

        train_loader = torch.utils.data.DataLoader(dataset,
                                                   sampler=train_sampler,
                                                   batch_size=args.batch_size)
        test_loader = torch.utils.data.DataLoader(dataset,
                                                  sampler=test_sampler,
                                                  batch_size=len(test_idx))

        optimizer = optim.Adam(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)

        for epoch in range(args.epochs):
            train(model, device, train_loader, epoch, optimizer)
            test(model, device, test_loader)
            if epoch % args.save_frequency == 0:
                torch.save(model, 'siamese_{:03}.pt'.format(epoch))
    else:
        prediction_loader = torch.utils.data.DataLoader(
            PPPairDataset(transform=None, target_transform=None),
            batch_size=1, shuffle=True)
        model = torch.load("siamese_002.pt")
        data = next(iter(prediction_loader))[0]
        print(data[0].shape)

        win = oneshot(model, device, data)
        if win > 0:
            print('The right image is safer!')
        else:
            print('The left image is safer!')

        f = plt.figure()

        for i in range(len(data)):
            f.add_subplot(1, 2, i + 1)
            img = np.moveaxis(data[i].squeeze().numpy(), 0, -1)
            plt.imshow(img.squeeze())

        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--do_learn', type=int, default=1)
    parser.add_argument('--save_frequency', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    main(args)
