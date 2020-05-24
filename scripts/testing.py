import torch
from torchvision import transforms
from model import *
from BalancedDataset import BalancedPairDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
trans = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (1.0,))])

model = SNNet().to(device)

train_set = BalancedPairDataset('../data', train=True, download=True,
                                transform=trans)
print(type(train_set[0]), len(train_set[0]))
print(type(train_set[0][0]), len(train_set[0][0]))
print(type(train_set[0][1]), len(train_set[0][1]))
for i in train_set[0][0]:
    print(type(i), i.shape)

print(train_set.train_data[0].shape)

train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=64,
            shuffle=True)
data = []
data.extend(next(iter(train_loader)))
target = data[1]
data = data[0]
print(torch.squeeze(target[:, 0]))
print(data[0].shape)
print(data[1].shape)
