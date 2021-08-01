#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

""" 0 Prepare dataset """
class DiabetesDataset(Dataset):
    def __init__(self):
        xy = np.loadtxt('./diabetes.csv.gz', delimiter=',', dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

dataset = DiabetesDataset()
train_loader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=2)

""" 1 Design model using class """
class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.linear1 = torch.nn.Linear(8, 6)
        self.linear2 = torch.nn.Linear(6, 4)
        self.linear3 = torch.nn.Linear(4, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        return x

model = Model()

""" 2 Construct loss and optimizer"""
criterion = torch.nn.BCELoss(reduction='mean')
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

""" 3 Train cycle """
if __name__ == "__main__":
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            # 1.prepare data
            inputs, labels = data
            # 2.forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            # 3.backward
            optimizer.zero_grad()
            loss.backward()
            # 4.update
            optimizer.step()