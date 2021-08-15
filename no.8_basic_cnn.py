#!/usr/bin/python
# -*- coding: UTF-8 -*-

''' basic cnn'''
# import torch
#
# ''' input & convolution & output '''
# in_channels, out_channels = 5, 10
# width, height = 100, 100
# kernel_size = 3
# batch_size = 1
#
# # 生成数据
# input = torch.randn(batch_size,
#                     in_channels,
#                     width,
#                     height)
#
# # 卷积层
# convolution_layer = torch.nn.Conv2d(in_channels,
#                                     out_channels,
#                                     kernel_size=kernel_size)
#
# # 输出
# output = convolution_layer(input)
#
# print(input.shape)
# print(output.shape)
# print(convolution_layer.weight.shape) # 卷积核的大小 m, n, kernel_size


''' padding 填充'''
# import torch
#
# input = [3,4,6,5,7,
#          2,4,6,8,2,
#          1,6,7,8,4,
#          9,7,4,6,2,
#          3,7,5,4,1]
#
# input = torch.Tensor(input).view(1, 1, 5, 5)
#
# convolution_layer = torch.nn.Conv2d(1, 1, kernel_size=(3, 3), padding=1, bias=False)
#
# # 定义卷积核
# kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1,1,3,3) #view(输出通道数，输入通道数，kernel_width, kernel_height)
# convolution_layer.weight.data = kernel.data
#
# output = convolution_layer(input)
# print(output)


''' stride 步长'''
# import torch
#
# input = [3,4,6,5,7,
#          2,4,6,8,2,
#          1,6,7,8,4,
#          9,7,4,6,2,
#          3,7,5,4,1]
#
# input = torch.Tensor(input).view(1, 1, 5, 5)
#
# convolution_layer = torch.nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(2, 2), bias=False)
#
# # 定义卷积核
# kernel = torch.Tensor([1,2,3,4,5,6,7,8,9]).view(1,1,3,3) #view(输出通道数，输入通道数，kernel_width, kernel_height)
# convolution_layer.weight.data = kernel.data
#
# output = convolution_layer(input)
# print(output)

''' max pooling layer'''
# import torch
#
# input = [3,4,6,5,
#          2,4,6,8,
#          1,6,7,8,
#          9,7,4,6,]
#
# input = torch.Tensor(input).view(1, 1, 4, 4)
#
# max_pooling_layer = torch.nn.MaxPool2d(kernel_size=2)
#
# output = max_pooling_layer(input)
# print(output)

''' *** A Simple Convolutional Neural Network '''
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 0 prepare dataset
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.0137, ), (0.03081,))
])

train_dataset = datasets.MNIST(root='./dataset/mnist/',
                               train=True,
                               download=True,
                               transform=transform)

train_loader = DataLoader(dataset=train_dataset,
                          shuffle=True,
                          batch_size=batch_size)

test_dataset = datasets.MNIST(root='./dataset/mnist/',
                              train=False,
                              download=True,
                              transform=transform)

test_loader = DataLoader(dataset=test_dataset,
                         shuffle=False,
                         batch_size=batch_size)

# 1 design model
class  CnnNet(torch.nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=(5, 5))
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=(5, 5))
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        batch_size = x.size(0) # flatten data from (1,n,28,28) to (n, 784)
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1) # flatten
        x = self.fc(x)
        return x

model = CnnNet()
# define device as the first visible cuda device if we have CUDA available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 2 construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

# 3 train and test
def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader):
        inputs, target = data
        # send inputs & targets at every step to the GPU
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch+1, batch_idx+1, running_loss))
            running_loss = 0.0

def test(epoch):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Accuracy on test set: %d %%' % (100 * correct / total))

if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test(epoch)