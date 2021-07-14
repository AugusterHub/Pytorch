#!/usr/bin/python
# -*- coding: UTF-8 -*-

""" 3 反向传播 back propagation"""
import torch

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = torch.Tensor([1.0]) # 初始化斜率
w.requires_grad = True  # 需要计算梯度
a = 0.01 # 初始化学习率

def forward(x):
    # 前向传播
    return w * x

def loss(x, y):
    # 损失函数
    y_pred = forward(x)
    return (y - y_pred) ** 2

print('predict (before training)', 4, forward(4).item())

for epoch in range(100):
    for x, y in zip(x_data, y_data):
        l = loss(x, y)
        l.backward()
        print('\tgrad:', x, y, w.grad.item())
        w.data = w.data - a * w.grad.data
        w.grad.data.zero_()
    print('process:', epoch, l.item())
print('predict (after training)', 4, forward(4).item())

