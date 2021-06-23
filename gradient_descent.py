#!/usr/bin/python
# -*- coding: UTF-8 -*-

""" 1 梯度下降&随机梯度下降(SGD)"""
import numpy as np
import matplotlib.pyplot as plt

x_data = [1.0, 2.0, 3.0]
y_data = [2.0, 4.0, 6.0]

w = 1.0 # 初始化斜率
a = 0.1 # 初始化学习率

def forward(x):
    # 前向传播
    return w * x

def cost(xs, ys):
    # 损失函数
    cost = 0
    for x, y in zip(xs, ys):
        y_pred = forward(x)
        cost += (y - y_pred) ** 2
    return cost / len(xs)

def gradient(xs, ys):
    # 梯度
    grad = 0
    for x, y in zip(xs, ys):
        grad += 2 * x * (x * w - y)
    return grad / len(xs)

cost_list = []
grad_list = []
for epoch in range(10):
    cost_val = cost(x_data, y_data)
    cost_list.append(cost_val)
    grad_val = gradient(x_data, y_data)
    grad_list.append(grad_val)
    w -= a * grad_val

plt.plot(cost_list)
plt.xlabel('epoch')
plt.ylabel('cost')
plt.show()