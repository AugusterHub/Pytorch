#!/usr/bin/python
# -*- coding: UTF-8 -*-
import torch

x_data = torch.Tensor([[1.0], [2.0], [3.0]])
y_data = torch.Tensor([[2.0], [4.0], [6.0]])

class LinearModel(torch.nn.Module):
    """
    1. 自定义一个LinearModel类，为方便它继承自torch里面的 Module；
    2. 该类应该有两个基本方法 (必须叫这两个名字) ：
    def __init__(self) 模型初始化相关
    def forward(self, ) 前向传播
    """
    def __init__(self):
        """
        初始化相关
        """
        super(LinearModel, self).__init__() # 继承自父类的构造（__init__）,传入的参数为自定义的类名（LinearModel）和self
        self.linear = torch.nn.Linear(1, 1) # liner继承自torch里面的Linear(1, 1) 参数Linear(输入维度size, 输出维度size)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

model = LinearModel()

""" 构造损失函数和优化器 """
criterion = torch.nn.MSELoss(size_average=False)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print('w=', model.linear.weight.item())
print('b=', model.linear.bias.item())

x_test = torch.Tensor([[4.0]])
y_test = model(x_test)
print('y_pred', y_test.data)
