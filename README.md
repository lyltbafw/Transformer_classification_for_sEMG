# 多通道肌电信号处理与Transformer模型

## 项目介绍
本项目旨在利用Transformer架构对多通道肌电信号（EMG）进行分析和分类。通过Transformer模型的自注意力机制，我们能够捕捉到肌电信号中的复杂模式和特征，从而实现高效的信号分类和识别。此外，我们使用了NAO机器人，来实现对肌电信号动作的呈现

## 目录结构
- `Datasets/`: 存放原始肌电数据集。
- `models/`: 包含训练所得预训练模型。
- `train.py`: 训练transformer，内部可以修改数据的输入方式，模型的定义以及训练参数
- `predict.py`: 对测试集数据进行测试，并将测试结果通过SSH连接的方式控制NAO机器人动作

## 环境配置
Pytorch(GPU版本)、Paramiko库
