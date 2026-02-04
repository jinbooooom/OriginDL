# OriginDL 设计文档

本目录包含 OriginDL 深度学习框架的详细设计文档，涵盖系统架构、设计理念、实现细节等内容。

## 文档目录

### [架构设计文档](architecture.md)

**系统架构与设计文档**，详细介绍 OriginDL 框架的整体架构设计和各模块实现：

- **第1章：架构总览与设计理念** - 框架整体架构、设计原则
- **第2章：Tensor 系统架构** - Tensor 四层架构、内存管理、打印系统
- **第3章：动态计算图构建** - 计算图架构、节点连接、前向传播构建
- **第4章：反向传播实现** - 反向传播算法、梯度累积、拓扑排序
- **第5章：算子系统架构** - Operator 基类、各类算子实现（详细理论请参见 [算子设计理论](operators_theory.md)）
- **第6章：神经网络模块架构** - Module、Layer、Sequential 设计
- **第7章：优化器架构** - Optimizer 基类、SGD、Adam 实现
- **第8章：数据处理架构** - Dataset、DataLoader 设计
- **第9章：IO 模块架构** - Checkpoint、Model IO 设计
- **第10章：PNNX 推理架构** - PNNX 图结构、推理流程
- **第11章：应用示例** - 线性回归、MNIST 训练、ResNet 分类、YOLOv5 推理

### [算子设计理论](operators_theory.md)

**算子前向与反向传播原理详解**，详细说明各个算子的数学原理和实现细节：

- 数学运算算子（Add、Mul、Sub、Div 等）
- 激活函数算子（ReLU、Sigmoid、Softmax 等）
- 卷积运算算子（Conv2d、Conv2dTranspose 等）
- 池化运算算子（MaxPool2d、AvgPool2d 等）
- 形状变换算子（Reshape、Transpose、Flatten 等）
- 神经网络层算子（Linear、BatchNorm 等）
- 归一化算子（BatchNorm、LayerNorm 等）
- 损失函数算子（SoftmaxCrossEntropy 等）

每个算子都包含：
- 前向传播数学原理
- 反向传播梯度推导
- 具体计算示例
- 实现要点说明
