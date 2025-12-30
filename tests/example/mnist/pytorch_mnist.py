#!/usr/bin/env python3
"""
PyTorch MNIST Training Example
用于对比 OriginDL 和 PyTorch 的实现差异
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

# 设置随机种子
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# 超参数
max_epoch = 10
batch_size = 256
learning_rate = 0.0005
weight_decay = 1e-4
log_interval = 50

# 设备选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"=== MNIST Handwritten Digit Recognition with CNN (PyTorch) ===")
print(f"Device: {device}")
print(f"Max epochs: {max_epoch}")
print(f"Batch size: {batch_size}")
print(f"Learning rate: {learning_rate}")
print(f"Weight decay: {weight_decay}")
print(f"Log interval: {log_interval}")

# 数据加载
print("Loading MNIST dataset...")
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 均值和标准差
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0  # Windows 兼容性
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0
)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# 定义模型
class SimpleCNN(nn.Module):
    """
    简单的CNN模型
    结构：Conv2d -> ReLU -> MaxPool2d -> Conv2d -> ReLU -> MaxPool2d -> Flatten -> Linear -> ReLU -> Linear
    """
    def __init__(self):
        super(SimpleCNN, self).__init__()
        # 第一层卷积：1通道 -> 64通道，3x3卷积核
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=True)
        # 第二层卷积：64通道 -> 128通道，3x3卷积核
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=True)
        # 全连接层1：7*7*128 -> 128
        self.fc1 = nn.Linear(7 * 7 * 128, 128, bias=True)
        # 全连接层2：128 -> 10
        self.fc2 = nn.Linear(128, 10, bias=True)
        # 激活函数和池化
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        # 输入形状: (N, 1, 28, 28)
        # 第一层：Conv2d(1, 64, 3x3, pad=1) -> ReLU -> MaxPool2d(2x2)
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 形状: (N, 64, 14, 14)
        
        # 第二层：Conv2d(64, 128, 3x3, pad=1) -> ReLU -> MaxPool2d(2x2)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # 形状: (N, 128, 7, 7)
        
        # Flatten: (N, 128, 7, 7) -> (N, 128*7*7) = (N, 6272)
        x = self.flatten(x)
        
        # 全连接层1：6272 -> 128
        x = self.fc1(x)
        x = self.relu(x)
        # 形状: (N, 128)
        
        # 全连接层2：128 -> 10
        x = self.fc2(x)
        # 形状: (N, 10)
        
        return x

# 创建模型
print("Creating CNN model...")
model = SimpleCNN().to(device)
print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# 创建优化器
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# 损失函数
criterion = nn.CrossEntropyLoss()

# 训练循环
print("Starting training...")
for epoch in range(max_epoch):
    print(f"========== Epoch {epoch + 1}/{max_epoch} ==========")
    
    # 训练阶段
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 前向传播
        output = model(data)
        loss = criterion(output, target)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 统计
        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        train_correct += pred.eq(target.view_as(pred)).sum().item()
        train_total += target.size(0)
        
        # 定期打印
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = train_loss / (batch_idx + 1)
            avg_acc = 100.0 * train_correct / train_total
            print(f"Epoch {epoch + 1}/{max_epoch} Batch {batch_idx + 1} Loss: {avg_loss:.4f} Acc: {avg_acc:.2f}%")
    
    avg_train_loss = train_loss / len(train_loader)
    train_acc = 100.0 * train_correct / train_total
    print(f"Epoch {epoch + 1}/{max_epoch} Training Complete - Loss: {avg_train_loss:.4f} Acc: {train_acc:.2f}%")
    
    # 测试阶段
    print("Evaluating on test set...")
    model.eval()
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            loss = criterion(output, target)
            
            # 统计
            test_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            test_correct += pred.eq(target.view_as(pred)).sum().item()
            test_total += target.size(0)
            
            # 定期打印
            if (test_total // batch_size) % log_interval == 0 and test_total % batch_size == 0:
                avg_test_loss_so_far = test_loss / (test_total // batch_size)
                avg_test_acc_so_far = 100.0 * test_correct / test_total
                print(f"Test Batch {test_total // batch_size} Loss: {avg_test_loss_so_far:.4f} Acc: {avg_test_acc_so_far:.2f}%")
    
    avg_test_loss = test_loss / len(test_loader)
    test_acc = 100.0 * test_correct / test_total
    
    # 输出epoch结果
    print(f"========== Epoch {epoch + 1}/{max_epoch} Summary ==========")
    print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
    print(f"  Test Loss:  {avg_test_loss:.4f}, Test Acc:  {test_acc:.2f}%")
    print("===========================================")

print("Training completed!")

