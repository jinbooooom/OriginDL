# 数据目录

本目录用于存放项目所需数据集。

## MNIST 数据集

训练与测试 MNIST 手写数字识别时，需先下载 MNIST 数据集。

### 使用脚本下载

在**项目根目录**下执行：

```bash
bash scripts/download_mnist.sh
```

默认会将数据下载并解压到 `data/mnist/` 目录。

### 指定下载目录

若希望下载到其他目录，可使用 `-d` 或 `--dir` 指定：

```bash
bash scripts/download_mnist.sh -d ./data/mnist
```
