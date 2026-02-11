# 模型文件目录

本目录用于存放训练好的模型文件和预训练权重。

## 📦 模型下载

由于模型文件较大，未包含在 Git 仓库中。请通过以下方式获取：

### 方式一：使用下载脚本（推荐）

```bash
bash scripts/download_data.sh
```

### 方式二：手动下载

1. 访问 [GitHub Releases](https://github.com/jinbooooom/origindl/releases)
2. 下载最新版本的 `origindl-model-v1.0.0.tar.gz`
3. 解压到项目根目录：
   ```bash
   tar -xzf origindl-model-v1.0.0.tar.gz
   ```

## 📂 目录结构

```
model/
└── pnnx/          # PNNX 格式的模型文件
```
