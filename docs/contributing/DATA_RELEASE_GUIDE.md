# 数据和模型发布指南

本文档说明如何在 GitHub Releases 上发布数据和模型文件（不包含源代码）。

## 📋 准备工作

### 1. 打包数据和模型

在项目根目录执行以下命令：

```bash
# 打包数据目录（排除不需要的文件）
tar -czf origindl-data-v1.0.0.tar.gz \
    --exclude='data/outputs/*' \
    --exclude='data/.gitkeep' \
    data/

# 打包模型目录
tar -czf origindl-model-v1.0.0.tar.gz \
    --exclude='model/.gitkeep' \
    model/

# 验证压缩包大小
ls -lh origindl-data-v1.0.0.tar.gz origindl-model-v1.0.0.tar.gz
```

### 2. 验证压缩包内容

```bash
# 查看压缩包内容（不解压）
tar -tzf origindl-data-v1.0.0.tar.gz | head -20
tar -tzf origindl-model-v1.0.0.tar.gz | head -20
```

## 🚀 创建 GitHub Release

### 方式一：通过 GitHub Web 界面（推荐）

1. **创建 Tag**
   - 访问仓库页面，点击 "Releases" → "Create a new release"
   - 或者直接访问：`https://github.com/jinbooooom/origindl/releases/new`
   - **Tag 名称**：输入 `data-v1.0.0`（使用独立的 tag，不与源代码版本混淆）
   - **Target**：选择当前的某个 commit（可以是 `main` 或 `master` 分支的最新 commit）
   - **Release title**：`Data and Model v1.0.0`

2. **填写 Release 说明**
   ```markdown
   # OriginDL 数据和模型发布 v1.0.0
   
   ## 📦 内容说明
   
   本 Release 仅包含运行 OriginDL 示例程序所需的数据集和模型文件，**不包含源代码**。
   
   源代码仍在开发中，请通过 `git clone` 获取最新代码。
   
   ## 📥 下载说明
   
   ### 方式一：使用下载脚本（推荐）
   ```bash
   bash scripts/download_data.sh
   ```
   
   ### 方式二：手动下载
   1. 下载 `origindl-data-v1.0.0.tar.gz` 和 `origindl-model-v1.0.0.tar.gz`
   2. 解压到项目根目录：
      ```bash
      tar -xzf origindl-data-v1.0.0.tar.gz
      tar -xzf origindl-model-v1.0.0.tar.gz
      ```
   
   ## 📂 文件说明
   
   - `origindl-data-v1.0.0.tar.gz` - 数据集文件（包含 MNIST 等）
   - `origindl-model-v1.0.0.tar.gz` - 预训练模型文件
   
   ## ⚠️ 注意事项
   
   - 数据文件较大，下载可能需要较长时间
   - 确保有足够的磁盘空间
   - 源代码请通过 `git clone` 获取
   ```

3. **上传附件**
   - 在 "Attach binaries" 区域，拖拽或选择以下文件：
     - `origindl-data-v1.0.0.tar.gz`
     - `origindl-model-v1.0.0.tar.gz`
   - **不要上传源代码相关的文件**

4. **发布**
   - 选择 "Set as the latest release"（如果需要）
   - 点击 "Publish release"

### 方式二：使用 GitHub CLI

```bash
# 安装 GitHub CLI（如果未安装）
# Ubuntu/Debian: sudo apt install gh
# macOS: brew install gh

# 登录 GitHub
gh auth login

# 创建 Release（只上传附件，不关联源代码）
gh release create data-v1.0.0 \
    --title "Data and Model v1.0.0" \
    --notes "本 Release 仅包含数据集和模型文件，不包含源代码。源代码请通过 git clone 获取。" \
    origindl-data-v1.0.0.tar.gz \
    origindl-model-v1.0.0.tar.gz
```

## ✅ 验证发布

1. **检查 Release 页面**
   - 访问：`https://github.com/jinbooooom/origindl/releases/tag/data-v1.0.0`
   - 确认两个压缩包都已上传
   - 确认文件大小正确

2. **测试下载脚本**
   ```bash
   # 在另一个目录测试
   git clone https://github.com/jinbooooom/origindl.git test-clone
   cd test-clone
   
   # 编辑脚本中的 REPO_OWNER 和 REPO_NAME
   # 然后运行
   bash scripts/download_data.sh
   ```

## 🔄 更新 Release

如果需要更新数据或模型：

1. **创建新版本**
   - 使用新的 tag，如 `data-v1.0.1`
   - 重新打包文件
   - 创建新的 Release

2. **更新下载脚本**
   - 修改 `scripts/download_data.sh` 中的 `VERSION` 变量

## 📝 注意事项

- ✅ **使用独立的 tag**：使用 `data-v1.0.0` 而不是 `v1.0.0`，避免与源代码版本混淆
- ✅ **明确说明**：在 Release 说明中明确标注"不包含源代码"
- ✅ **文件命名**：使用清晰的命名，如 `origindl-data-v1.0.0.tar.gz`
- ✅ **验证下载**：发布后测试下载链接是否正常
- ❌ **不要上传源代码**：只上传数据和模型的压缩包

## 🔗 相关链接

- [GitHub Releases 文档](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
- [数据下载说明](../../data/README.md)
- [模型下载说明](../../model/README.md)
