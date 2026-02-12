# 数据和模型发布指南

本文档说明如何在 GitHub Releases 上发布模型文件（不包含源代码）。  
**数据说明**：项目数据目前仅 MNIST，由 `scripts/download_mnist.sh` 从镜像下载，不通过本 Release 发布。

## 📋 准备工作

### 1. 打包模型

在项目根目录使用打包脚本（推荐）或直接使用 tar：

```bash
# 使用打包脚本（推荐）：-d 输入目录，-o 输出路径，--exclude 排除项可多次指定
bash scripts/pack_release.sh -d model -o origindl-model-v1.0.0.tar.gz --exclude '.gitkeep'

# 预览不执行：加 -n
bash scripts/pack_release.sh -d model -o origindl-model-v1.0.0.tar.gz --exclude '.gitkeep' -n

# 或直接使用 tar
tar -czf origindl-model-v1.0.0.tar.gz --exclude='model/.gitkeep' -C . model
```

### 2. 验证压缩包内容

```bash
# 查看压缩包内容（不解压）
tar -tzf origindl-model-v1.0.0.tar.gz | head -20
```

## 🚀 创建 GitHub Release

### 方式一：通过 GitHub Web 界面（推荐）

1. **创建 Tag**
   - 访问仓库页面，点击 "Releases" → "Create a new release"
   - 或者直接访问：`https://github.com/jinbooooom/OriginDL/releases/new`
   - **Tag 名称**：输入 `v1.0.0`（与 `download_model.sh` 中的 VERSION 一致）
   - **Target**：选择当前的某个 commit（可以是 `main` 或 `master` 分支的最新 commit）
   - **Release title**：`Model v1.0.0`

2. **填写 Release 说明**
   ```markdown
   # OriginDL 模型发布 v1.0.0
   
   ## 📦 内容说明
   
   本 Release 仅包含运行 OriginDL 示例程序所需的**预训练模型文件**，不包含源代码与数据集。
   
   - 源代码请通过 `git clone` 获取
   - 数据（MNIST）请使用 `bash scripts/download_mnist.sh` 下载
   
   ## 📥 下载说明
   
   ### 方式一：使用下载脚本（推荐）
   ```bash
   bash scripts/download_model.sh
   ```
   默认保存到 `./model`，可使用 `-d DIR` 指定目录，详见 `scripts/download_model.sh -h`。
   
   ### 方式二：手动下载
   1. 下载 `origindl-model-v1.0.0.tar.gz`
   2. 在项目根目录解压到 model 目录：
      ```bash
      tar -xzf origindl-model-v1.0.0.tar.gz
      ```
   
   ## 📂 文件说明
   
   - `origindl-model-v1.0.0.tar.gz` - 预训练模型文件
   
   ## ⚠️ 注意事项
   
   - 确保有足够的磁盘空间
   - 源代码与数据请按上文说明另行获取
   ```

3. **上传附件**
   - 在 "Attach binaries" 区域，拖拽或选择：`origindl-model-v1.0.0.tar.gz`
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

# 创建 Release（只上传模型压缩包）
gh release create v1.0.0 \
    --title "Model v1.0.0" \
    --notes "本 Release 仅包含预训练模型文件，不包含源代码与数据。数据请使用 scripts/download_mnist.sh 下载。" \
    origindl-model-v1.0.0.tar.gz
```

## ✅ 验证发布

1. **检查 Release 页面**
   - 访问：`https://github.com/jinbooooom/OriginDL/releases/tag/v1.0.0`
   - 确认模型压缩包已上传且文件大小正确

2. **测试下载脚本**
   ```bash
   # 在另一个目录测试
   git clone https://github.com/jinbooooom/OriginDL.git test-clone
   cd test-clone
   
   # 如需修改仓库或版本，编辑 scripts/download_model.sh 中的 REPO_OWNER、REPO_NAME、VERSION
   bash scripts/download_model.sh
   ```

## 🔄 更新 Release

如果需要更新模型：

1. **创建新版本**
   - 使用新的 tag，如 `v1.0.1`（与 `download_model.sh` 中 VERSION 一致）
   - 重新打包模型目录
   - 创建新的 Release

2. **更新下载脚本**
   - 修改 `scripts/download_model.sh` 中的 `VERSION` 变量

## 📝 注意事项

- ✅ **Tag 与脚本一致**：Release tag（如 `v1.0.0`）需与 `scripts/download_model.sh` 中的 VERSION 一致
- ✅ **明确说明**：在 Release 说明中标注“不包含源代码与数据”，并说明数据用 `download_mnist.sh` 获取
- ✅ **文件命名**：使用清晰的命名，如 `origindl-model-v1.0.0.tar.gz`
- ✅ **验证下载**：发布后测试 `download_model.sh` 是否正常
- ❌ **不要上传源代码**：只上传模型压缩包

## 🔗 相关链接

- [GitHub Releases 文档](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
- [数据下载说明](../../data/README.md)
- [模型下载说明](../../model/README.md)
