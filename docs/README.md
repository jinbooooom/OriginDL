# OriginDL 文档中心

欢迎来到 OriginDL 深度学习框架的文档中心！本文档提供了框架的完整文档导航，帮助您快速找到所需的信息。

## 📖 文档结构

```
docs/
├── README.md                    # 本文档（文档导航）
├── design/                      # 设计文档目录
│   ├── README.md               # 设计文档目录介绍
│   ├── architecture.md         # 系统架构设计文档
│   └── operators_theory.md     # 算子设计理论文档
├── user_guide/                 # 用户指南目录
│   ├── api.md                  # API 参考文档
│   └── compare.md              # 与 PyTorch 对比文档
└── contributing/               # 贡献指南目录
    └── DATA_RELEASE_GUIDE.md   # 数据和模型发布指南
```

## 📚 文档目录

### [设计文档](design/)

**系统架构与设计文档**，深入介绍 OriginDL 框架的架构设计、实现原理和设计理念。

- **[架构设计文档](design/architecture.md)** - 完整的系统架构设计，包含11个章节：
  - 架构总览与设计理念
  - Tensor 系统架构（四层架构设计）
  - 动态计算图构建
  - 反向传播实现
  - 算子系统架构
  - 神经网络模块架构
  - 优化器架构
  - 数据处理架构
  - IO 模块架构
  - PNNX 推理架构
  - 应用示例

- **[算子设计理论](design/operators_theory.md)** - 详细的算子数学原理和实现细节：
  - 数学运算算子（Add、Mul、Sub、Div 等）
  - 激活函数算子（ReLU、Sigmoid、Softmax 等）
  - 卷积运算算子（Conv2d、Conv2dTranspose 等）
  - 池化运算算子（MaxPool2d、AvgPool2d 等）
  - 形状变换算子（Reshape、Transpose、Flatten 等）
  - 神经网络层算子（Linear、BatchNorm 等）
  - 归一化算子（BatchNorm、LayerNorm 等）
  - 损失函数算子（SoftmaxCrossEntropy 等）

### [用户指南](user_guide/)

**API 文档和使用指南**，帮助用户快速上手和使用 OriginDL 框架。

- **[API 文档](user_guide/api.md)** - 完整的 API 参考文档：
  - 张量创建与操作
  - 数学运算函数
  - 神经网络模块
  - 优化器使用
  - 数据处理
  - CUDA 支持
  - 每个 API 都包含详细的参数说明、返回值说明和使用示例

- **[与 PyTorch 对比](user_guide/compare.md)** - OriginDL 与 PyTorch 的 API 对比：
  - 帮助从 PyTorch 迁移到 OriginDL
  - 详细的 API 对比表格
  - 语法差异说明
  - 使用示例对比

### [贡献指南](contributing/)

**项目维护和贡献指南**，帮助维护者和贡献者了解项目的维护流程。

- **[数据和模型发布指南](contributing/DATA_RELEASE_GUIDE.md)** - 如何在 GitHub Releases 上发布数据和模型文件：
  - 数据打包和验证
  - GitHub Release 创建流程
  - 下载脚本配置
  - 版本管理最佳实践

## 🔗 相关资源

- **项目主页**：项目根目录的 [README.md](../README.md)
- **代码仓库**：查看源码了解具体实现
- **测试示例**：`tests/example/` 目录下的应用示例
- **单元测试**：`tests/unit_test/` 目录下的测试用例

## 📝 文档维护

本文档随框架开发持续更新。如果您发现文档有误或需要补充，欢迎：
- 提交 Issue 反馈问题
- 提交 Pull Request 改进文档
- 参与文档编写和维护

---

**提示**：建议使用支持 Markdown 的编辑器或文档查看器阅读本文档，以获得最佳的阅读体验。
