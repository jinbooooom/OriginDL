# GoogleTest 精简版

这是基于原始GoogleTest的精简版本，专门为Origindl项目使用，删除了不必要的文件和目录。

## 原始GoogleTest vs 精简版对比

### 已删除的文件和目录

#### 1. 根目录文件
- `BUILD.bazel` - Bazel构建文件
- `CONTRIBUTING.md` - 贡献指南
- `CONTRIBUTORS` - 贡献者列表
- `fake_fuchsia_sdk.bzl` - Fuchsia SDK相关文件
- `googletest_deps.bzl` - Bazel依赖文件
- `LICENSE` - 许可证文件
- `MODULE.bazel` - Bazel模块文件
- `README.md` - 原始说明文档
- `WORKSPACE` - Bazel工作空间文件
- `WORKSPACE.bzlmod` - Bazel模块工作空间文件

#### 2. 目录
- `ci/` - 持续集成相关文件
  - `linux-presubmit.sh` - Linux预提交脚本
  - `macos-presubmit.sh` - macOS预提交脚本
  - `windows-presubmit.bat` - Windows预提交脚本
- `docs/` - 文档目录
  - 各种Markdown文档文件（用户指南、FAQ、教程等）
  - 网站相关文件（Jekyll配置、布局等）
- `googlemock/` - GoogleMock模拟对象框架
  - 完整的GoogleMock源码和头文件
  - GoogleMock测试用例
  - GoogleMock示例程序

#### 3. GoogleTest子目录
- `googletest/docs/` - GoogleTest文档
- `googletest/samples/` - GoogleTest示例程序
- `googletest/test/` - GoogleTest自身测试用例

### 保留的核心文件

#### 1. 根目录
- `CMakeLists.txt` - 主CMake配置文件（已禁用GoogleMock）

#### 2. GoogleTest核心
- `googletest/CMakeLists.txt` - GoogleTest CMake配置
- `googletest/include/` - GoogleTest头文件
- `googletest/src/` - GoogleTest源码文件
- `googletest/cmake/` - CMake辅助文件
- `googletest/README.md` - GoogleTest说明文档

## 配置变更

### CMake配置修改
1. **禁用GoogleMock**：`BUILD_GMOCK` 设置为 `OFF`
2. **禁用示例程序**：`gtest_build_samples` 默认为 `OFF`
3. **禁用自身测试**：`gtest_build_tests` 默认为 `OFF`

### 编译选项
- 继承主项目的C++标准（C++17）
- 继承主项目的编译选项
- 库文件输出到 `build/lib/` 目录

## 使用方式

### 编译
```bash
cd /path/to/your/project
bash build.sh
```

### 运行测试
```bash
./build/bin/3rd_googletest
```

## 功能特性

### 保留的功能
- ✅ 基本断言宏（ASSERT_*, EXPECT_*）
- ✅ 测试用例组织（TEST, TEST_F, TEST_P）
- ✅ 参数化测试
- ✅ 测试固件（Test Fixtures）
- ✅ 死亡测试（Death Tests）
- ✅ 测试运行器和报告

### 移除的功能
- ❌ GoogleMock模拟对象框架
- ❌ 示例程序
- ❌ 自身测试用例
- ❌ 文档和教程
- ❌ 持续集成脚本

## 文件大小对比

### 原始GoogleTest
- 包含所有文件：~50MB+
- 编译时间：较长
- 库文件：包含GoogleMock

### 精简版GoogleTest
- 核心文件：~5MB
- 编译时间：显著减少
- 库文件：仅GoogleTest核心

## 注意事项

1. **仅支持基本单元测试**：此版本专注于基本的单元测试功能
2. **不支持模拟对象**：如需模拟对象功能，请使用完整版GoogleTest
3. **自定义配置**：已针对Origindl项目进行优化配置
4. **维护性**：精简版本更容易维护和更新

---

**注意**：此精简版本仅包含GoogleTest的核心测试功能，如需完整功能请使用原始GoogleTest。
