# VSCode 配置说明

本项目已配置完整的 VSCode 开发环境，支持 C++ 代码跳转、自动补全、构建和调试。

## 配置文件说明

### c_cpp_properties.json
- 配置 C++ IntelliSense
- 设置包含路径：include/, 3rd/, ArrayFire 路径
- 配置 C++17 标准和编译器参数
- 支持代码跳转和自动补全

### settings.json
- 工作区设置
- CMake 配置
- 文件关联和编码设置
- clangd 配置

### tasks.json
- CMake 配置任务
- 构建任务（默认快捷键：Ctrl+Shift+P -> Tasks: Run Build Task）
- 清理任务
- 构建并运行测试任务

### launch.json
- GDB 调试配置
- 支持调试测试程序
- 配置了库路径和调试参数

### extensions.json
- 推荐的 VSCode 扩展列表
- 包含 C++、CMake 相关扩展

## 使用方法

### 1. 安装推荐扩展
打开 VSCode 后，会提示安装推荐扩展，点击安装即可。

### 2. 构建项目
- 方法1：使用快捷键 `Ctrl+Shift+P`，输入 "Tasks: Run Build Task"
- 方法2：在终端运行 `cmake --build build --parallel $(nproc)`

### 3. 调试程序
- 按 F5 或点击调试按钮
- 选择要调试的程序（如 Debug Test Function）

### 4. 代码跳转
- 按住 Ctrl 点击函数名或变量名即可跳转到定义
- 右键选择 "Go to Definition" 或 "Go to Declaration"
- 使用 F12 快捷键跳转到定义

### 5. 自动补全
- 输入代码时会自动显示补全建议
- 使用 Ctrl+Space 手动触发补全

## 故障排除

如果跳转功能不工作：
1. 确保已安装 C/C++ 扩展
2. 检查 compile_commands.json 是否存在
3. 重启 VSCode
4. 使用 Ctrl+Shift+P -> "C/C++: Reset IntelliSense Database"

## 项目结构
```
.vscode/
├── c_cpp_properties.json  # C++ IntelliSense 配置
├── settings.json          # 工作区设置
├── tasks.json             # 构建任务
├── launch.json            # 调试配置
└── extensions.json        # 推荐扩展
