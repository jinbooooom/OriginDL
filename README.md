# OriginDL: 从零开始构建的分布式深度学习框架

## TODO
 
- [x] ~~numCpp~~ ArrayFire

- [x] 自动求导
- [ ] 卷积
- [ ] 支持 GPU
- [ ] 分布式

## build
```shell
source setEnv.sh  # 设置环境变量
bash ./build.sh
# 在 build/libs 下生成 origindl.so
# 在 build/bin/ 下生成所有的测试程序
```

## ArrayFire 配置说明

本项目使用 ArrayFire 库进行张量操作。以下是配置和使用 ArrayFire 的说明。
```shell
wget https://arrayfire.s3.amazonaws.com/3.9.0/ArrayFire-v3.9.0_Linux_x86_64.sh  
sudo sh ArrayFire-v3.9.0_Linux_x86_64.sh  --skip-license --prefix=/opt/arrayfire 
```

### 环境变量设置

项目使用环境变量 `ARRAYFIRE_PATH` 来指定 ArrayFire 的安装路径。您可以通过以下方式设置此环境变量：

#### 方法一：使用 setEnv.sh 脚本

项目提供了 `setEnv.sh` 脚本，默认将 ArrayFire 路径设置为 `/opt/arrayfire`：

```bash
source setEnv.sh
```

#### 方法二：手动设置环境变量

如果您的 ArrayFire 安装在其他位置，可以手动设置环境变量：

```bash
export ARRAYFIRE_PATH=/path/to/your/arrayfire
export LD_LIBRARY_PATH=${ARRAYFIRE_PATH}/lib64:$LD_LIBRARY_PATH
```

### 运行测试

编译成功后，可以运行测试程序：

```bash
./build/bin/dl_variable
./build/bin/dl_function
./build/bin/dl_backward
./build/bin/dl_numericalDiff
```

### ArrayFire 后端选择

ArrayFire 支持多种计算后端（CPU、CUDA、OpenCL）。默认情况下，ArrayFire 会自动选择可用的最佳后端。您可以通过以下方式显式指定后端：

```cpp
// 在代码中设置后端
af::setBackend(AF_BACKEND_CPU);    // 使用 CPU 后端
af::setBackend(AF_BACKEND_CUDA);   // 使用 CUDA 后端
af::setBackend(AF_BACKEND_OPENCL); // 使用 OpenCL 后端
```

或者通过环境变量设置：

```bash
export AF_BACKEND=CPU    # 使用 CPU 后端
export AF_BACKEND=CUDA   # 使用 CUDA 后端
export AF_BACKEND=OPENCL # 使用 OpenCL 后端
```
