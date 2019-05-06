## 深度学习

### NVIDIA 显卡驱动

法一: 添加新 nvidia 官方驱动源

```
sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update
```

安装驱动(对应我的笔记本 GTX1060 显卡，驱动是 410)

```
sudo apt-get install nvidia-410 nvidia-settings
```

法二： 打开 **设置>>软件和更新>>附加驱动** 选中驱动,**应用更改** 如果失败，勾选 **设置>>软件和更新>>更新>>推荐更新**

### CUDA

```
sudo sh cuda_9*.run
```

安装过程中，会问你几个问题。只要在安装 Nvidia 显卡驱动时选择 no 就好了。(Install NVIDIA Accelerated Graphics Driver for Linux-x86_64 361.62? n)其它的都可以选择 yes. 完成后，执行:

```
sudo gedit ~/.bashrc
```

在末尾添加:

```
export PATH=/usr/local/cuda-9.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-9.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
```

保存后：

```
source ~/.bashrc
```

测试是否安装成功：

```
cd /usr/local/cuda-9.0/samples/1_Utilities/deviceQuery
sudo make
sudo ./deviceQuery
```

### 

### cuDNN

```
tar -xzvf cudnn-9.0-linux-*.tgz
```

解压后的文件夹名为 cuda，文件夹中包含两个文件夹：一个为 include，另一个为 lib64。最好把这个 cuda 文件夹放在 home 目录下，方便操作。 将解压后的 lib64 文件夹关联到环境变量中。这一步很重要。

```
sudo gedit ~/.bashrc
```

在最后一行添加：

```
export LD_LIBRARY_PATH=/home/cuda/lib64:$LD_LIBRARY_PATH
```

紧接着：

```
source ~/.bashrc
```

配置cuDNN的最后一步就是将解压后的/home/cuda/include目录下的一些文件拷贝到/usr/local/cuda/include中。由于进入了系统路径，因此执行该操作时需要获取管理员权限。

```
sudo cp cuda/include/cudnn.h /usr/local/cuda/include
sudo cp cuda/lib64/libcudnn* /usr/local/cuda/lib64
sudo chmod a+r /usr/local/cuda/include/cudnn.h
sudo chmod a+r /usr/local/cuda/lib64/libcudnn*
```

查看 cuDNN 版本：

```
cat /usr/local/cuda/include/cudnn.h | grep CUDNN_MAJOR -A 2
```

### 

### Anaconda

```
bash Anaconda*.sh
```

## 
