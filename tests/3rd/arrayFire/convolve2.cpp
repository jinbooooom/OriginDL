#include <arrayfire.h>

int main()
{
    try
    {
        af::setDevice(0);  // 选择GPU设备
        af::info();        // 显示设备信息

        // 创建输入数据（32x32随机矩阵）
        af::array input = af::randu(32, 32);

        // 定义3x3卷积核（边缘检测）
        af::array kernel = af::constant(0, 3, 3);
        kernel(1, 1)     = 4;             // 中心权重
        kernel(af::span, af::span) -= 1;  // 周围-1

        // 执行二维卷积
        af::array output = af::convolve2(input, kernel);

        // 打印结果
        af::print("Input", input(af::seq(0, 4), af::seq(0, 4)));
        af::print("Output", output(af::seq(0, 4), af::seq(0, 4)));
    }
    catch (af::exception &e)
    {
        fprintf(stderr, "Error: %s\n", e.what());
    }
    return 0;
}

/*
$ g++ convolve2.cpp  -o ./bin/convolve2 -I/opt/arrayfire/include -L/opt/arrayfire/lib64 -lafcuda
$ ./bin/convolve2
ArrayFire v3.9.0 (CUDA, 64-bit Linux, build b59a1ae53)
Platform: CUDA Runtime 12.2, Driver: for
[0] NVIDIA GeForce RTX 4090, 24112 MB, CUDA Compute 8.9
-1- NVIDIA A100-PCIE-40GB, 40445 MB, CUDA Compute 8.0
Input
[5 5 1 1]
    0.6010     0.3336     0.8578     0.9033     0.1330
    0.0278     0.0363     0.0192     0.5131     0.4696
    0.9806     0.5349     0.7191     0.5784     0.3014
    0.2126     0.0123     0.4035     0.6910     0.6938
    0.0655     0.3988     0.4692     0.4792     0.3513

Output
[5 5 1 1]
    1.4052    -0.5413     0.7680     0.7172    -2.9830
   -2.4030    -3.9651    -4.4190    -2.4426    -2.5243
    2.1178    -0.8065    -0.6313    -2.0755    -2.9865
   -1.3541    -3.7474    -2.6726    -1.9230    -1.4520
   -1.9557    -1.8302    -2.5175    -2.8507    -3.8617
*/
