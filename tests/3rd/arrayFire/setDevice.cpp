#include <arrayfire.h>
#include <getopt.h>
#include <iostream>

void print_help()
{
    std::cout << "Usage: ./backend_demo -b [backend_code]\n"
              << "Available backends:\n"
              << "0 : CPU\n"
              << "1 : CUDA\n"
              << "2 : OpenCL\n";
}

int main(int argc, char **argv)
{
    int backend_code = 0;  // 默认CPU
    int c;

    // 解析命令行参数
    while ((c = getopt(argc, argv, "b:h")) != -1)
    {
        switch (c)
        {
            case 'b':
                backend_code = atoi(optarg);
                break;
            case 'h':
                print_help();
                return 0;
            default:
                return 1;
        }
    }

    // 设置计算后端
    try
    {
        switch (backend_code)
        {
            case 1:
                af::setBackend(AF_BACKEND_CUDA);  // [4]() CUDA后端
                break;
            case 2:
                af::setBackend(AF_BACKEND_OPENCL);  // OpenCL后端
                break;
            default:
                af::setBackend(AF_BACKEND_CPU);  // CPU后端
        }
    }
    catch (const af::exception &e)
    {
        std::cerr << "Failed to set backend: " << e.what() << std::endl;
        return 1;
    }

    // 验证后端设置
    af::info();  // 输出设备信息
    std::cout << "\nActive Backend: ";
    switch (af::getActiveBackend())
    {  // 获取当前后端
        case AF_BACKEND_CUDA:
            std::cout << "CUDA";
            break;
        case AF_BACKEND_OPENCL:
            std::cout << "OpenCL";
            break;
        case AF_BACKEND_CPU:
            std::cout << "CPU";
            break;
    }
    std::cout << std::endl;

    // 创建数据并计算
    try
    {
        af::array a      = af::randu(3, 3);  // 自动使用当前后端
        af::array b      = af::constant(2, 3, 3);
        af::array result = a + b;  // 后端计算验证

        af::print("a + b = ", result);
    }
    catch (const af::exception &e)
    {
        std::cerr << "Computation failed: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

/*

直接链接 af, 让 ArrayFire 自动选择可用计算后端，可通过 af::setBackend() 显式指定(推荐做法)
$ g++ setDevice.cpp  -o ./bin/setDevice -I/opt/arrayfire/include -L/opt/arrayfire/lib64  -laf
$ ./bin/setDevice -b 0
ERROR: GLFW wasn't able to initalize
ArrayFire v3.9.0 (CPU, 64-bit Linux, build b59a1ae53)
[0] Intel: Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz
Active Backend: CPU
a + b =
[3 3 1 1]
    2.6010     2.2126     2.2864
    2.0278     2.0655     2.3410
    2.9806     2.5497     2.7509

$ ./bin/setDevice -b 1
ERROR: GLFW wasn't able to initalize
ArrayFire v3.9.0 (CUDA, 64-bit Linux, build b59a1ae53)
Platform: CUDA Runtime 12.2, Driver: for
[0] NVIDIA GeForce RTX 4090, 24112 MB, CUDA Compute 8.9
-1- NVIDIA A100-PCIE-40GB, 40445 MB, CUDA Compute 8.0

Active Backend: CUDA
a + b =
[3 3 1 1]
    2.6010     2.2126     2.2864
    2.0278     2.0655     2.3410
    2.9806     2.5497     2.7509

$ ./bin/setDevice -b 2
ERROR: GLFW wasn't able to initalize
ArrayFire v3.9.0 (OpenCL, 64-bit Linux, build b59a1ae53)
[0] NVIDIA: NVIDIA A100-PCIE-40GB, 40444 MB
-1- NVIDIA: NVIDIA GeForce RTX 4090, 24111 MB

Active Backend: OpenCL
a + b =
[3 3 1 1]
    2.6010     2.2126     2.2864
    2.0278     2.0655     2.3410
    2.9806     2.5497     2.7509


下面三个是链接指定具体的后端的 so (不推荐做法)
$ g++ setDevice.cpp  -o ./bin/setDevice -I/opt/arrayfire/include -L/opt/arrayfire/lib64 -lafcuda
$ ./bin/setDevice -b 1
ArrayFire v3.9.0 (CUDA, 64-bit Linux, build b59a1ae53)
Platform: CUDA Runtime 12.2, Driver: for
[0] NVIDIA GeForce RTX 4090, 24112 MB, CUDA Compute 8.9
-1- NVIDIA A100-PCIE-40GB, 40445 MB, CUDA Compute 8.0

Active Backend: CUDA
a + b =
[3 3 1 1]
    2.6010     2.2126     2.2864
    2.0278     2.0655     2.3410
    2.9806     2.5497     2.7509

$ g++ setDevice.cpp  -o ./bin/setDevice -I/opt/arrayfire/include -L/opt/arrayfire/lib64 -lafcpu
$ ./bin/setDevice -b 0
ArrayFire v3.9.0 (CPU, 64-bit Linux, build b59a1ae53)
[0] Intel: Intel(R) Xeon(R) Gold 6226R CPU @ 2.90GHz
Active Backend: CPU
a + b =
[3 3 1 1]
    2.6010     2.2126     2.2864
    2.0278     2.0655     2.3410
    2.9806     2.5497     2.7509

$ g++ setDevice.cpp  -o ./bin/setDevice -I/opt/arrayfire/include -L/opt/arrayfire/lib64  -lafopencl
$ ./bin/setDevice -b 2
ERROR: GLFW wasn't able to initalize
ArrayFire v3.9.0 (OpenCL, 64-bit Linux, build b59a1ae53)
[0] NVIDIA: NVIDIA A100-PCIE-40GB, 40444 MB
-1- NVIDIA: NVIDIA GeForce RTX 4090, 24111 MB

Active Backend: OpenCL
a + b =
[3 3 1 1]
    2.6010     2.2126     2.2864
    2.0278     2.0655     2.3410
    2.9806     2.5497     2.7509
*/