#include <chrono>
#include <iostream>
#include "origin.h"

using namespace origin;
namespace F = origin::functional;

int main()
{
    // 检查CUDA是否可用
    cuda::device_info();

    if (!cuda::is_available())
    {
        std::cout << "CUDA is not available on this system!" << std::endl;
        return 1;
    }

    {

        Tensor a({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}, dtype(Float32).device(kCUDA));
        Tensor b({10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f}, Shape{2, 3}, dtype(Float32).device(kCUDA));

        a.print("A");

        b.print("B");

        auto c = a + b;
        c.print("A + B");
    }

    // 大张量性能测试
    {
        std::cout << "Large tensor performance test: shape {1000, 1000}" << std::endl;

        // 创建较大的张量
        Shape shape = {1000, 1000};
        std::vector<float> data_a(shape.elements());
        std::vector<float> data_b(shape.elements());

        for (size_t i = 0; i < shape.elements(); i++)
        {
            data_a[i] = static_cast<float>(i);
            data_b[i] = static_cast<float>(i * 0.1f);
        }

        Tensor a(data_a, shape, dtype(Float32).device(kCUDA));
        Tensor b(data_b, shape, dtype(Float32).device(kCUDA));

        // 计时CUDA加法
        auto start = std::chrono::high_resolution_clock::now();
        auto c     = a + b;
        auto end   = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "CUDA addition time: " << duration.count() << " microseconds" << std::endl;
    }

    return 0;
}

/*
CUDA devices available: 2
Device 0: NVIDIA A100 80GB PCIe
  Compute capability: 8.0
  Memory: 81152 MB
  Multiprocessors: 108
  Max threads per block: 1024
Device 1: NVIDIA A100 80GB PCIe
  Compute capability: 8.0
  Memory: 81152 MB
  Multiprocessors: 108
  Max threads per block: 1024
A:
[[1, 2, 3],
 [4, 5, 6]]
 OriginMat(shape=[2, 3], dtype=float32, device=cuda:0)
B:
[[10, 20, 30],
 [40, 50, 60]]
 OriginMat(shape=[2, 3], dtype=float32, device=cuda:0)
A + B:
[[11, 22, 33],
 [44, 55, 66]]
 OriginMat(shape=[2, 3], dtype=float32, device=cuda:0)
Large tensor performance test: shape {1000, 1000}
CUDA addition time: 236 microseconds
*/
