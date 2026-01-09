#include <chrono>
#include <iostream>
#include "origin.h"
namespace F = origin::functional;

/**
 * @brief 扩展的CUDA张量测试
 * @details 测试新实现的CUDA算子
 */
int main()
{
    // 检查CUDA是否可用
#ifdef WITH_CUDA
    origin::cuda::print_cuda_device_info();

    if (!origin::cuda::is_cuda_available())
    {
        std::cout << "CUDA is not available on this system!" << std::endl;
        return 1;
    }

    std::cout << "Testing extended CUDA operators..." << std::endl;

    // 测试基础二元运算
    {
        std::cout << "\n=== Testing Basic Binary Operations ===" << std::endl;

        origin::Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, origin::Shape{2, 2},
                         origin::dtype(origin::Float32).device(origin::kCUDA));
        origin::Tensor b({0.5f, 1.5f, 2.5f, 3.5f}, origin::Shape{2, 2},
                         origin::dtype(origin::Float32).device(origin::kCUDA));

        a.print("A");
        b.print("B");

        // 测试减法
        auto c_sub = a - b;
        c_sub.print("A - B");

        // 测试乘法
        auto c_mul = a * b;
        c_mul.print("A * B");

        // 测试除法
        auto c_div = a / b;
        c_div.print("A / B");
    }

    // 测试一元运算
    {
        std::cout << "\n=== Testing Unary Operations ===" << std::endl;

        origin::Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, origin::Shape{2, 2},
                         origin::dtype(origin::Float32).device(origin::kCUDA));

        a.print("A");

        // 测试平方
        auto c_square = F::square(a);
        c_square.print("A^2");

        // 测试指数
        auto c_exp = F::exp(a);
        c_exp.print("F::exp(A)");

        // 测试取负
        auto c_neg = -a;
        c_neg.print("-A");
    }

    // 测试标量运算
    {
        std::cout << "\n=== Testing Scalar Operations ===" << std::endl;

        origin::Tensor a({1.0f, 2.0f, 3.0f, 4.0f}, origin::Shape{2, 2},
                         origin::dtype(origin::Float32).device(origin::kCUDA));

        a.print("A");

        // 测试标量加法
        auto c_add_scalar = a + 5.0f;
        c_add_scalar.print("A + 5.0");

        // 测试标量乘法
        auto c_mul_scalar = a * 2.0f;
        c_mul_scalar.print("A * 2.0");
    }

    // 测试广播运算 - 暂时跳过，因为CUDA广播支持还未实现
    {
        std::cout << "\n=== Testing Broadcast Operations ===" << std::endl;
        std::cout << "Broadcast operations skipped - CUDA broadcast support not yet implemented" << std::endl;
    }

    // 性能测试
    {
        std::cout << "\n=== Performance Test ===" << std::endl;

        const size_t size = 1000;
        std::vector<float> data_a(size * size);
        std::vector<float> data_b(size * size);

        for (size_t i = 0; i < size * size; i++)
        {
            data_a[i] = static_cast<float>(i % 100);
            data_b[i] = static_cast<float>((i + 1) % 100);
        }

        origin::Tensor a(data_a, origin::Shape{size, size}, origin::dtype(origin::Float32).device(origin::kCUDA));
        origin::Tensor b(data_b, origin::Shape{size, size}, origin::dtype(origin::Float32).device(origin::kCUDA));

        // 预热
        auto warmup = a + b;
        cudaDeviceSynchronize();

        // 测试加法性能
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i)
        {
            auto c = a + b;
        }
        cudaDeviceSynchronize();
        auto end      = std::chrono::high_resolution_clock::now();
        auto add_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // 测试乘法性能
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i)
        {
            auto c = a * b;
        }
        cudaDeviceSynchronize();
        end           = std::chrono::high_resolution_clock::now();
        auto mul_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // 测试指数性能
        start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 100; ++i)
        {
            auto c = F::exp(a);
        }
        cudaDeviceSynchronize();
        end           = std::chrono::high_resolution_clock::now();
        auto exp_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "Performance results for " << size << "x" << size << " tensors (100 iterations):" << std::endl;
        std::cout << "Addition: " << add_time.count() / 100.0 << " microseconds per operation" << std::endl;
        std::cout << "Multiplication: " << mul_time.count() / 100.0 << " microseconds per operation" << std::endl;
        std::cout << "Exponential: " << exp_time.count() / 100.0 << " microseconds per operation" << std::endl;
    }

    std::cout << "\nAll CUDA operator tests completed successfully!" << std::endl;

#else
    std::cout << "CUDA support is not enabled in this build!" << std::endl;
    std::cout << "Please rebuild with --cuda flag: ./build.sh ORIGIN --cuda" << std::endl;
    return 1;
#endif

    return 0;
}
