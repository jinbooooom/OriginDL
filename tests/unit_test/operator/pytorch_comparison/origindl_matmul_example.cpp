#include <arrayfire.h>
#include <iostream>
#include <memory>
#include <vector>
#include "origin.h"

using namespace origin;

void origindl_matmul_example()
{
    std::cout << "=== OriginDL MatMul操作示例 ===" << std::endl;

    try
    {
        // 基本矩阵乘法
        auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
        auto w = Tensor({5.0, 6.0, 7.0, 8.0}, Shape{2, 2});

        std::cout << "矩阵x形状: " << x.shape() << std::endl;
        std::cout << "矩阵x数据: ";
        auto x_data = x.to_vector();
        for (size_t i = 0; i < x_data.size(); ++i)
        {
            std::cout << x_data[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "矩阵w形状: " << w.shape() << std::endl;
        std::cout << "矩阵w数据: ";
        auto w_data = w.to_vector();
        for (size_t i = 0; i < w_data.size(); ++i)
        {
            std::cout << w_data[i] << " ";
        }
        std::cout << std::endl;

        // 矩阵乘法
        auto result = mat_mul(x, w);
        std::cout << "\n矩阵乘法结果形状: " << result.shape() << std::endl;
        std::cout << "矩阵乘法结果: ";
        auto result_data = result.to_vector();
        for (size_t i = 0; i < result_data.size(); ++i)
        {
            std::cout << result_data[i] << " ";
        }
        std::cout << std::endl;

        // 不同尺寸的矩阵乘法
        std::cout << "\n=== 不同尺寸矩阵乘法 ===" << std::endl;
        auto x2 = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape{2, 3});
        auto w2 = Tensor({7.0, 8.0, 9.0, 10.0, 11.0, 12.0}, Shape{3, 2});

        std::cout << "矩阵x2形状: " << x2.shape() << std::endl;
        std::cout << "矩阵x2数据: ";
        auto x2_data = x2.to_vector();
        for (size_t i = 0; i < x2_data.size(); ++i)
        {
            std::cout << x2_data[i] << " ";
        }
        std::cout << std::endl;

        std::cout << "矩阵w2形状: " << w2.shape() << std::endl;
        std::cout << "矩阵w2数据: ";
        auto w2_data = w2.to_vector();
        for (size_t i = 0; i < w2_data.size(); ++i)
        {
            std::cout << w2_data[i] << " ";
        }
        std::cout << std::endl;

        auto result2 = mat_mul(x2, w2);
        std::cout << "\n不同尺寸矩阵乘法结果形状: " << result2.shape() << std::endl;
        std::cout << "不同尺寸矩阵乘法结果: ";
        auto result2_data = result2.to_vector();
        for (size_t i = 0; i < result2_data.size(); ++i)
        {
            std::cout << result2_data[i] << " ";
        }
        std::cout << std::endl;

        // 标量矩阵乘法
        std::cout << "\n=== 标量矩阵乘法 ===" << std::endl;
        auto x_scalar = Tensor({2.0}, Shape{1, 1});
        auto w_scalar = Tensor({3.0}, Shape{1, 1});

        std::cout << "标量矩阵x形状: " << x_scalar.shape() << std::endl;
        std::cout << "标量矩阵x数据: " << x_scalar.item() << std::endl;
        std::cout << "标量矩阵w形状: " << w_scalar.shape() << std::endl;
        std::cout << "标量矩阵w数据: " << w_scalar.item() << std::endl;

        auto result_scalar = mat_mul(x_scalar, w_scalar);
        std::cout << "\n标量矩阵乘法结果形状: " << result_scalar.shape() << std::endl;
        std::cout << "标量矩阵乘法结果: " << result_scalar.item() << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "错误: " << e.what() << std::endl;
    }
}

int main()
{
    origindl_matmul_example();
    return 0;
}
