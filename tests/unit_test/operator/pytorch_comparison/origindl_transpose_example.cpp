#include <arrayfire.h>
#include <iostream>
#include <memory>
#include <vector>
#include "operator.h"
#include "tensor.h"

using namespace origin;

void origindl_transpose_example()
{
    std::cout << "=== OriginDL Transpose操作示例 ===" << std::endl;

    try
    {
        // 2D矩阵转置
        auto x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape{2, 3});
        std::cout << "原始矩阵形状: " << x.shape() << std::endl;
        std::cout << "原始矩阵数据: ";
        auto x_data = x.to_vector();
        for (size_t i = 0; i < x_data.size(); ++i)
        {
            std::cout << x_data[i] << " ";
        }
        std::cout << std::endl;

        // 转置操作
        auto x_t = transpose(x);
        std::cout << "\n转置后形状: " << x_t.shape() << std::endl;
        std::cout << "转置后数据: ";
        auto x_t_data = x_t.to_vector();
        for (size_t i = 0; i < x_t_data.size(); ++i)
        {
            std::cout << x_t_data[i] << " ";
        }
        std::cout << std::endl;

        // 一维张量转置
        std::cout << "\n=== 一维张量转置 ===" << std::endl;
        auto x1d = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{4});
        std::cout << "一维张量形状: " << x1d.shape() << std::endl;
        std::cout << "一维张量数据: ";
        auto x1d_data = x1d.to_vector();
        for (size_t i = 0; i < x1d_data.size(); ++i)
        {
            std::cout << x1d_data[i] << " ";
        }
        std::cout << std::endl;

        // 一维张量转置（应该保持一维）
        auto x1d_t = transpose(x1d);
        std::cout << "\n一维张量转置后形状: " << x1d_t.shape() << std::endl;
        std::cout << "一维张量转置后数据: ";
        auto x1d_t_data = x1d_t.to_vector();
        for (size_t i = 0; i < x1d_t_data.size(); ++i)
        {
            std::cout << x1d_t_data[i] << " ";
        }
        std::cout << std::endl;

        // 方阵转置
        std::cout << "\n=== 方阵转置 ===" << std::endl;
        auto x_square = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0}, Shape{3, 3});
        std::cout << "方阵形状: " << x_square.shape() << std::endl;
        std::cout << "方阵数据: ";
        auto x_square_data = x_square.to_vector();
        for (size_t i = 0; i < x_square_data.size(); ++i)
        {
            std::cout << x_square_data[i] << " ";
        }
        std::cout << std::endl;

        auto x_square_t = transpose(x_square);
        std::cout << "\n方阵转置后形状: " << x_square_t.shape() << std::endl;
        std::cout << "方阵转置后数据: ";
        auto x_square_t_data = x_square_t.to_vector();
        for (size_t i = 0; i < x_square_t_data.size(); ++i)
        {
            std::cout << x_square_t_data[i] << " ";
        }
        std::cout << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "错误: " << e.what() << std::endl;
    }
}

int main()
{
    origindl_transpose_example();
    return 0;
}
