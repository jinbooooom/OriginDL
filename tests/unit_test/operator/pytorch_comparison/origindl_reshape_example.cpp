#include <arrayfire.h>
#include <iostream>
#include <memory>
#include <vector>
#include "origin.h"

using namespace origin;

void origindl_reshape_example()
{
    std::cout << "=== OriginDL Reshape操作示例 ===" << std::endl;

    try
    {
        // 基本reshape操作
        auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{1, 4});
        std::cout << "原始张量形状: " << x.shape() << std::endl;
        std::cout << "原始张量数据: ";
        auto x_data = x.to_vector();
        for (size_t i = 0; i < x_data.size(); ++i)
        {
            std::cout << x_data[i] << " ";
        }
        std::cout << std::endl;

        // reshape为4x1
        auto x_reshaped = reshape(x, Shape{4, 1});
        std::cout << "\nreshape为(4,1)后形状: " << x_reshaped.shape() << std::endl;
        std::cout << "reshape为(4,1)后数据: ";
        auto x_reshaped_data = x_reshaped.to_vector();
        for (size_t i = 0; i < x_reshaped_data.size(); ++i)
        {
            std::cout << x_reshaped_data[i] << " ";
        }
        std::cout << std::endl;

        // reshape为1x4
        auto x_reshaped2 = reshape(x, Shape{1, 4});
        std::cout << "\nreshape为(1,4)后形状: " << x_reshaped2.shape() << std::endl;
        std::cout << "reshape为(1,4)后数据: ";
        auto x_reshaped2_data = x_reshaped2.to_vector();
        for (size_t i = 0; i < x_reshaped2_data.size(); ++i)
        {
            std::cout << x_reshaped2_data[i] << " ";
        }
        std::cout << std::endl;

        // 从2D到1D
        std::cout << "\n=== 从2D到1D ===" << std::endl;
        auto x2d = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, Shape{2, 4});
        std::cout << "2D张量形状: " << x2d.shape() << std::endl;
        std::cout << "2D张量数据: ";
        auto x2d_data = x2d.to_vector();
        for (size_t i = 0; i < x2d_data.size(); ++i)
        {
            std::cout << x2d_data[i] << " ";
        }
        std::cout << std::endl;

        auto x1d = reshape(x2d, Shape{8});
        std::cout << "\nreshape为1D后形状: " << x1d.shape() << std::endl;
        std::cout << "reshape为1D后数据: ";
        auto x1d_data = x1d.to_vector();
        for (size_t i = 0; i < x1d_data.size(); ++i)
        {
            std::cout << x1d_data[i] << " ";
        }
        std::cout << std::endl;

        // 从1D到2D
        std::cout << "\n=== 从1D到2D ===" << std::endl;
        auto x1d_orig = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape{6});
        std::cout << "1D张量形状: " << x1d_orig.shape() << std::endl;
        std::cout << "1D张量数据: ";
        auto x1d_orig_data = x1d_orig.to_vector();
        for (size_t i = 0; i < x1d_orig_data.size(); ++i)
        {
            std::cout << x1d_orig_data[i] << " ";
        }
        std::cout << std::endl;

        auto x2d_new = reshape(x1d_orig, Shape{2, 3});
        std::cout << "\nreshape为(2,3)后形状: " << x2d_new.shape() << std::endl;
        std::cout << "reshape为(2,3)后数据: ";
        auto x2d_new_data = x2d_new.to_vector();
        for (size_t i = 0; i < x2d_new_data.size(); ++i)
        {
            std::cout << x2d_new_data[i] << " ";
        }
        std::cout << std::endl;

        // 标量reshape
        std::cout << "\n=== 标量reshape ===" << std::endl;
        auto x_scalar = Tensor({5.0}, Shape{1});
        std::cout << "标量张量形状: " << x_scalar.shape() << std::endl;
        std::cout << "标量张量数据: " << x_scalar.item() << std::endl;

        auto x_scalar_reshaped = reshape(x_scalar, Shape{1, 1});
        std::cout << "\n标量reshape为(1,1)后形状: " << x_scalar_reshaped.shape() << std::endl;
        std::cout << "标量reshape为(1,1)后数据: " << x_scalar_reshaped.item() << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cerr << "错误: " << e.what() << std::endl;
    }
}

int main()
{
    origindl_reshape_example();
    return 0;
}
