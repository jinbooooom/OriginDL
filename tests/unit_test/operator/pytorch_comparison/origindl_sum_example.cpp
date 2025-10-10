#include <iostream>
#include <memory>
#include <vector>
#include "origin.h"

using namespace origin;

void origindl_sum_example()
{
    std::cout << "=== OriginDL Sum操作示例 ===" << std::endl;

    try
    {
        // 创建测试数据
        auto x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape{2, 3});
        std::cout << "原始张量形状: " << x.shape() << std::endl;
        std::cout << "原始张量数据: ";
        auto x_data = x.to_vector();
        for (size_t i = 0; i < x_data.size(); ++i)
        {
            std::cout << x_data[i] << " ";
        }
        std::cout << std::endl;

        // 沿轴0求和（列求和）
        auto sum_axis0 = sum(x, 0);
        std::cout << "\n沿轴0求和结果形状: " << sum_axis0.shape() << std::endl;
        std::cout << "沿轴0求和结果: ";
        auto sum0_data = sum_axis0.to_vector();
        for (size_t i = 0; i < sum0_data.size(); ++i)
        {
            std::cout << sum0_data[i] << " ";
        }
        std::cout << std::endl;

        // 沿轴1求和（行求和）
        auto sum_axis1 = sum(x, 1);
        std::cout << "\n沿轴1求和结果形状: " << sum_axis1.shape() << std::endl;
        std::cout << "沿轴1求和结果: ";
        auto sum1_data = sum_axis1.to_vector();
        for (size_t i = 0; i < sum1_data.size(); ++i)
        {
            std::cout << sum1_data[i] << " ";
        }
        std::cout << std::endl;

        // 全局求和
        auto sum_all = sum(x, -1);
        std::cout << "\n全局求和结果形状: " << sum_all.shape() << std::endl;
        std::cout << "全局求和结果: " << sum_all.item() << std::endl;

        // 三维张量测试
        std::cout << "\n=== 三维张量测试 ===" << std::endl;
        auto x3d = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, Shape{2, 2, 2});
        std::cout << "三维张量形状: " << x3d.shape() << std::endl;
        std::cout << "三维张量数据: ";
        auto x3d_data = x3d.to_vector();
        for (size_t i = 0; i < x3d_data.size(); ++i)
        {
            std::cout << x3d_data[i] << " ";
        }
        std::cout << std::endl;

        // 沿轴0求和
        auto sum3d_axis0 = sum(x3d, 0);
        std::cout << "\n三维张量沿轴0求和结果形状: " << sum3d_axis0.shape() << std::endl;
        std::cout << "三维张量沿轴0求和结果: ";
        auto sum3d0_data = sum3d_axis0.to_vector();
        for (size_t i = 0; i < sum3d0_data.size(); ++i)
        {
            std::cout << sum3d0_data[i] << " ";
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
    origindl_sum_example();
    return 0;
}
