#include <iostream>
#include "origin/core/tensor.h"
#include "origin/operators/sum.h"

void test_origindl_sum()
{
    std::cout << "=== OriginDL Sum算子测试 ===" << std::endl;

    // 测试1: 全局求和
    std::cout << "\n1. 全局求和测试:" << std::endl;
    auto x = origin::Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, origin::Shape{2, 3});
    std::cout << "输入张量: ";
    auto x_data = x.to_vector();
    for (size_t i = 0; i < x_data.size(); ++i)
    {
        std::cout << x_data[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "输入形状: " << x.shape().to_string() << std::endl;

    auto result = origin::sum(x, -1);
    std::cout << "全局求和结果: ";
    auto result_data = result.to_vector();
    for (size_t i = 0; i < result_data.size(); ++i)
    {
        std::cout << result_data[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "全局求和形状: " << result.shape().to_string() << std::endl;

    // 测试2: 轴求和
    std::cout << "\n2. 轴求和测试:" << std::endl;
    auto result0 = origin::sum(x, 0);
    std::cout << "沿轴0求和: ";
    auto result0_data = result0.to_vector();
    for (size_t i = 0; i < result0_data.size(); ++i)
    {
        std::cout << result0_data[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "沿轴0求和形状: " << result0.shape().to_string() << std::endl;

    auto result1 = origin::sum(x, 1);
    std::cout << "沿轴1求和: ";
    auto result1_data = result1.to_vector();
    for (size_t i = 0; i < result1_data.size(); ++i)
    {
        std::cout << result1_data[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "沿轴1求和形状: " << result1.shape().to_string() << std::endl;

    // 测试3: 三维张量
    std::cout << "\n3. 三维张量测试:" << std::endl;
    auto x3d = origin::Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, origin::Shape{2, 2, 2});
    std::cout << "三维输入张量: ";
    auto x3d_data = x3d.to_vector();
    for (size_t i = 0; i < x3d_data.size(); ++i)
    {
        std::cout << x3d_data[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "三维输入形状: " << x3d.shape().to_string() << std::endl;

    auto result3d_0 = origin::sum(x3d, 0);
    std::cout << "沿轴0求和: ";
    auto result3d_0_data = result3d_0.to_vector();
    for (size_t i = 0; i < result3d_0_data.size(); ++i)
    {
        std::cout << result3d_0_data[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "沿轴0求和形状: " << result3d_0.shape().to_string() << std::endl;

    auto result3d_1 = origin::sum(x3d, 1);
    std::cout << "沿轴1求和: ";
    auto result3d_1_data = result3d_1.to_vector();
    for (size_t i = 0; i < result3d_1_data.size(); ++i)
    {
        std::cout << result3d_1_data[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "沿轴1求和形状: " << result3d_1.shape().to_string() << std::endl;

    auto result3d_2 = origin::sum(x3d, 2);
    std::cout << "沿轴2求和: ";
    auto result3d_2_data = result3d_2.to_vector();
    for (size_t i = 0; i < result3d_2_data.size(); ++i)
    {
        std::cout << result3d_2_data[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "沿轴2求和形状: " << result3d_2.shape().to_string() << std::endl;
}

int main()
{
    test_origindl_sum();
    return 0;
}
