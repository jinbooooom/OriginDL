#include <torch/torch.h>
#include <iostream>

void test_libtorch_sum()
{
    std::cout << "=== libtorch Sum算子测试 ===" << std::endl;

    // 测试1: 全局求和
    std::cout << "\n1. 全局求和测试:" << std::endl;
    auto x = torch::tensor({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    std::cout << "输入张量: " << x << std::endl;
    std::cout << "输入形状: " << x.sizes() << std::endl;

    auto result = x.sum();
    std::cout << "全局求和结果: " << result << std::endl;
    std::cout << "全局求和形状: " << result.sizes() << std::endl;

    // 测试2: 轴求和
    std::cout << "\n2. 轴求和测试:" << std::endl;
    auto result0 = x.sum(0, false);
    std::cout << "沿轴0求和: " << result0 << std::endl;
    std::cout << "沿轴0求和形状: " << result0.sizes() << std::endl;

    auto result1 = x.sum(1, false);
    std::cout << "沿轴1求和: " << result1 << std::endl;
    std::cout << "沿轴1求和形状: " << result1.sizes() << std::endl;

    // 测试3: keepdim参数
    std::cout << "\n3. keepdim参数测试:" << std::endl;
    auto result0_keepdim = x.sum(0, true);
    std::cout << "沿轴0求和(keepdim=true): " << result0_keepdim << std::endl;
    std::cout << "沿轴0求和(keepdim=true)形状: " << result0_keepdim.sizes() << std::endl;

    auto result0_no_keepdim = x.sum(0, false);
    std::cout << "沿轴0求和(keepdim=false): " << result0_no_keepdim << std::endl;
    std::cout << "沿轴0求和(keepdim=false)形状: " << result0_no_keepdim.sizes() << std::endl;

    // 测试4: 三维张量
    std::cout << "\n4. 三维张量测试:" << std::endl;
    auto x3d = torch::tensor({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    std::cout << "三维输入张量: " << x3d << std::endl;
    std::cout << "三维输入形状: " << x3d.sizes() << std::endl;

    auto result3d_0 = x3d.sum(0, false);
    std::cout << "沿轴0求和: " << result3d_0 << std::endl;
    std::cout << "沿轴0求和形状: " << result3d_0.sizes() << std::endl;

    auto result3d_1 = x3d.sum(1, false);
    std::cout << "沿轴1求和: " << result3d_1 << std::endl;
    std::cout << "沿轴1求和形状: " << result3d_1.sizes() << std::endl;

    auto result3d_2 = x3d.sum(2, false);
    std::cout << "沿轴2求和: " << result3d_2 << std::endl;
    std::cout << "沿轴2求和形状: " << result3d_2.sizes() << std::endl;
}

int main()
{
    test_libtorch_sum();
    return 0;
}
