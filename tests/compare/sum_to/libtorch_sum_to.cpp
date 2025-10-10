#include <torch/torch.h>
#include <iostream>

void test_libtorch_sum_to()
{
    std::cout << "=== libtorch Sum_to算子测试 ===" << std::endl;

    // 测试1: 正常压缩
    std::cout << "\n1. 正常压缩测试:" << std::endl;
    auto x1 = torch::tensor({{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}});
    std::cout << "输入张量: " << x1 << std::endl;
    std::cout << "输入形状: " << x1.sizes() << std::endl;

    try
    {
        auto result1 = torch::sum_to(x1, std::vector<int64_t>{2, 1});
        std::cout << "sum_to({2, 1})结果: " << result1 << std::endl;
        std::cout << "sum_to({2, 1})形状: " << result1.sizes() << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "sum_to({2, 1})失败: " << e.what() << std::endl;
    }

    // 测试2: 广播尝试（不支持）
    std::cout << "\n2. 广播尝试测试（应该不支持）:" << std::endl;
    auto x2 = torch::tensor({5.0});
    std::cout << "输入张量: " << x2 << std::endl;
    std::cout << "输入形状: " << x2.sizes() << std::endl;

    try
    {
        auto result2 = torch::sum_to(x2, std::vector<int64_t>{3});
        std::cout << "sum_to({3})结果: " << result2 << std::endl;
        std::cout << "sum_to({3})形状: " << result2.sizes() << std::endl;
        std::cout << "注意：libtorch的sum_to不支持广播，返回原始张量" << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "sum_to({3})失败: " << e.what() << std::endl;
    }

    // 测试3: 标量压缩
    std::cout << "\n3. 标量压缩测试:" << std::endl;
    try
    {
        auto result3 = torch::sum_to(x1, std::vector<int64_t>{});
        std::cout << "sum_to({})结果: " << result3 << std::endl;
        std::cout << "sum_to({})形状: " << result3.sizes() << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "sum_to({})失败: " << e.what() << std::endl;
    }

    // 测试4: 三维张量压缩
    std::cout << "\n4. 三维张量压缩测试:" << std::endl;
    auto x4 = torch::tensor({{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}});
    std::cout << "输入张量: " << x4 << std::endl;
    std::cout << "输入形状: " << x4.sizes() << std::endl;

    try
    {
        auto result4 = torch::sum_to(x4, std::vector<int64_t>{2, 2});
        std::cout << "sum_to({2, 2})结果: " << result4 << std::endl;
        std::cout << "sum_to({2, 2})形状: " << result4.sizes() << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "sum_to({2, 2})失败: " << e.what() << std::endl;
    }

    // 测试5: 相同形状
    std::cout << "\n5. 相同形状测试:" << std::endl;
    try
    {
        auto result5 = torch::sum_to(x1, std::vector<int64_t>{2, 3});
        std::cout << "sum_to({2, 3})结果: " << result5 << std::endl;
        std::cout << "sum_to({2, 3})形状: " << result5.sizes() << std::endl;
    }
    catch (const std::exception &e)
    {
        std::cout << "sum_to({2, 3})失败: " << e.what() << std::endl;
    }
}

int main()
{
    test_libtorch_sum_to();
    return 0;
}
