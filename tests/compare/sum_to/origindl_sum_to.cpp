#include "origin/core/tensor.h"
#include "origin/operators/sum_to.h"
#include <iostream>

void test_origindl_sum_to() {
    std::cout << "=== OriginDL Sum_to算子测试 ===" << std::endl;
    
    // 测试1: 正常压缩
    std::cout << "\n1. 正常压缩测试:" << std::endl;
    auto x1 = origin::Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, origin::Shape{2, 3});
    std::cout << "输入张量: ";
    auto x1_data = x1.to_vector();
    for (size_t i = 0; i < x1_data.size(); ++i) {
        std::cout << x1_data[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "输入形状: " << x1.shape().to_string() << std::endl;
    
    try {
        auto result1 = origin::sum_to(x1, origin::Shape{2, 1});
        std::cout << "sum_to({2, 1})结果: ";
        auto result1_data = result1.to_vector();
        for (size_t i = 0; i < result1_data.size(); ++i) {
            std::cout << result1_data[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "sum_to({2, 1})形状: " << result1.shape().to_string() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "sum_to({2, 1})失败: " << e.what() << std::endl;
    }
    
    // 测试2: 广播尝试（应该抛出异常）
    std::cout << "\n2. 广播尝试测试（应该抛出异常）:" << std::endl;
    auto x2 = origin::Tensor({5.0}, origin::Shape{1});
    std::cout << "输入张量: ";
    auto x2_data = x2.to_vector();
    for (size_t i = 0; i < x2_data.size(); ++i) {
        std::cout << x2_data[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "输入形状: " << x2.shape().to_string() << std::endl;
    
    try {
        auto result2 = origin::sum_to(x2, origin::Shape{3});
        std::cout << "sum_to({3})结果: ";
        auto result2_data = result2.to_vector();
        for (size_t i = 0; i < result2_data.size(); ++i) {
            std::cout << result2_data[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "sum_to({3})形状: " << result2.shape().to_string() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "sum_to({3})失败: " << e.what() << std::endl;
        std::cout << "注意：OriginDL的sum_to不支持广播，抛出异常" << std::endl;
    }
    
    // 测试3: 标量压缩
    std::cout << "\n3. 标量压缩测试:" << std::endl;
    try {
        auto result3 = origin::sum_to(x1, origin::Shape{});
        std::cout << "sum_to({})结果: ";
        auto result3_data = result3.to_vector();
        for (size_t i = 0; i < result3_data.size(); ++i) {
            std::cout << result3_data[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "sum_to({})形状: " << result3.shape().to_string() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "sum_to({})失败: " << e.what() << std::endl;
    }
    
    // 测试4: 三维张量压缩
    std::cout << "\n4. 三维张量压缩测试:" << std::endl;
    auto x4 = origin::Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, origin::Shape{2, 2, 2});
    std::cout << "输入张量: ";
    auto x4_data = x4.to_vector();
    for (size_t i = 0; i < x4_data.size(); ++i) {
        std::cout << x4_data[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "输入形状: " << x4.shape().to_string() << std::endl;
    
    try {
        auto result4 = origin::sum_to(x4, origin::Shape{2, 2});
        std::cout << "sum_to({2, 2})结果: ";
        auto result4_data = result4.to_vector();
        for (size_t i = 0; i < result4_data.size(); ++i) {
            std::cout << result4_data[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "sum_to({2, 2})形状: " << result4.shape().to_string() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "sum_to({2, 2})失败: " << e.what() << std::endl;
    }
    
    // 测试5: 相同形状
    std::cout << "\n5. 相同形状测试:" << std::endl;
    try {
        auto result5 = origin::sum_to(x1, origin::Shape{2, 3});
        std::cout << "sum_to({2, 3})结果: ";
        auto result5_data = result5.to_vector();
        for (size_t i = 0; i < result5_data.size(); ++i) {
            std::cout << result5_data[i] << " ";
        }
        std::cout << std::endl;
        std::cout << "sum_to({2, 3})形状: " << result5.shape().to_string() << std::endl;
    } catch (const std::exception& e) {
        std::cout << "sum_to({2, 3})失败: " << e.what() << std::endl;
    }
}

int main() {
    test_origindl_sum_to();
    return 0;
}
