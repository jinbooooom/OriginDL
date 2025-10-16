#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "origin.h"

using namespace origin;

/**
 * @brief CUDA加法算子测试类
 * @details 测试CUDA张量的加法运算
 */
class CudaAddTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 检查CUDA可用性
        if (!cuda::is_cuda_available()) {
            GTEST_SKIP() << "CUDA is not available on this system";
        }
    }
    
    void TearDown() override {
        // 清理CUDA资源
        cudaDeviceSynchronize();
    }
    
    // 精度容忍常量
    static constexpr double kFloatTolerance = 1e-5;
};

// ============================================================================
// 基础加法测试
// ============================================================================

TEST_F(CudaAddTest, BasicAddition) {
    // 测试基本加法运算
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = a + b;
    
    EXPECT_EQ(result.shape(), Shape({2, 2}));
    EXPECT_EQ(result.dtype(), DataType::kFloat32);
    EXPECT_EQ(result.device().type(), DeviceType::kCUDA);
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {6.0f, 8.0f, 10.0f, 12.0f};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaAddTest, AdditionWithZeros) {
    // 测试包含零值的加法
    auto a = Tensor(std::vector<float>{0.0f, 1.0f, 2.0f, 0.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{3.0f, 0.0f, 0.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = a + b;
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {3.0f, 1.0f, 2.0f, 4.0f};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaAddTest, AdditionWithNegatives) {
    // 测试包含负值的加法
    auto a = Tensor(std::vector<float>{-1.0f, -2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{1.0f, 2.0f, -3.0f, -4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = a + b;
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {0.0f, 0.0f, 0.0f, 0.0f};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

// ============================================================================
// 标量加法测试
// ============================================================================

TEST_F(CudaAddTest, ScalarAddition) {
    // 测试标量加法 - 使用相同形状的张量进行逐元素加法
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{5.0f, 5.0f, 5.0f, 5.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = a + b;
    
    EXPECT_EQ(result.shape(), Shape({2, 2}));
    
    auto data = result.to_vector<float>();
    std::vector<float> expected = {6.0f, 7.0f, 8.0f, 9.0f};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(data[i], expected[i], kFloatTolerance);
    }
}

// ============================================================================
// 形状测试
// ============================================================================

TEST_F(CudaAddTest, OneDimensionalAddition) {
    // 测试一维张量加法
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, Shape{5}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f, 0.5f}, Shape{5}, dtype(Float32).device(kCUDA));
    
    auto result = a + b;
    
    EXPECT_EQ(result.shape(), Shape({5}));
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaAddTest, ThreeDimensionalAddition) {
    // 测试三维张量加法
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2, 2}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f}, Shape{2, 2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = a + b;
    
    EXPECT_EQ(result.shape(), Shape({2, 2, 2}));
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

// ============================================================================
// 边界情况测试
// ============================================================================

TEST_F(CudaAddTest, SingleElementAddition) {
    // 测试单元素张量加法
    auto a = Tensor(std::vector<float>{42.0f}, Shape{1}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{8.0f}, Shape{1}, dtype(Float32).device(kCUDA));
    
    auto result = a + b;
    
    EXPECT_EQ(result.shape(), Shape({1}));
    auto result_data = result.to_vector<float>();
    EXPECT_NEAR(result_data[0], 50.0f, kFloatTolerance);
}

TEST_F(CudaAddTest, LargeTensorAddition) {
    // 测试大张量加法
    const size_t size = 10000;
    std::vector<float> data_a(size);
    std::vector<float> data_b(size);
    
    for (size_t i = 0; i < size; ++i) {
        data_a[i] = static_cast<float>(i);
        data_b[i] = static_cast<float>(i * 2);
    }
    
    auto a = Tensor(data_a, Shape{size}, dtype(Float32).device(kCUDA));
    auto b = Tensor(data_b, Shape{size}, dtype(Float32).device(kCUDA));
    
    auto result = a + b;
    
    EXPECT_EQ(result.shape(), Shape({size}));
    
    auto result_data = result.to_vector<float>();
    
    // 验证结果正确性（只检查前100个元素）
    for (size_t i = 0; i < std::min(size, size_t(100)); ++i) {
        float expected = static_cast<float>(i + i * 2);
        EXPECT_NEAR(result_data[i], expected, kFloatTolerance);
    }
}
