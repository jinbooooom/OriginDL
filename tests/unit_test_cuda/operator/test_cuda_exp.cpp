#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "origin.h"

using namespace origin;

/**
 * @brief CUDA指数算子测试类
 * @details 测试CUDA张量的指数运算
 */
class CudaExpTest : public ::testing::Test {
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
    static constexpr double kFloatTolerance = 1e-3;
};

// ============================================================================
// 基础指数测试
// ============================================================================

TEST_F(CudaExpTest, BasicExponential) {
    // 测试基本指数运算
    auto a = Tensor(std::vector<float>{0.0f, 1.0f, 2.0f, -1.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = origin::exp(a);
    
    EXPECT_EQ(result.shape(), Shape({2, 2}));
    EXPECT_EQ(result.dtype(), DataType::kFloat32);
    EXPECT_EQ(result.device().type(), DeviceType::kCUDA);
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {1.0f, std::exp(1.0f), std::exp(2.0f), std::exp(-1.0f)};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaExpTest, ExponentialOfZero) {
    // 测试零的指数
    auto a = Tensor(std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = origin::exp(a);
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {1.0f, 1.0f, 1.0f, 1.0f};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaExpTest, ExponentialOfNegativeNumbers) {
    // 测试负数的指数
    auto a = Tensor(std::vector<float>{-1.0f, -2.0f, -3.0f, -4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = origin::exp(a);
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {std::exp(-1.0f), std::exp(-2.0f), std::exp(-3.0f), std::exp(-4.0f)};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

// ============================================================================
// 特殊值测试
// ============================================================================

TEST_F(CudaExpTest, ExponentialOfOne) {
    // 测试1的指数
    auto a = Tensor(std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = origin::exp(a);
    
    auto result_data = result.to_vector<float>();
    float e_value = std::exp(1.0f);
    std::vector<float> expected = {e_value, e_value, e_value, e_value};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaExpTest, ExponentialOfLargeNumbers) {
    {
        // 测试大数的指数 - Float32精度
        auto a = Tensor(std::vector<double>{5.0, 10.0, 15.0, 20.0}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
        auto result = origin::exp(a);
        
        auto result_data = result.to_vector<double>();
        std::vector<double> expected = {std::exp(5.0), std::exp(10.0), std::exp(15.0), std::exp(20.0)};
        
        for (size_t i = 0; i < expected.size(); ++i) {
            // printf("result_data[%zu]: %.15f, expected[%zu]: %.15f\n", i, result_data[i], i, expected[i]);
            // 使用相对误差进行测试，更适合大数值
            double relative_error = std::abs(result_data[i] - expected[i]) / expected[i];
            // printf("相对误差[%zu]: %.2e\n", i, relative_error);
            EXPECT_LT(relative_error, kFloatTolerance);   // 考虑CUDA实现的精度限制
        }
    }

    // 测试大数的指数 - Float64精度
    {
    auto a = Tensor(std::vector<double>{5.0, 10.0, 15.0, 20.0}, Shape{2, 2}, dtype(Float64).device(kCUDA));
    
    auto result = origin::exp(a);
    
    auto result_data = result.to_vector<double>();
    std::vector<double> expected = {std::exp(5.0), std::exp(10.0), std::exp(15.0), std::exp(20.0)};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        // printf("result_data[%zu]: %.15f, expected[%zu]: %.15f\n", i, result_data[i], i, expected[i]);
        // 使用相对误差进行测试，更适合大数值
        double relative_error = std::abs(result_data[i] - expected[i]) / expected[i];
        // printf("相对误差[%zu]: %.2e\n", i, relative_error);
        EXPECT_LT(relative_error, kFloatTolerance);   // 考虑CUDA实现的精度限制
    }
}

}

// ============================================================================
// 边界情况测试
// ============================================================================

TEST_F(CudaExpTest, SingleElementExponential) {
    // 测试单元素张量指数
    auto a = Tensor(std::vector<float>{2.0f}, Shape{1}, dtype(Float32).device(kCUDA));
    
    auto result = origin::exp(a);
    
    EXPECT_EQ(result.shape(), Shape({1}));
    auto result_data = result.to_vector<float>();
    EXPECT_NEAR(result_data[0], std::exp(2.0f), kFloatTolerance);
}

TEST_F(CudaExpTest, LargeTensorExponential) {
    // 测试大张量指数
    const size_t size = 1000;
    std::vector<float> data_a(size);
    
    for (size_t i = 0; i < size; ++i) {
        data_a[i] = static_cast<float>(i) * 0.1f;
    }
    
    auto a = Tensor(data_a, Shape{size}, dtype(Float32).device(kCUDA));
    
    auto result = origin::exp(a);
    
    EXPECT_EQ(result.shape(), Shape({size}));
    
    auto result_data = result.to_vector<float>();
    
    // 验证结果正确性（只检查前100个元素）- 使用相对误差
    for (size_t i = 0; i < std::min(size, size_t(100)); ++i) {
        float expected = std::exp(static_cast<float>(i) * 0.1f);
        // 使用相对误差进行测试，更适合大数值
        double relative_error = std::abs(result_data[i] - expected) / expected;
        EXPECT_LT(relative_error, kFloatTolerance);
    }
}

TEST_F(CudaExpTest, ThreeDimensionalExponential) {
    // 测试三维张量指数
    auto a = Tensor(std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}, Shape{2, 2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = origin::exp(a);
    
    EXPECT_EQ(result.shape(), Shape({2, 2, 2}));
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {
        std::exp(0.0f), std::exp(1.0f), std::exp(2.0f), std::exp(3.0f),
        std::exp(4.0f), std::exp(5.0f), std::exp(6.0f), std::exp(7.0f)
    };
    
    for (size_t i = 0; i < expected.size(); ++i) {
        // 对于包含较大数值的测试，使用相对误差
        double relative_error = std::abs(result_data[i] - expected[i]) / expected[i];
        EXPECT_LT(relative_error, kFloatTolerance);
    }
}

// ============================================================================
// 数值稳定性测试
// ============================================================================

TEST_F(CudaExpTest, SmallNumbersExponential) {
    // 测试小数值的指数
    auto a = Tensor(std::vector<float>{0.001f, 0.01f, 0.1f, 0.5f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = origin::exp(a);
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {std::exp(0.001f), std::exp(0.01f), std::exp(0.1f), std::exp(0.5f)};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaExpTest, VerySmallNumbersExponential) {
    // 测试非常小数值的指数
    auto a = Tensor(std::vector<float>{-10.0f, -20.0f, -30.0f, -40.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = origin::exp(a);
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {std::exp(-10.0f), std::exp(-20.0f), std::exp(-30.0f), std::exp(-40.0f)};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

// ============================================================================
// 数学性质测试
// ============================================================================

TEST_F(CudaExpTest, ExponentialIdentity) {
    // 测试指数恒等式：exp(0) = 1
    auto a = Tensor(std::vector<float>{0.0f}, Shape{1}, dtype(Float32).device(kCUDA));
    
    auto result = origin::exp(a);
    
    auto result_data = result.to_vector<float>();
    EXPECT_NEAR(result_data[0], 1.0f, kFloatTolerance);
}

TEST_F(CudaExpTest, ExponentialMonotonicity) {
    // 测试指数的单调性：如果 a < b，则 exp(a) < exp(b)
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = origin::exp(a);
    
    auto result_data = result.to_vector<float>();
    
    // 验证单调性
    for (size_t i = 1; i < result_data.size(); ++i) {
        EXPECT_GT(result_data[i], result_data[i-1]);
    }
}


