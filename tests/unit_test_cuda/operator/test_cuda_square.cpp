#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "origin.h"

using namespace origin;

/**
 * @brief CUDA平方算子测试类
 * @details 测试CUDA张量的平方运算
 */
class CudaSquareTest : public ::testing::Test {
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
// 基础平方测试
// ============================================================================

TEST_F(CudaSquareTest, BasicSquare) {
    // 测试基本平方运算
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = square(a);
    
    EXPECT_EQ(result.shape(), Shape({2, 2}));
    EXPECT_EQ(result.dtype(), DataType::kFloat32);
    EXPECT_EQ(result.device().type(), DeviceType::kCUDA);
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {1.0f, 4.0f, 9.0f, 16.0f};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaSquareTest, SquareOfZero) {
    // 测试零的平方
    auto a = Tensor(std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = square(a);
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {0.0f, 0.0f, 0.0f, 0.0f};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaSquareTest, SquareOfOne) {
    // 测试1的平方
    auto a = Tensor(std::vector<float>{1.0f, 1.0f, 1.0f, 1.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = square(a);
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {1.0f, 1.0f, 1.0f, 1.0f};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

// ============================================================================
// 特殊值测试
// ============================================================================

TEST_F(CudaSquareTest, SquareOfNegativeNumbers) {
    // 测试负数的平方
    auto a = Tensor(std::vector<float>{-1.0f, -2.0f, -3.0f, -4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = square(a);
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {1.0f, 4.0f, 9.0f, 16.0f};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaSquareTest, SquareOfMixedNumbers) {
    // 测试正负混合数的平方
    auto a = Tensor(std::vector<float>{-2.0f, 3.0f, -4.0f, 5.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = square(a);
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {4.0f, 9.0f, 16.0f, 25.0f};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaSquareTest, SquareOfSmallNumbers) {
    // 测试小数的平方
    auto a = Tensor(std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = square(a);
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {0.01f, 0.04f, 0.09f, 0.16f};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaSquareTest, SquareOfLargeNumbers) {
    // 测试大数的平方
    auto a = Tensor(std::vector<float>{10.0f, 20.0f, 30.0f, 40.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = square(a);
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {100.0f, 400.0f, 900.0f, 1600.0f};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

// ============================================================================
// 边界情况测试
// ============================================================================

TEST_F(CudaSquareTest, SingleElementSquare) {
    // 测试单元素张量平方
    auto a = Tensor(std::vector<float>{5.0f}, Shape{1}, dtype(Float32).device(kCUDA));
    
    auto result = square(a);
    
    EXPECT_EQ(result.shape(), Shape({1}));
    auto result_data = result.to_vector<float>();
    EXPECT_NEAR(result_data[0], 25.0f, kFloatTolerance);
}

TEST_F(CudaSquareTest, LargeTensorSquare) {
    // 测试大张量平方
    const size_t size = 1000;
    std::vector<float> data_a(size);
    
    for (size_t i = 0; i < size; ++i) {
        data_a[i] = static_cast<float>(i + 1);
    }
    
    auto a = Tensor(data_a, Shape{size}, dtype(Float32).device(kCUDA));
    
    auto result = square(a);
    
    EXPECT_EQ(result.shape(), Shape({size}));
    
    auto result_data = result.to_vector<float>();
    
    // 验证结果正确性（只检查前100个元素）
    for (size_t i = 0; i < std::min(size, size_t(100)); ++i) {
        float expected = static_cast<float>((i + 1) * (i + 1));
        EXPECT_NEAR(result_data[i], expected, kFloatTolerance);
    }
}

TEST_F(CudaSquareTest, ThreeDimensionalSquare) {
    // 测试三维张量平方
    auto a = Tensor(std::vector<float>{
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    }, Shape{2, 2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = square(a);
    
    EXPECT_EQ(result.shape(), Shape({2, 2, 2}));
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {1.0f, 4.0f, 9.0f, 16.0f, 25.0f, 36.0f, 49.0f, 64.0f};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

// ============================================================================
// 数值稳定性测试
// ============================================================================

TEST_F(CudaSquareTest, VerySmallNumbersSquare) {
    // 测试非常小数的平方
    auto a = Tensor(std::vector<float>{1e-5f, 1e-4f, 1e-3f, 1e-2f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = square(a);
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {1e-10f, 1e-8f, 1e-6f, 1e-4f};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaSquareTest, VeryLargeNumbersSquare) {
    // 测试非常大数的平方
    auto a = Tensor(std::vector<float>{1e5f, 1e6f, 1e7f, 1e8f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = square(a);
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {1e10f, 1e12f, 1e14f, 1e16f};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

// ============================================================================
// 数学性质测试
// ============================================================================

TEST_F(CudaSquareTest, SquareIdentity) {
    // 测试平方恒等式：0^2 = 0, 1^2 = 1
    auto a = Tensor(std::vector<float>{0.0f, 1.0f}, Shape{2}, dtype(Float32).device(kCUDA));
    
    auto result = square(a);
    
    auto result_data = result.to_vector<float>();
    EXPECT_NEAR(result_data[0], 0.0f, kFloatTolerance);
    EXPECT_NEAR(result_data[1], 1.0f, kFloatTolerance);
}

TEST_F(CudaSquareTest, SquareSymmetry) {
    // 测试平方的对称性：(-x)^2 = x^2
    auto a = Tensor(std::vector<float>{2.0f, 3.0f, 4.0f, 5.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto neg_a = Tensor(std::vector<float>{-2.0f, -3.0f, -4.0f, -5.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result1 = square(a);
    auto result2 = square(neg_a);
    
    auto data1 = result1.to_vector<float>();
    auto data2 = result2.to_vector<float>();
    
    for (size_t i = 0; i < data1.size(); ++i) {
        EXPECT_NEAR(data1[i], data2[i], kFloatTolerance);
    }
}

TEST_F(CudaSquareTest, SquareInverseOfSquareRoot) {
    // 测试平方和平方根的逆运算关系：sqrt(x^2) = |x|
    // 注意：由于没有sqrt算子，我们跳过这个测试
    // 这个测试可以用于验证square算子的正确性
    auto a = Tensor(std::vector<float>{2.0f, 3.0f, 4.0f, 5.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto square_result = square(a);
    
    auto original_data = a.to_vector<float>();
    auto result_data = square_result.to_vector<float>();
    
    // 验证平方运算的正确性
    for (size_t i = 0; i < original_data.size(); ++i) {
        EXPECT_NEAR(result_data[i], original_data[i] * original_data[i], kFloatTolerance);
    }
}

TEST_F(CudaSquareTest, SquareMonotonicity) {
    // 测试平方的单调性：对于正数，如果 a < b，则 a^2 < b^2
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = square(a);
    
    auto result_data = result.to_vector<float>();
    
    // 验证单调性
    for (size_t i = 1; i < result_data.size(); ++i) {
        EXPECT_GT(result_data[i], result_data[i-1]);
    }
}


