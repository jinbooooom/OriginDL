#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "origin.h"

using namespace origin;

/**
 * @brief CUDA求和算子测试类
 * @details 测试CUDA张量的求和运算
 */
class CudaSumTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 跳过所有测试：sum目前只有CPU实现，没有CUDA实现
        GTEST_SKIP() << "sum CUDA implementation not available yet";
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    
    static constexpr double kFloatTolerance = 1e-5;
};

// ============================================================================
// 基础求和测试
// ============================================================================

TEST_F(CudaSumTest, BasicSum) {
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = origin::sum(a);
    
    EXPECT_EQ(result.shape(), Shape({1}));
    EXPECT_EQ(result.dtype(), DataType::kFloat32);
    EXPECT_EQ(result.device().type(), DeviceType::kCUDA);
    
    auto result_data = result.to_vector<float>();
    EXPECT_NEAR(result_data[0], 10.0f, kFloatTolerance);
}

TEST_F(CudaSumTest, SumWithZeros) {
    auto a = Tensor(std::vector<float>{0.0f, 1.0f, 0.0f, 2.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = origin::sum(a);
    
    auto result_data = result.to_vector<float>();
    EXPECT_NEAR(result_data[0], 3.0f, kFloatTolerance);
}

TEST_F(CudaSumTest, SumWithNegatives) {
    auto a = Tensor(std::vector<float>{-1.0f, 2.0f, -3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = origin::sum(a);
    
    auto result_data = result.to_vector<float>();
    EXPECT_NEAR(result_data[0], 2.0f, kFloatTolerance);
}

// ============================================================================
// 边界情况测试
// ============================================================================

TEST_F(CudaSumTest, SumSingleElement) {
    auto a = Tensor(std::vector<float>{42.0f}, Shape{1}, dtype(Float32).device(kCUDA));
    
    auto result = origin::sum(a);
    
    EXPECT_EQ(result.shape(), Shape({1}));
    auto result_data = result.to_vector<float>();
    EXPECT_NEAR(result_data[0], 42.0f, kFloatTolerance);
}

TEST_F(CudaSumTest, SumLargeTensor) {
    const size_t size = 1000;
    std::vector<float> data(size);
    float expected_sum = 0.0f;
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i + 1);
        expected_sum += static_cast<float>(i + 1);
    }
    
    auto a = Tensor(data, Shape{size}, dtype(Float32).device(kCUDA));
    
    auto result = origin::sum(a);
    
    EXPECT_EQ(result.shape(), Shape({1}));
    auto result_data = result.to_vector<float>();
    EXPECT_NEAR(result_data[0], expected_sum, kFloatTolerance);
}

TEST_F(CudaSumTest, SumThreeDimensional) {
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = origin::sum(a);
    
    EXPECT_EQ(result.shape(), Shape({1}));
    auto result_data = result.to_vector<float>();
    EXPECT_NEAR(result_data[0], 36.0f, kFloatTolerance);
}

// ============================================================================
// 数值稳定性测试
// ============================================================================

TEST_F(CudaSumTest, SumSmallNumbers) {
    auto a = Tensor(std::vector<float>{0.001f, 0.002f, 0.003f, 0.004f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = origin::sum(a);
    
    auto result_data = result.to_vector<float>();
    EXPECT_NEAR(result_data[0], 0.01f, kFloatTolerance);
}

TEST_F(CudaSumTest, SumLargeNumbers) {
    auto a = Tensor(std::vector<float>{1000.0f, 2000.0f, 3000.0f, 4000.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = origin::sum(a);
    
    auto result_data = result.to_vector<float>();
    EXPECT_NEAR(result_data[0], 10000.0f, kFloatTolerance);
}

// ============================================================================
// 数学性质测试
// ============================================================================

TEST_F(CudaSumTest, SumCommutativity) {
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto sum_a = origin::sum(a);
    auto sum_b = origin::sum(b);
    auto sum_a_plus_b = origin::sum(a + b);
    
    auto data1 = sum_a.to_vector<float>();
    auto data2 = sum_b.to_vector<float>();
    auto data3 = sum_a_plus_b.to_vector<float>();
    
    EXPECT_NEAR(data3[0], data1[0] + data2[0], kFloatTolerance);
}

TEST_F(CudaSumTest, SumWithScalarMultiplication) {
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    float k = 3.0f;
    
    auto sum_a = origin::sum(a);
    auto sum_ka = origin::sum(a * k);
    
    auto data1 = sum_a.to_vector<float>();
    auto data2 = sum_ka.to_vector<float>();
    
    EXPECT_NEAR(data2[0], k * data1[0], kFloatTolerance);
}


