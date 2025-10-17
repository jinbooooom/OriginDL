#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include "origin.h"

using namespace origin;

/**
 * @brief CUDA重塑算子测试类
 * @details 测试CUDA张量的重塑运算
 */
class CudaReshapeTest : public ::testing::Test {
protected:
    void SetUp() override {
        // 跳过所有测试：reshape目前只有CPU实现，没有CUDA实现
        GTEST_SKIP() << "reshape CUDA implementation not available yet";
    }
    
    void TearDown() override {
        cudaDeviceSynchronize();
    }
    
    static constexpr double kFloatTolerance = 1e-5;
};

// ============================================================================
// 基础重塑测试
// ============================================================================

TEST_F(CudaReshapeTest, BasicReshape) {
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = origin::reshape(a, Shape{4});
    
    EXPECT_EQ(result.shape(), Shape({4}));
    EXPECT_EQ(result.dtype(), DataType::kFloat32);
    EXPECT_EQ(result.device().type(), DeviceType::kCUDA);
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaReshapeTest, Reshape2DTo3D) {
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}, dtype(Float32).device(kCUDA));
    
    auto result = origin::reshape(a, Shape{2, 3, 1});
    
    EXPECT_EQ(result.shape(), Shape({2, 3, 1}));
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaReshapeTest, Reshape3DTo2D) {
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2, 2}, dtype(Float32).device(kCUDA));
    
    auto result = origin::reshape(a, Shape{4, 2});
    
    EXPECT_EQ(result.shape(), Shape({4, 2}));
    
    auto result_data = result.to_vector<float>();
    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

// ============================================================================
// 边界情况测试
// ============================================================================

TEST_F(CudaReshapeTest, ReshapeToSingleElement) {
    auto a = Tensor(std::vector<float>{42.0f}, Shape{1}, dtype(Float32).device(kCUDA));
    
    auto result = origin::reshape(a, Shape{1, 1});
    
    EXPECT_EQ(result.shape(), Shape({1, 1}));
    auto result_data = result.to_vector<float>();
    EXPECT_NEAR(result_data[0], 42.0f, kFloatTolerance);
}

TEST_F(CudaReshapeTest, ReshapeLargeTensor) {
    const size_t size = 1000;
    std::vector<float> data(size);
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<float>(i);
    }
    
    auto a = Tensor(data, Shape{size}, dtype(Float32).device(kCUDA));
    
    auto result = origin::reshape(a, Shape{10, 100});
    
    EXPECT_EQ(result.shape(), Shape({10, 100}));
    
    auto result_data = result.to_vector<float>();
    for (size_t i = 0; i < std::min(size, size_t(100)); ++i) {
        EXPECT_NEAR(result_data[i], static_cast<float>(i), kFloatTolerance);
    }
}

// ============================================================================
// 数据一致性测试
// ============================================================================

TEST_F(CudaReshapeTest, DataConsistency) {
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}, dtype(Float32).device(kCUDA));
    
    auto result1 = origin::reshape(a, Shape{3, 2});
    auto result2 = origin::reshape(a, Shape{6});
    auto result3 = origin::reshape(a, Shape{1, 6});
    
    auto data1 = result1.to_vector<float>();
    auto data2 = result2.to_vector<float>();
    auto data3 = result3.to_vector<float>();
    
    for (size_t i = 0; i < 6; ++i) {
        EXPECT_NEAR(data1[i], data2[i], kFloatTolerance);
        EXPECT_NEAR(data1[i], data3[i], kFloatTolerance);
    }
}


