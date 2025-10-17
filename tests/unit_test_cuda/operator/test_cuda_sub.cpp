#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "origin.h"

using namespace origin;

/**
 * @brief CUDA减法算子测试类
 * @details 测试CUDA张量的减法运算
 */
class CudaSubTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // 检查CUDA可用性
        if (!cuda::is_cuda_available())
        {
            GTEST_SKIP() << "CUDA is not available on this system";
        }
    }

    void TearDown() override
    {
        // 清理CUDA资源
        cudaDeviceSynchronize();
    }

    // 精度容忍常量
    static constexpr double kFloatTolerance = 1e-5;
};

// ============================================================================
// 基础减法测试
// ============================================================================

TEST_F(CudaSubTest, BasicSubtraction)
{
    // 测试基本减法运算
    auto a = Tensor(std::vector<float>{5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto result = a - b;

    EXPECT_EQ(result.shape(), Shape({2, 2}));
    EXPECT_EQ(result.dtype(), DataType::kFloat32);
    EXPECT_EQ(result.device().type(), DeviceType::kCUDA);

    auto result_data            = result.to_vector<float>();
    std::vector<float> expected = {4.0f, 4.0f, 4.0f, 4.0f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaSubTest, SubtractionWithZeros)
{
    // 测试包含零值的减法
    auto a = Tensor(std::vector<float>{3.0f, 0.0f, 0.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{0.0f, 1.0f, 2.0f, 0.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto result = a - b;

    auto result_data            = result.to_vector<float>();
    std::vector<float> expected = {3.0f, -1.0f, -2.0f, 4.0f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaSubTest, SubtractionWithNegatives)
{
    // 测试包含负值的减法
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, -3.0f, -4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{-1.0f, -2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto result = a - b;

    auto result_data            = result.to_vector<float>();
    std::vector<float> expected = {2.0f, 4.0f, -6.0f, -8.0f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

// ============================================================================
// 标量减法测试
// ============================================================================

TEST_F(CudaSubTest, ScalarSubtraction)
{
    // 测试标量减法 - 使用相同形状的张量进行逐元素减法
    auto a = Tensor(std::vector<float>{6.0f, 7.0f, 8.0f, 9.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{5.0f, 5.0f, 5.0f, 5.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto result1 = a - b;
    auto result2 = b - a;

    EXPECT_EQ(result1.shape(), Shape({2, 2}));
    EXPECT_EQ(result2.shape(), Shape({2, 2}));

    auto data1                   = result1.to_vector<float>();
    auto data2                   = result2.to_vector<float>();
    std::vector<float> expected1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> expected2 = {-1.0f, -2.0f, -3.0f, -4.0f};

    for (size_t i = 0; i < expected1.size(); ++i)
    {
        EXPECT_NEAR(data1[i], expected1[i], kFloatTolerance);
        EXPECT_NEAR(data2[i], expected2[i], kFloatTolerance);
    }
}

// ============================================================================
// 边界情况测试
// ============================================================================

TEST_F(CudaSubTest, SingleElementSubtraction)
{
    // 测试单元素张量减法
    auto a = Tensor(std::vector<float>{50.0f}, Shape{1}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{8.0f}, Shape{1}, dtype(Float32).device(kCUDA));

    auto result = a - b;

    EXPECT_EQ(result.shape(), Shape({1}));
    auto result_data = result.to_vector<float>();
    EXPECT_NEAR(result_data[0], 42.0f, kFloatTolerance);
}

TEST_F(CudaSubTest, LargeTensorSubtraction)
{
    // 测试大张量减法
    const size_t size = 10000;
    std::vector<float> data_a(size);
    std::vector<float> data_b(size);

    for (size_t i = 0; i < size; ++i)
    {
        data_a[i] = static_cast<float>(i * 3);
        data_b[i] = static_cast<float>(i * 2);
    }

    auto a = Tensor(data_a, Shape{size}, dtype(Float32).device(kCUDA));
    auto b = Tensor(data_b, Shape{size}, dtype(Float32).device(kCUDA));

    auto result = a - b;

    EXPECT_EQ(result.shape(), Shape({size}));

    auto result_data = result.to_vector<float>();

    // 验证结果正确性（只检查前100个元素）
    for (size_t i = 0; i < std::min(size, size_t(100)); ++i)
    {
        float expected = static_cast<float>(i * 3 - i * 2);
        EXPECT_NEAR(result_data[i], expected, kFloatTolerance);
    }
}
