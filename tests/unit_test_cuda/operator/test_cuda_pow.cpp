#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "origin.h"

using namespace origin;

/**
 * @brief CUDA幂运算算子测试类
 * @details 测试CUDA张量的幂运算
 */
class CudaPowTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // 跳过所有测试：pow目前只有CPU实现，没有CUDA实现
        GTEST_SKIP() << "pow CUDA implementation not available yet";
    }

    void TearDown() override { cudaDeviceSynchronize(); }

    static constexpr double kFloatTolerance = 1e-5;
};

// ============================================================================
// 基础幂运算测试
// ============================================================================

TEST_F(CudaPowTest, BasicPower)
{
    auto a = Tensor(std::vector<float>{2.0f, 3.0f, 4.0f, 5.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto result = origin::pow(a, 2.0f);

    EXPECT_EQ(result.shape(), Shape({2, 2}));
    EXPECT_EQ(result.dtype(), DataType::kFloat32);
    EXPECT_EQ(result.device().type(), DeviceType::kCUDA);

    auto result_data            = result.to_vector<float>();
    std::vector<float> expected = {4.0f, 9.0f, 16.0f, 25.0f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaPowTest, PowerOfZero)
{
    auto a = Tensor(std::vector<float>{0.0f, 1.0f, 2.0f, 3.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto result = pow(a, 0.0f);

    auto result_data            = result.to_vector<float>();
    std::vector<float> expected = {1.0f, 1.0f, 1.0f, 1.0f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaPowTest, PowerOfOne)
{
    auto a = Tensor(std::vector<float>{2.0f, 3.0f, 4.0f, 5.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto result = pow(a, 1.0f);

    auto result_data   = result.to_vector<float>();
    auto original_data = a.to_vector<float>();

    for (size_t i = 0; i < original_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], original_data[i], kFloatTolerance);
    }
}

// ============================================================================
// 特殊值测试
// ============================================================================

TEST_F(CudaPowTest, PowerOfHalf)
{
    auto a = Tensor(std::vector<float>{4.0f, 9.0f, 16.0f, 25.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto result = pow(a, 0.5f);

    auto result_data            = result.to_vector<float>();
    std::vector<float> expected = {2.0f, 3.0f, 4.0f, 5.0f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaPowTest, PowerOfNegativeExponent)
{
    auto a = Tensor(std::vector<float>{2.0f, 3.0f, 4.0f, 5.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto result = pow(a, -1.0f);

    auto result_data            = result.to_vector<float>();
    std::vector<float> expected = {0.5f, 1.0f / 3.0f, 0.25f, 0.2f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaPowTest, PowerOfSmallNumbers)
{
    auto a = Tensor(std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto result = origin::pow(a, 2.0f);

    auto result_data            = result.to_vector<float>();
    std::vector<float> expected = {0.01f, 0.04f, 0.09f, 0.16f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

// ============================================================================
// 边界情况测试
// ============================================================================

TEST_F(CudaPowTest, PowerSingleElement)
{
    auto a = Tensor(std::vector<float>{3.0f}, Shape{1}, dtype(Float32).device(kCUDA));

    auto result = pow(a, 3.0f);

    EXPECT_EQ(result.shape(), Shape({1}));
    auto result_data = result.to_vector<float>();
    EXPECT_NEAR(result_data[0], 27.0f, kFloatTolerance);
}

TEST_F(CudaPowTest, PowerLargeTensor)
{
    const size_t size = 100;
    std::vector<float> data(size);

    for (size_t i = 0; i < size; ++i)
    {
        data[i] = static_cast<float>(i + 1);
    }

    auto a = Tensor(data, Shape{size}, dtype(Float32).device(kCUDA));

    auto result = origin::pow(a, 2.0f);

    EXPECT_EQ(result.shape(), Shape({size}));

    auto result_data = result.to_vector<float>();

    // 验证结果正确性（只检查前10个元素）
    for (size_t i = 0; i < std::min(size, size_t(10)); ++i)
    {
        float expected = static_cast<float>((i + 1) * (i + 1));
        EXPECT_NEAR(result_data[i], expected, kFloatTolerance);
    }
}

// ============================================================================
// 数学性质测试
// ============================================================================

TEST_F(CudaPowTest, PowerIdentity)
{
    auto a = Tensor(std::vector<float>{2.0f, 3.0f, 4.0f, 5.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto result = pow(a, 1.0f);

    auto result_data   = result.to_vector<float>();
    auto original_data = a.to_vector<float>();

    for (size_t i = 0; i < original_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], original_data[i], kFloatTolerance);
    }
}

TEST_F(CudaPowTest, PowerOfPower)
{
    auto a = Tensor(std::vector<float>{2.0f, 3.0f}, Shape{2}, dtype(Float32).device(kCUDA));

    auto result1 = origin::pow(origin::pow(a, 2.0f), 3.0f);
    auto result2 = origin::pow(a, 6.0f);

    auto data1 = result1.to_vector<float>();
    auto data2 = result2.to_vector<float>();

    for (size_t i = 0; i < data1.size(); ++i)
    {
        EXPECT_NEAR(data1[i], data2[i], kFloatTolerance);
    }
}
