#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "origin.h"

using namespace origin;

/**
 * @brief CUDA转置算子测试类
 * @details 测试CUDA张量的转置运算
 */
class CudaTransposeTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // 跳过所有测试：transpose目前只有CPU实现，没有CUDA实现
        GTEST_SKIP() << "transpose CUDA implementation not available yet";
    }

    void TearDown() override { cudaDeviceSynchronize(); }

    static constexpr double kFloatTolerance = 1e-5;
};

// ============================================================================
// 基础转置测试
// ============================================================================

TEST_F(CudaTransposeTest, BasicTranspose)
{
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto result = origin::transpose(a);

    EXPECT_EQ(result.shape(), Shape({2, 2}));
    EXPECT_EQ(result.dtype(), DataType::kFloat32);
    EXPECT_EQ(result.device().type(), DeviceType::kCUDA);

    auto result_data            = result.to_vector<float>();
    std::vector<float> expected = {1.0f, 3.0f, 2.0f, 4.0f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaTransposeTest, TransposeRectangularMatrix)
{
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}, dtype(Float32).device(kCUDA));

    auto result = origin::transpose(a);

    EXPECT_EQ(result.shape(), Shape({3, 2}));

    auto result_data            = result.to_vector<float>();
    std::vector<float> expected = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaTransposeTest, TransposeVector)
{
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{4}, dtype(Float32).device(kCUDA));

    auto result = origin::transpose(a);

    EXPECT_EQ(result.shape(), Shape({4}));

    auto result_data            = result.to_vector<float>();
    std::vector<float> expected = {1.0f, 2.0f, 3.0f, 4.0f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

// ============================================================================
// 边界情况测试
// ============================================================================

TEST_F(CudaTransposeTest, TransposeSingleElement)
{
    auto a = Tensor(std::vector<float>{42.0f}, Shape{1, 1}, dtype(Float32).device(kCUDA));

    auto result = origin::transpose(a);

    EXPECT_EQ(result.shape(), Shape({1, 1}));
    auto result_data = result.to_vector<float>();
    EXPECT_NEAR(result_data[0], 42.0f, kFloatTolerance);
}

TEST_F(CudaTransposeTest, TransposeLargeMatrix)
{
    const size_t m = 32, n = 32;
    std::vector<float> data(m * n);
    for (size_t i = 0; i < m * n; ++i)
    {
        data[i] = static_cast<float>(i);
    }

    auto a = Tensor(data, Shape{m, n}, dtype(Float32).device(kCUDA));

    auto result = origin::transpose(a);

    EXPECT_EQ(result.shape(), Shape({n, m}));

    auto result_data   = result.to_vector<float>();
    auto original_data = a.to_vector<float>();

    // 验证转置的正确性
    for (size_t i = 0; i < m; ++i)
    {
        for (size_t j = 0; j < n; ++j)
        {
            size_t original_idx   = i * n + j;
            size_t transposed_idx = j * m + i;
            EXPECT_NEAR(result_data[transposed_idx], original_data[original_idx], kFloatTolerance);
        }
    }
}

// ============================================================================
// 数学性质测试
// ============================================================================

TEST_F(CudaTransposeTest, DoubleTranspose)
{
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto transposed        = origin::transpose(a);
    auto double_transposed = origin::transpose(transposed);

    auto original_data = a.to_vector<float>();
    auto result_data   = double_transposed.to_vector<float>();

    for (size_t i = 0; i < original_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], original_data[i], kFloatTolerance);
    }
}

TEST_F(CudaTransposeTest, TransposeWithMatrixMultiplication)
{
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{2.0f, 0.0f, 1.0f, 2.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto ab            = origin::mat_mul(a, b);
    auto ab_transposed = origin::transpose(ab);

    auto a_transposed  = origin::transpose(a);
    auto b_transposed  = origin::transpose(b);
    auto ba_transposed = origin::mat_mul(b_transposed, a_transposed);

    auto data1 = ab_transposed.to_vector<float>();
    auto data2 = ba_transposed.to_vector<float>();

    for (size_t i = 0; i < data1.size(); ++i)
    {
        EXPECT_NEAR(data1[i], data2[i], kFloatTolerance);
    }
}
