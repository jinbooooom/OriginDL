#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "origin.h"

using namespace origin;

/**
 * @brief CUDA除法算子测试类
 * @details 测试CUDA张量的除法运算
 */
class CudaDivTest : public ::testing::Test
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
// 基础除法测试
// ============================================================================

TEST_F(CudaDivTest, BasicDivision)
{
    // 测试基本除法运算
    auto a = Tensor(std::vector<float>{8.0f, 12.0f, 16.0f, 20.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{2.0f, 3.0f, 4.0f, 5.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto result = a / b;

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

TEST_F(CudaDivTest, DivisionWithZeros)
{
    // 测试包含零值的除法（被除数为零）
    auto a = Tensor(std::vector<float>{0.0f, 6.0f, 0.0f, 8.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{2.0f, 3.0f, 4.0f, 2.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto result = a / b;

    auto result_data            = result.to_vector<float>();
    std::vector<float> expected = {0.0f, 2.0f, 0.0f, 4.0f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaDivTest, DivisionWithNegatives)
{
    // 测试包含负值的除法
    auto a = Tensor(std::vector<float>{-8.0f, -12.0f, 16.0f, 20.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{2.0f, -3.0f, -4.0f, 5.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto result = a / b;

    auto result_data            = result.to_vector<float>();
    std::vector<float> expected = {-4.0f, 4.0f, -4.0f, 4.0f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

// ============================================================================
// 标量除法测试
// ============================================================================

TEST_F(CudaDivTest, ScalarDivision)
{
    // 测试标量除法 - 使用相同形状的张量进行逐元素除法
    auto a = Tensor(std::vector<float>{8.0f, 12.0f, 16.0f, 20.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{2.0f, 2.0f, 2.0f, 2.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto result = a / b;

    EXPECT_EQ(result.shape(), Shape({2, 2}));

    auto data                   = result.to_vector<float>();
    std::vector<float> expected = {4.0f, 6.0f, 8.0f, 10.0f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(data[i], expected[i], kFloatTolerance);
    }
}

// ============================================================================
// 边界情况测试
// ============================================================================

TEST_F(CudaDivTest, SingleElementDivision)
{
    // 测试单元素张量除法
    auto a = Tensor(std::vector<float>{42.0f}, Shape{1}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{6.0f}, Shape{1}, dtype(Float32).device(kCUDA));

    auto result = a / b;

    EXPECT_EQ(result.shape(), Shape({1}));
    auto result_data = result.to_vector<float>();
    EXPECT_NEAR(result_data[0], 7.0f, kFloatTolerance);
}

TEST_F(CudaDivTest, LargeTensorDivision)
{
    // 测试大张量除法
    const size_t size = 10000;
    std::vector<float> data_a(size);
    std::vector<float> data_b(size);

    for (size_t i = 0; i < size; ++i)
    {
        data_a[i] = static_cast<float>((i + 1) * 2);
        data_b[i] = 2.0f;
    }

    auto a = Tensor(data_a, Shape{size}, dtype(Float32).device(kCUDA));
    auto b = Tensor(data_b, Shape{size}, dtype(Float32).device(kCUDA));

    auto result = a / b;

    EXPECT_EQ(result.shape(), Shape({size}));

    auto result_data = result.to_vector<float>();

    // 验证结果正确性（只检查前100个元素）
    for (size_t i = 0; i < std::min(size, size_t(100)); ++i)
    {
        float expected = static_cast<float>(i + 1);
        EXPECT_NEAR(result_data[i], expected, kFloatTolerance);
    }
}

TEST_F(CudaDivTest, ThreeDimensionalDivision)
{
    // 测试三维张量除法
    auto a = Tensor(std::vector<float>{8.0f, 16.0f, 24.0f, 32.0f, 40.0f, 48.0f, 56.0f, 64.0f}, Shape{2, 2, 2},
                    dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f, 14.0f, 16.0f}, Shape{2, 2, 2},
                    dtype(Float32).device(kCUDA));

    auto result = a / b;

    EXPECT_EQ(result.shape(), Shape({2, 2, 2}));

    auto result_data            = result.to_vector<float>();
    std::vector<float> expected = {4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f, 4.0f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

// ============================================================================
// 数值稳定性测试
// ============================================================================

TEST_F(CudaDivTest, SmallNumbersDivision)
{
    // 测试小数值除法
    auto a = Tensor(std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{0.01f, 0.02f, 0.03f, 0.04f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto result = a / b;

    auto result_data            = result.to_vector<float>();
    std::vector<float> expected = {10.0f, 10.0f, 10.0f, 10.0f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaDivTest, LargeNumbersDivision)
{
    // 测试大数值除法
    auto a = Tensor(std::vector<float>{1000000.0f, 2000000.0f, 3000000.0f, 4000000.0f}, Shape{2, 2},
                    dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{1000.0f, 2000.0f, 3000.0f, 4000.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto result = a / b;

    auto result_data            = result.to_vector<float>();
    std::vector<float> expected = {1000.0f, 1000.0f, 1000.0f, 1000.0f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}
