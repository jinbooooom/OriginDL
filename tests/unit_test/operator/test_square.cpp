#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "origin.h"

using namespace origin;

class SquareOperatorTest : public ::testing::Test
{
protected:
    // 精度忍受常量
    static constexpr double kTolerance = 1e-3;
    void SetUp() override
    {
        // 测试前的设置
        // 测试前的设置
    }

    void TearDown() override
    {
        // 测试后的清理
    }

    // 辅助函数：比较两个浮点数是否相等（考虑浮点精度）
    bool isEqual(double a, double b, double tolerance = kTolerance) { return std::abs(a - b) < tolerance; }

    // 辅助函数：比较两个Tensor是否相等
    bool tensorsEqual(const Tensor &a, const Tensor &b, double tolerance = kTolerance)
    {
        if (a.shape() != b.shape())
        {
            return false;
        }

        auto data_a = a.to_vector<float>();
        auto data_b = b.to_vector<float>();

        if (data_a.size() != data_b.size())
        {
            return false;
        }

        for (size_t i = 0; i < data_a.size(); ++i)
        {
            if (!isEqual(data_a[i], data_b[i], tolerance))
            {
                return false;
            }
        }
        return true;
    }
};

// ==================== 前向传播测试 ====================

TEST_F(SquareOperatorTest, ForwardBasic)
{
    // 测试基本平方运算
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});

    auto result = square(x);

    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {1.0, 4.0, 9.0, 16.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(SquareOperatorTest, ForwardZeroTensor)
{
    // 测试零张量
    auto x = Tensor({0.0, 0.0}, Shape{2});

    auto result = square(x);

    auto result_data = result.to_vector<float>();
    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], 0.0, kTolerance);
    }
}

TEST_F(SquareOperatorTest, ForwardNegativeValues)
{
    // 测试负值
    auto x = Tensor({-1.0, -2.0, -3.0}, Shape{3});

    auto result = square(x);

    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {1.0, 4.0, 9.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(SquareOperatorTest, ForwardMixedSigns)
{
    // 测试混合符号
    auto x = Tensor({-2.0, 0.0, 2.0}, Shape{3});

    auto result = square(x);

    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {4.0, 0.0, 4.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

// ==================== 反向传播测试 ====================

TEST_F(SquareOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x = Tensor({2.0, 3.0}, Shape{2});

    auto y = square(x);
    y.backward();

    // 平方算子的梯度：∂y/∂x = 2x
    auto gx_data = x.grad().to_vector<float>();
    auto x_data  = x.to_vector<float>();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], 2.0 * x_data[i], kTolerance);
    }
}

TEST_F(SquareOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x = Tensor({2.0, 3.0}, Shape{2});

    auto y = square(x);
    y.backward();

    // 设置输出梯度为2
    auto gy = Tensor({2.0, 2.0}, Shape{2});
    y.backward();

    auto gx_data = x.grad().to_vector<float>();
    auto x_data  = x.to_vector<float>();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], 2.0 * 2.0 * x_data[i], kTolerance);  // gy * 2x = 2 * 2x
    }
}

TEST_F(SquareOperatorTest, BackwardZeroGradient)
{
    // 测试零点的梯度
    auto x = Tensor({0.0, 0.0}, Shape{2});

    auto y = square(x);
    y.backward();

    auto gx_data = x.grad().to_vector<float>();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], 0.0, kTolerance);
    }
}

TEST_F(SquareOperatorTest, BackwardNegativeValues)
{
    // 测试负值的梯度
    auto x = Tensor({-2.0, -3.0}, Shape{2});

    auto y = square(x);
    y.backward();

    auto gx_data = x.grad().to_vector<float>();
    auto x_data  = x.to_vector<float>();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], 2.0 * x_data[i], kTolerance);
    }
}

// ==================== 边界情况测试 ====================

TEST_F(SquareOperatorTest, SingleElement)
{
    // 测试单元素张量
    auto x = Tensor({5.0}, Shape{1});

    auto result = square(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 25.0, kTolerance);
}

TEST_F(SquareOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<data_t> data(100, 3.0);
    auto x = Tensor(data, Shape{10, 10});

    auto result = square(x);

    Shape expected_shape{10, 10};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector<float>();

    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], 9.0, kTolerance);
    }
}

TEST_F(SquareOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    auto x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, Shape{2, 2, 2});

    auto result = square(x);

    Shape expected_shape{2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector<float>();

    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], (i + 1.0) * (i + 1.0), kTolerance);
    }
}

// ==================== 数值稳定性测试 ====================

TEST_F(SquareOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x = Tensor({1e5, kTolerance}, Shape{2});

    auto result = square(x);

    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {1e10, 1e-10};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(SquareOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x = Tensor({0.1, 0.2}, Shape{2});

    auto result = square(x);

    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {0.01, 0.04};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

// ==================== 特殊值测试 ====================

TEST_F(SquareOperatorTest, SmallValues)
{
    // 测试小值
    auto x = Tensor({kTolerance, 2e-6}, Shape{2});

    auto result = square(x);

    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {kTolerance, 4e-12};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(SquareOperatorTest, LargeValues)
{
    // 测试大值
    auto x = Tensor({1e6, 2e6}, Shape{2});

    auto result = square(x);

    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {1e12, 4e12};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(SquareOperatorTest, IdentityProperty)
{
    // 测试恒等性质：square(square(x)) = x^4
    auto x = Tensor({2.0, 3.0}, Shape{2});

    auto y1 = square(x);
    auto y2 = square(y1);

    auto y2_data = y2.to_vector<float>();
    auto x_data  = x.to_vector<float>();

    for (size_t i = 0; i < y2_data.size(); ++i)
    {
        EXPECT_NEAR(y2_data[i], x_data[i] * x_data[i] * x_data[i] * x_data[i], kTolerance);
    }
}
