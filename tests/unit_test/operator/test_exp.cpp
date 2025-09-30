#include <arrayfire.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "dlOperator.h"
#include "dlTensor.h"

using namespace dl;

class ExpOperatorTest : public ::testing::Test
{
protected:
    // 精度忍受常量
    static constexpr double kTolerance = 1e-3;
    void SetUp() override
    {
        // 测试前的设置
        // 初始化ArrayFire后端
        try
        {
            af::setBackend(AF_BACKEND_CPU);
        }
        catch (const af::exception &e)
        {
            // 忽略错误，继续测试
        }
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

        auto data_a = a.to_vector();
        auto data_b = b.to_vector();

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

TEST_F(ExpOperatorTest, ForwardBasic)
{
    // 测试基本指数运算
    auto x = Tensor({0.0, 1.0, 2.0}, Shape{3});

    auto result = exp(x);

    Shape expected_shape{3};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {1.0, std::exp(1.0), std::exp(2.0)};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(ExpOperatorTest, ForwardZero)
{
    // 测试零值
    auto x = Tensor({0.0, 0.0}, Shape{2});

    auto result = exp(x);

    auto result_data = result.to_vector();
    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], 1.0, kTolerance);
    }
}

TEST_F(ExpOperatorTest, ForwardNegativeValues)
{
    // 测试负值
    auto x = Tensor({-1.0, -2.0}, Shape{2});

    auto result = exp(x);

    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {std::exp(-1.0), std::exp(-2.0)};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(ExpOperatorTest, ForwardLargeValues)
{
    // 测试大值
    auto x = Tensor({5.0, 10.0}, Shape{2});

    auto result = exp(x);

    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {std::exp(5.0), std::exp(10.0)};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

// ==================== 反向传播测试 ====================

TEST_F(ExpOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x = Tensor({1.0, 2.0}, Shape{2});

    auto y = exp(x);
    y.backward();

    // 指数算子的梯度：∂y/∂x = exp(x)
    auto gx_data = x.grad().to_vector();
    auto y_data  = y.to_vector();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], y_data[i], kTolerance);
    }
}

TEST_F(ExpOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x = Tensor({1.0, 2.0}, Shape{2});

    auto y = exp(x);
    y.backward();

    // 设置输出梯度为2
    auto gy = Tensor({2.0, 2.0}, Shape{2});
    y.backward();

    auto gx_data = x.grad().to_vector();
    auto y_data  = y.to_vector();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], 2.0 * y_data[i], kTolerance);  // gy * exp(x)
    }
}

TEST_F(ExpOperatorTest, BackwardZeroGradient)
{
    // 测试零点的梯度
    auto x = Tensor({0.0, 0.0}, Shape{2});

    auto y = exp(x);
    y.backward();

    auto gx_data = x.grad().to_vector();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], 1.0, kTolerance);  // exp(0) = 1
    }
}

TEST_F(ExpOperatorTest, BackwardNegativeValues)
{
    // 测试负值的梯度
    auto x = Tensor({-1.0, -2.0}, Shape{2});

    auto y = exp(x);
    y.backward();

    auto gx_data = x.grad().to_vector();
    auto y_data  = y.to_vector();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], y_data[i], kTolerance);
    }
}

// ==================== 边界情况测试 ====================

TEST_F(ExpOperatorTest, SingleElement)
{
    // 测试单元素张量
    auto x = Tensor({1.0}, Shape{1});

    auto result = exp(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item(), std::exp(1.0), kTolerance);
}

TEST_F(ExpOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<data_t> data(100, 1.0);
    auto x = Tensor(data, Shape{10, 10});

    auto result = exp(x);

    Shape expected_shape{10, 10};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector();
    data_t expected  = std::exp(1.0);

    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected, kTolerance);
    }
}

TEST_F(ExpOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    auto x = Tensor({0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0}, Shape{2, 2, 2});

    auto result = exp(x);

    Shape expected_shape{2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector();

    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], std::exp(i), kTolerance);
    }
}

// ==================== 数值稳定性测试 ====================

TEST_F(ExpOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x = Tensor({-10.0, 0.0, 10.0}, Shape{3});

    auto result = exp(x);

    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {std::exp(-10.0), 1.0, std::exp(10.0)};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(ExpOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x = Tensor({0.1, 0.2}, Shape{2});

    auto result = exp(x);

    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {std::exp(0.1), std::exp(0.2)};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

// ==================== 特殊值测试 ====================

TEST_F(ExpOperatorTest, SmallValues)
{
    // 测试小值
    auto x = Tensor({kTolerance, 2e-6}, Shape{2});

    auto result = exp(x);

    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {std::exp(kTolerance), std::exp(2e-6)};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(ExpOperatorTest, VerySmallValues)
{
    // 测试非常小的值
    auto x = Tensor({-50.0, -100.0}, Shape{2});

    auto result = exp(x);

    auto result_data = result.to_vector();
    // 这些值应该非常接近0
    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_LT(result_data[i], 1e-20);
    }
}

TEST_F(ExpOperatorTest, IdentityProperty)
{
    // 测试恒等性质：exp(0) = 1
    auto x = Tensor({0.0}, Shape{1});

    auto result = exp(x);

    EXPECT_NEAR(result.item(), 1.0, kTolerance);
}

TEST_F(ExpOperatorTest, MonotonicProperty)
{
    // 测试单调性：如果 x1 < x2，则 exp(x1) < exp(x2)
    auto x1 = Tensor({1.0}, Shape{1});
    auto x2 = Tensor({2.0}, Shape{1});

    auto y1 = exp(x1);
    auto y2 = exp(x2);

    EXPECT_LT(y1.item(), y2.item());
}
