#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "origin.h"

using namespace origin;

class PowOperatorTest : public ::testing::Test
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

TEST_F(PowOperatorTest, ForwardBasic)
{
    // 测试基本幂运算
    auto x       = Tensor({2.0, 3.0}, Shape{2});
    int exponent = 2;

    auto result = pow(x, exponent);

    Shape expected_shape{2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {4.0, 9.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(PowOperatorTest, ForwardOperatorOverload)
{
    // 测试运算符重载
    auto x       = Tensor({2.0, 3.0}, Shape{2});
    int exponent = 3;

    auto result = x ^ exponent;

    Shape expected_shape{2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {8.0, 27.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(PowOperatorTest, ForwardZeroExponent)
{
    // 测试零指数
    auto x       = Tensor({2.0, 3.0}, Shape{2});
    int exponent = 0;

    auto result = pow(x, exponent);

    auto result_data = result.to_vector<float>();
    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], 1.0, kTolerance);
    }
}

TEST_F(PowOperatorTest, ForwardNegativeExponent)
{
    // 测试负指数
    auto x       = Tensor({2.0, 4.0}, Shape{2});
    int exponent = -1;

    auto result = pow(x, exponent);

    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {0.5, 0.25};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(PowOperatorTest, ForwardZeroBase)
{
    // 测试零底数
    auto x       = Tensor({0.0, 0.0}, Shape{2});
    int exponent = 2;

    auto result = pow(x, exponent);

    auto result_data = result.to_vector<float>();
    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], 0.0, kTolerance);
    }
}

// ==================== 反向传播测试 ====================

TEST_F(PowOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x       = Tensor({2.0, 3.0}, Shape{2});
    int exponent = 2;

    auto y = pow(x, exponent);
    y.backward();

    // 幂算子的梯度：∂y/∂x = exponent * x^(exponent-1)
    auto gx_data = x.grad().to_vector<float>();
    auto x_data  = x.to_vector<float>();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], exponent * std::pow(x_data[i], exponent - 1), kTolerance);
    }
}

TEST_F(PowOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x       = Tensor({2.0, 3.0}, Shape{2});
    int exponent = 3;

    auto y = pow(x, exponent);
    y.backward();

    // 设置输出梯度为2
    auto gy = Tensor({2.0, 2.0}, Shape{2});
    y.backward();

    auto gx_data = x.grad().to_vector<float>();
    auto x_data  = x.to_vector<float>();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], 2.0 * exponent * std::pow(x_data[i], exponent - 1), kTolerance);
    }
}

TEST_F(PowOperatorTest, BackwardZeroExponent)
{
    // 测试零指数的梯度
    auto x       = Tensor({2.0, 3.0}, Shape{2});
    int exponent = 0;

    auto y = pow(x, exponent);
    y.backward();

    auto gx_data = x.grad().to_vector<float>();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], 0.0, kTolerance);
    }
}

TEST_F(PowOperatorTest, BackwardNegativeExponent)
{
    // 测试负指数的梯度
    auto x       = Tensor({2.0, 3.0}, Shape{2});
    int exponent = -2;

    auto y = pow(x, exponent);
    y.backward();

    auto gx_data = x.grad().to_vector<float>();
    auto x_data  = x.to_vector<float>();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], exponent * std::pow(x_data[i], exponent - 1), kTolerance);
    }
}

// ==================== 边界情况测试 ====================

TEST_F(PowOperatorTest, SingleElement)
{
    // 测试单元素张量
    auto x       = Tensor({2.0}, Shape{1});
    int exponent = 3;

    auto result = pow(x, exponent);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 8.0, kTolerance);
}

TEST_F(PowOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<data_t> data(100, 2.0);
    auto x       = Tensor(data, Shape{10, 10});
    int exponent = 2;

    auto result = pow(x, exponent);

    Shape expected_shape{10, 10};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector<float>();

    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], 4.0, kTolerance);
    }
}

TEST_F(PowOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    auto x       = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, Shape{2, 2, 2});
    int exponent = 2;

    auto result = pow(x, exponent);

    Shape expected_shape{2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector<float>();

    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], (i + 1.0) * (i + 1.0), kTolerance);
    }
}

// ==================== 数值稳定性测试 ====================

TEST_F(PowOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x       = Tensor({kTolerance, 1e5}, Shape{2});
    int exponent = 2;

    auto result = pow(x, exponent);

    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {1e-10, 1e10};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(PowOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x       = Tensor({0.1, 0.2}, Shape{2});
    int exponent = 3;

    auto result = pow(x, exponent);

    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {0.001, 0.008};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

// ==================== 特殊值测试 ====================

TEST_F(PowOperatorTest, DifferentExponents)
{
    // 测试不同指数
    auto x = Tensor({2.0}, Shape{1});

    // 测试指数为1
    auto result1 = pow(x, 1);
    EXPECT_NEAR(result1.item<float>(), 2.0, kTolerance);

    // 测试指数为2
    auto result2 = pow(x, 2);
    EXPECT_NEAR(result2.item<float>(), 4.0, kTolerance);

    // 测试指数为3
    auto result3 = pow(x, 3);
    EXPECT_NEAR(result3.item<float>(), 8.0, kTolerance);
}

TEST_F(PowOperatorTest, IdentityProperty)
{
    // 测试恒等性质：x^1 = x
    auto x = Tensor({2.0, 3.0, 4.0}, Shape{3});

    auto result = pow(x, 1);

    EXPECT_TRUE(tensorsEqual(result, x));
}

TEST_F(PowOperatorTest, ZeroPowerProperty)
{
    // 测试零幂性质：x^0 = 1
    auto x = Tensor({2.0, 3.0, 4.0}, Shape{3});

    auto result = pow(x, 0);

    auto result_data = result.to_vector<float>();
    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], 1.0, kTolerance);
    }
}

TEST_F(PowOperatorTest, LargeExponent)
{
    // 测试大指数
    auto x       = Tensor({1.1}, Shape{1});
    int exponent = 10;

    auto result = pow(x, exponent);

    EXPECT_NEAR(result.item<float>(), std::pow(1.1, 10), kTolerance);
}
