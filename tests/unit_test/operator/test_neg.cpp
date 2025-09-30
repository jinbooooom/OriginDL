#include <arrayfire.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "dlOperator.h"
#include "dlTensor.h"

using namespace dl;

class NegOperatorTest : public ::testing::Test
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

TEST_F(NegOperatorTest, ForwardBasic)
{
    // 测试基本负号运算
    auto x = Tensor({1.0, -2.0, 3.0, -4.0}, Shape{2, 2});

    auto result = neg(x);

    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {-1.0, 2.0, -3.0, 4.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(NegOperatorTest, ForwardOperatorOverload)
{
    // 测试运算符重载
    auto x = Tensor({2.0, -3.0}, Shape{2});

    auto result = -x;

    Shape expected_shape{2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {-2.0, 3.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(NegOperatorTest, ForwardZeroTensor)
{
    // 测试零张量
    auto x = Tensor({0.0, 0.0}, Shape{2});

    auto result = neg(x);

    EXPECT_TRUE(tensorsEqual(result, x));
}

TEST_F(NegOperatorTest, ForwardPositiveValues)
{
    // 测试正值
    auto x = Tensor({1.0, 2.0, 3.0}, Shape{3});

    auto result = neg(x);

    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {-1.0, -2.0, -3.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(NegOperatorTest, ForwardNegativeValues)
{
    // 测试负值
    auto x = Tensor({-1.0, -2.0, -3.0}, Shape{3});

    auto result = neg(x);

    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {1.0, 2.0, 3.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

// ==================== 反向传播测试 ====================

TEST_F(NegOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x = Tensor({1.0, 2.0}, Shape{2});

    auto y = neg(x);
    y.backward();

    // 负号算子的梯度：∂y/∂x = -1
    auto gx_data = x.grad().to_vector();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], -1.0, kTolerance);
    }
}

TEST_F(NegOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x = Tensor({2.0, 3.0}, Shape{2});

    auto y = neg(x);
    y.backward();

    // 设置输出梯度为2
    auto gy = Tensor({2.0, 2.0}, Shape{2});
    y.backward();

    auto gx_data = x.grad().to_vector();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], -2.0, kTolerance);  // gy * (-1) = 2 * (-1) = -2
    }
}

TEST_F(NegOperatorTest, BackwardDifferentShapes)
{
    // 测试不同形状的张量
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});

    auto y = neg(x);
    y.backward();

    auto gx_data = x.grad().to_vector();

    EXPECT_EQ(gx_data.size(), 4);

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], -1.0, kTolerance);
    }
}

// ==================== 边界情况测试 ====================

TEST_F(NegOperatorTest, SingleElement)
{
    // 测试单元素张量
    auto x = Tensor({5.0}, Shape{1});

    auto result = neg(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item(), -5.0, kTolerance);
}

TEST_F(NegOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<data_t> data(100, 2.0);
    auto x = Tensor(data, Shape{10, 10});

    auto result = neg(x);

    Shape expected_shape{10, 10};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector();

    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], -2.0, kTolerance);
    }
}

TEST_F(NegOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    auto x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, Shape{2, 2, 2});

    auto result = neg(x);

    Shape expected_shape{2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector();

    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], -(i + 1.0), kTolerance);
    }
}

// ==================== 数值稳定性测试 ====================

TEST_F(NegOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x = Tensor({1e10, 1e-10}, Shape{2});

    auto result = neg(x);

    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {-1e10, -1e-10};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(NegOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x = Tensor({0.1, 0.2}, Shape{2});

    auto result = neg(x);

    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {-0.1, -0.2};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

// ==================== 特殊值测试 ====================

TEST_F(NegOperatorTest, DoubleNegation)
{
    // 测试双重负号
    auto x = Tensor({1.0, 2.0, 3.0}, Shape{3});

    auto result = neg(neg(x));

    EXPECT_TRUE(tensorsEqual(result, x));
}

TEST_F(NegOperatorTest, MixedSigns)
{
    // 测试混合符号
    auto x = Tensor({1.0, -2.0, 0.0, -4.0, 5.0}, Shape{5});

    auto result = neg(x);

    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {-1.0, 2.0, 0.0, 4.0, -5.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}
