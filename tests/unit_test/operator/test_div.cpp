#include <arrayfire.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "dlOperator.h"
#include "dlTensor.h"

using namespace dl;

class DivOperatorTest : public ::testing::Test
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

TEST_F(DivOperatorTest, ForwardBasic)
{
    // 测试基本除法运算
    auto x0 = Tensor({6.0, 8.0, 10.0, 12.0}, Shape{2, 2});
    auto x1 = Tensor({2.0, 4.0, 5.0, 6.0}, Shape{2, 2});

    auto result = div(x0, x1);

    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {3.0, 2.0, 2.0, 2.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(DivOperatorTest, ForwardOperatorOverload)
{
    // 测试运算符重载
    auto x0 = Tensor({6.0, 8.0}, Shape{2});
    auto x1 = Tensor({2.0, 4.0}, Shape{2});

    auto result = x0 / x1;

    Shape expected_shape{2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {3.0, 2.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(DivOperatorTest, ForwardScalarTensor)
{
    // 测试标量与张量的除法
    auto x        = Tensor({6.0, 8.0, 10.0}, Shape{3});
    data_t scalar = 2.0;

    auto result1 = x / scalar;
    auto result2 = scalar / x;

    EXPECT_EQ(result1.shape(), Shape{3});
    EXPECT_EQ(result2.shape(), Shape{3});

    auto data1                    = result1.to_vector();
    auto data2                    = result2.to_vector();
    std::vector<data_t> expected1 = {3.0, 4.0, 5.0};
    std::vector<data_t> expected2 = {1.0 / 3.0, 0.25, 0.2};

    for (size_t i = 0; i < expected1.size(); ++i)
    {
        EXPECT_NEAR(data1[i], expected1[i], kTolerance);
        EXPECT_NEAR(data2[i], expected2[i], kTolerance);
    }
}

TEST_F(DivOperatorTest, ForwardOneTensor)
{
    // 测试除以1的张量
    auto x0 = Tensor({1.0, 2.0}, Shape{2});
    auto x1 = Tensor({1.0, 1.0}, Shape{2});

    auto result = div(x0, x1);

    EXPECT_TRUE(tensorsEqual(result, x0));
}

TEST_F(DivOperatorTest, ForwardNegativeValues)
{
    // 测试负值除法
    auto x0 = Tensor({-6.0, -8.0}, Shape{2});
    auto x1 = Tensor({2.0, 4.0}, Shape{2});

    auto result = div(x0, x1);

    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {-3.0, -2.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

// ==================== 反向传播测试 ====================

TEST_F(DivOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x0 = Tensor({6.0, 8.0}, Shape{2});
    auto x1 = Tensor({2.0, 4.0}, Shape{2});

    auto y = div(x0, x1);
    y.backward();

    // 除法算子的梯度：
    // ∂y/∂x0 = 1/x1, ∂y/∂x1 = -x0/x1²
    auto gx0_data = x0.grad().to_vector();
    auto gx1_data = x1.grad().to_vector();

    auto x0_data = x0.to_vector();
    auto x1_data = x1.to_vector();

    for (size_t i = 0; i < gx0_data.size(); ++i)
    {
        EXPECT_NEAR(gx0_data[i], 1.0 / x1_data[i], kTolerance);
        EXPECT_NEAR(gx1_data[i], -x0_data[i] / (x1_data[i] * x1_data[i]), kTolerance);
    }
}

TEST_F(DivOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x0 = Tensor({4.0, 6.0}, Shape{2});
    auto x1 = Tensor({2.0, 3.0}, Shape{2});

    auto y = div(x0, x1);
    y.backward();

    // 设置输出梯度为2
    auto gy = Tensor({2.0, 2.0}, Shape{2});
    y.backward();

    auto gx0_data = x0.grad().to_vector();
    auto gx1_data = x1.grad().to_vector();

    auto x0_data = x0.to_vector();
    auto x1_data = x1.to_vector();

    for (size_t i = 0; i < gx0_data.size(); ++i)
    {
        EXPECT_NEAR(gx0_data[i], 2.0 / x1_data[i], kTolerance);                               // gy * (1/x1)
        EXPECT_NEAR(gx1_data[i], -2.0 * x0_data[i] / (x1_data[i] * x1_data[i]), kTolerance);  // gy * (-x0/x1²)
    }
}

TEST_F(DivOperatorTest, BackwardDifferentShapes)
{
    // 测试不同形状的张量除法反向传播
    auto x0 = Tensor({6.0, 8.0}, Shape{2});
    auto x1 = Tensor({2.0}, Shape{1});

    auto y = div(x0, x1);
    y.backward();

    // 梯度应该正确广播
    auto gx0_data = x0.grad().to_vector();
    auto gx1_data = x1.grad().to_vector();

    EXPECT_EQ(gx0_data.size(), 2);
    EXPECT_EQ(gx1_data.size(), 1);

    for (size_t i = 0; i < gx0_data.size(); ++i)
    {
        EXPECT_NEAR(gx0_data[i], 0.5, kTolerance);  // 1/2
    }
    EXPECT_NEAR(gx1_data[0], -3.5, kTolerance);  // -(6+8)/4 = -14/4 = -3.5
}

// ==================== 边界情况测试 ====================

TEST_F(DivOperatorTest, SingleElement)
{
    // 测试单元素张量
    auto x0 = Tensor({15.0}, Shape{1});
    auto x1 = Tensor({3.0}, Shape{1});

    auto result = div(x0, x1);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item(), 5.0, kTolerance);
}

TEST_F(DivOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<data_t> data1(100, 6.0);
    std::vector<data_t> data2(100, 2.0);
    auto x0 = Tensor(data1, Shape{10, 10});
    auto x1 = Tensor(data2, Shape{10, 10});

    auto result = div(x0, x1);

    Shape expected_shape{10, 10};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector();

    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], 3.0, kTolerance);
    }
}

TEST_F(DivOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    auto x0 = Tensor({2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0}, Shape{2, 2, 2});
    auto x1 = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, Shape{2, 2, 2});

    auto result = div(x0, x1);

    Shape expected_shape{2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector();

    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], 2.0, kTolerance);
    }
}

// ==================== 数值稳定性测试 ====================

TEST_F(DivOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x0 = Tensor({1e10, 1e-10}, Shape{2});
    auto x1 = Tensor({1e-10, 1e10}, Shape{2});

    auto result = div(x0, x1);

    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {1e20, 1e-20};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(DivOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x0 = Tensor({0.1, 0.2}, Shape{2});
    auto x1 = Tensor({0.2, 0.4}, Shape{2});

    auto result = div(x0, x1);

    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {0.5, 0.5};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}
