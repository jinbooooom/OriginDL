#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "origin.h"

using namespace origin;

class MulOperatorTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
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
    bool isEqual(double a, double b, double tolerance = 1e-6) { return std::abs(a - b) < tolerance; }

    // 辅助函数：比较两个Tensor是否相等
    bool tensorsEqual(const Tensor &a, const Tensor &b, double tolerance = 1e-6)
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

TEST_F(MulOperatorTest, ForwardBasic)
{
    // 测试基本乘法运算
    auto x0 = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
    auto x1 = Tensor({2.0, 3.0, 4.0, 5.0}, Shape{2, 2});

    auto result = mul(x0, x1);

    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {2.0, 6.0, 12.0, 20.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], 1e-6);
    }
}

TEST_F(MulOperatorTest, ForwardOperatorOverload)
{
    // 测试运算符重载
    auto x0 = Tensor({2.0, 3.0}, Shape{2});
    auto x1 = Tensor({4.0, 5.0}, Shape{2});

    auto result = x0 * x1;

    EXPECT_EQ(result.shape(), Shape{2});
    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {8.0, 15.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], 1e-6);
    }
}

TEST_F(MulOperatorTest, ForwardScalarTensor)
{
    // 测试标量与张量的乘法
    auto x        = Tensor({1.0, 2.0, 3.0}, Shape{3});
    data_t scalar = 2.0;

    auto result1 = x * scalar;
    auto result2 = scalar * x;

    EXPECT_EQ(result1.shape(), Shape{3});
    EXPECT_EQ(result2.shape(), Shape{3});

    auto data1                   = result1.to_vector();
    auto data2                   = result2.to_vector();
    std::vector<data_t> expected = {2.0, 4.0, 6.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(data1[i], expected[i], 1e-6);
        EXPECT_NEAR(data2[i], expected[i], 1e-6);
    }
}

TEST_F(MulOperatorTest, ForwardZeroTensor)
{
    // 测试零张量乘法
    auto x0 = Tensor({1.0, 2.0}, Shape{2});
    auto x1 = Tensor({0.0, 0.0}, Shape{2});

    auto result = mul(x0, x1);

    auto result_data = result.to_vector();
    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], 0.0, 1e-6);
    }
}

TEST_F(MulOperatorTest, ForwardNegativeValues)
{
    // 测试负值乘法
    auto x0 = Tensor({-1.0, -2.0}, Shape{2});
    auto x1 = Tensor({3.0, 4.0}, Shape{2});

    auto result = mul(x0, x1);

    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {-3.0, -8.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], 1e-6);
    }
}

// ==================== 反向传播测试 ====================

TEST_F(MulOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x0 = Tensor({2.0, 3.0}, Shape{2});
    auto x1 = Tensor({4.0, 5.0}, Shape{2});

    auto y = mul(x0, x1);
    y.backward();

    // 乘法算子的梯度：∂y/∂x0 = x1, ∂y/∂x1 = x0
    auto gx0_data = x0.grad().to_vector();
    auto gx1_data = x1.grad().to_vector();

    for (size_t i = 0; i < gx0_data.size(); ++i)
    {
        EXPECT_NEAR(gx0_data[i], x1.to_vector()[i], 1e-6);
        EXPECT_NEAR(gx1_data[i], x0.to_vector()[i], 1e-6);
    }
}

TEST_F(MulOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x0 = Tensor({2.0, 3.0}, Shape{2});
    auto x1 = Tensor({1.0, 1.0}, Shape{2});

    auto y = mul(x0, x1);
    // 乘法算子的梯度：∂y/∂x0 = x1, ∂y/∂x1 = x0
    y.backward();

    auto gx0_data = x0.grad().to_vector();
    auto gx1_data = x1.grad().to_vector();

    for (size_t i = 0; i < gx0_data.size(); ++i)
    {
        EXPECT_NEAR(gx0_data[i], 1.0, 1e-6);                // ∂y/∂x0 = x1 = 1
        EXPECT_NEAR(gx1_data[i], x0.to_vector()[i], 1e-6);  // ∂y/∂x1 = x0
    }
}

TEST_F(MulOperatorTest, BackwardDifferentShapes)
{
    // 测试不同形状的张量乘法反向传播
    auto x0 = Tensor({2.0, 3.0}, Shape{2});
    auto x1 = Tensor({4.0}, Shape{1});

    auto y = mul(x0, x1);
    y.backward();

    // 梯度应该正确广播
    auto gx0_data = x0.grad().to_vector();
    auto gx1_data = x1.grad().to_vector();

    EXPECT_EQ(gx0_data.size(), 2);
    EXPECT_EQ(gx1_data.size(), 1);

    for (size_t i = 0; i < gx0_data.size(); ++i)
    {
        EXPECT_NEAR(gx0_data[i], 4.0, 1e-6);  // 广播后的x1值
    }
    EXPECT_NEAR(gx1_data[0], 5.0, 1e-6);  // sum(x0) = 2 + 3
}

// ==================== 边界情况测试 ====================

TEST_F(MulOperatorTest, SingleElement)
{
    // 测试单元素张量
    auto x0 = Tensor({5.0}, Shape{1});
    auto x1 = Tensor({3.0}, Shape{1});

    auto result = mul(x0, x1);

    EXPECT_EQ(result.shape(), Shape{1});
    EXPECT_NEAR(result.item(), 15.0, 1e-6);
}

TEST_F(MulOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<data_t> data1(100, 2.0);
    std::vector<data_t> data2(100, 3.0);
    auto x0 = Tensor(data1, Shape{10, 10});
    auto x1 = Tensor(data2, Shape{10, 10});

    auto result = mul(x0, x1);

    Shape expected_shape{10, 10};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector();

    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], 6.0, 1e-6);
    }
}

TEST_F(MulOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    auto x0 = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, Shape{2, 2, 2});
    auto x1 = Tensor({0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5}, Shape{2, 2, 2});

    auto result = mul(x0, x1);

    Shape expected_shape{2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector();

    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], (i + 1) * 0.5, 1e-6);
    }
}

// ==================== 数值稳定性测试 ====================

TEST_F(MulOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x0 = Tensor({1e10, 1e-10}, Shape{2});
    auto x1 = Tensor({1e-10, 1e10}, Shape{2});

    auto result = mul(x0, x1);

    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {1.0, 1.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], 1e-6);
    }
}

TEST_F(MulOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x0 = Tensor({0.1, 0.2}, Shape{2});
    auto x1 = Tensor({0.3, 0.4}, Shape{2});

    auto result = mul(x0, x1);

    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {0.03, 0.08};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], 1e-6);
    }
}
