#include <arrayfire.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "origin.h"

using namespace origin;

class AddOperatorTest : public ::testing::Test
{
protected:
    // 精度忍受常量
    static constexpr double kTolerance = 1e-3;
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

TEST_F(AddOperatorTest, ForwardBasic)
{
    // 测试基本加法运算
    Shape shape{2, 2};
    auto x0 = Tensor({1.0, 2.0, 3.0, 4.0}, shape);
    auto x1 = Tensor({5.0, 6.0, 7.0, 8.0}, shape);

    auto result = add(x0, x1);

    EXPECT_EQ(result.shape(), shape);
    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {6.0, 8.0, 10.0, 12.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(AddOperatorTest, ForwardOperatorOverload)
{
    // 测试运算符重载
    Shape shape{2};
    auto x0 = Tensor({1.0, 2.0}, shape);
    auto x1 = Tensor({3.0, 4.0}, shape);

    auto result = x0 + x1;

    EXPECT_EQ(result.shape(), shape);
    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {4.0, 6.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(AddOperatorTest, ForwardScalarTensor)
{
    // 测试标量与张量的加法
    auto x        = Tensor({1.0, 2.0, 3.0}, Shape{3});
    data_t scalar = 5.0;

    auto result1 = x + scalar;
    auto result2 = scalar + x;

    EXPECT_EQ(result1.shape(), Shape{3});
    EXPECT_EQ(result2.shape(), Shape{3});

    auto data1                   = result1.to_vector();
    auto data2                   = result2.to_vector();
    std::vector<data_t> expected = {6.0, 7.0, 8.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(data1[i], expected[i], kTolerance);
        EXPECT_NEAR(data2[i], expected[i], kTolerance);
    }
}

TEST_F(AddOperatorTest, ForwardZeroTensor)
{
    // 测试零张量加法
    auto x0 = Tensor({1.0, 2.0}, Shape{2});
    auto x1 = Tensor({0.0, 0.0}, Shape{2});

    auto result = add(x0, x1);

    EXPECT_TRUE(tensorsEqual(result, x0));
}

TEST_F(AddOperatorTest, ForwardNegativeValues)
{
    // 测试负值加法
    auto x0 = Tensor({-1.0, -2.0}, Shape{2});
    auto x1 = Tensor({3.0, 4.0}, Shape{2});

    auto result = add(x0, x1);

    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {2.0, 2.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

// ==================== 反向传播测试 ====================

TEST_F(AddOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x0 = Tensor({1.0, 2.0}, Shape{2});
    auto x1 = Tensor({3.0, 4.0}, Shape{2});

    auto y = add(x0, x1);
    y.backward();

    // 加法算子的梯度应该都是1
    auto gx0_data = x0.grad().to_vector();
    auto gx1_data = x1.grad().to_vector();

    for (size_t i = 0; i < gx0_data.size(); ++i)
    {
        EXPECT_NEAR(gx0_data[i], 1.0, kTolerance);
        EXPECT_NEAR(gx1_data[i], 1.0, kTolerance);
    }
}

TEST_F(AddOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x0 = Tensor({2.0, 3.0}, Shape{2});
    auto x1 = Tensor({1.0, 1.0}, Shape{2});

    auto y = add(x0, x1);
    y.backward();

    // 加法算子的梯度：∂y/∂x0 = 1, ∂y/∂x1 = 1
    auto gx0_data = x0.grad().to_vector();
    auto gx1_data = x1.grad().to_vector();

    for (size_t i = 0; i < gx0_data.size(); ++i)
    {
        EXPECT_NEAR(gx0_data[i], 1.0, kTolerance);
        EXPECT_NEAR(gx1_data[i], 1.0, kTolerance);
    }
}

TEST_F(AddOperatorTest, BackwardDifferentShapes)
{
    // 测试不同形状的张量加法反向传播
    auto x0 = Tensor({1.0, 2.0}, Shape{2});
    auto x1 = Tensor({3.0}, Shape{1});

    auto y = add(x0, x1);
    y.backward();

    // 梯度应该正确广播
    auto gx0_data = x0.grad().to_vector();
    auto gx1_data = x1.grad().to_vector();

    EXPECT_EQ(gx0_data.size(), 2U);
    EXPECT_EQ(gx1_data.size(), 1U);

    for (size_t i = 0; i < gx0_data.size(); ++i)
    {
        EXPECT_NEAR(gx0_data[i], 1.0, kTolerance);
    }
    EXPECT_NEAR(gx1_data[0], 2.0, kTolerance);  // 广播后的梯度
}

// ==================== 边界情况测试 ====================

TEST_F(AddOperatorTest, SingleElement)
{
    // 测试单元素张量
    auto x0 = Tensor({5.0}, Shape{1});
    auto x1 = Tensor({3.0}, Shape{1});

    auto result = add(x0, x1);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item(), 8.0, kTolerance);
}

TEST_F(AddOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<data_t> data1(100, 1.0);
    std::vector<data_t> data2(100, 2.0);
    auto x0 = Tensor(data1, Shape{10, 10});
    auto x1 = Tensor(data2, Shape{10, 10});

    auto result = add(x0, x1);

    Shape expected_shape{10, 10};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector();

    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], 3.0, kTolerance);
    }
}

TEST_F(AddOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    auto x0 = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, Shape{2, 2, 2});
    auto x1 = Tensor({0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, Shape{2, 2, 2});

    auto result = add(x0, x1);

    Shape expected_shape{2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector();

    // 期望值：x0[i] + x1[i]
    std::vector<double> expected = {1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8};

    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

// ==================== 数值稳定性测试 ====================

TEST_F(AddOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x0 = Tensor({1e-10, 1e10}, Shape{2});
    auto x1 = Tensor({1e-10, 1e10}, Shape{2});

    auto result = add(x0, x1);

    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {2e-10, 2e10};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(AddOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x0 = Tensor({0.1, 0.2}, Shape{2});
    auto x1 = Tensor({0.3, 0.4}, Shape{2});

    auto result = add(x0, x1);

    auto result_data             = result.to_vector();
    std::vector<data_t> expected = {0.4, 0.6};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}
