#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "origin.h"

using namespace origin;

class ReshapeOperatorTest : public ::testing::Test
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

TEST_F(ReshapeOperatorTest, ForwardBasic)
{
    // ===== 测试reshape操作的基本功能 =====

    // 测试基本重塑运算
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
    Shape target_shape{4, 1};

    auto result = reshape(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {1.0, 2.0, 3.0, 4.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(ReshapeOperatorTest, ForwardToSameShape)
{
    // 测试相同形状（应该不变）
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
    Shape target_shape{2, 2};

    auto result = reshape(x, target_shape);

    EXPECT_TRUE(tensorsEqual(result, x));
}

TEST_F(ReshapeOperatorTest, ForwardTo1D)
{
    // 测试重塑为一维
    auto x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape{2, 3});
    Shape target_shape{6};

    auto result = reshape(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(ReshapeOperatorTest, ForwardTo2D)
{
    // 测试重塑为二维
    auto x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape{6});
    Shape target_shape{2, 3};

    auto result = reshape(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(ReshapeOperatorTest, ForwardTo3D)
{
    // 测试重塑为三维
    auto x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, Shape{8});
    Shape target_shape{2, 2, 2};

    auto result = reshape(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(ReshapeOperatorTest, ForwardZeroTensor)
{
    // 测试零张量
    auto x = Tensor({0.0, 0.0, 0.0, 0.0}, Shape{2, 2});
    Shape target_shape{4};

    auto result = reshape(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data = result.to_vector<float>();

    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], 0.0, kTolerance);
    }
}

// ==================== 反向传播测试 ====================

TEST_F(ReshapeOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
    Shape target_shape{4, 1};

    auto y = reshape(x, target_shape);
    y.backward();

    // 重塑算子的梯度：∂y/∂x = reshape(gy, x.shape())
    auto gx_data = x.grad().to_vector<float>();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], 1.0, kTolerance);
    }
}

TEST_F(ReshapeOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
    Shape target_shape{4, 1};

    auto y = reshape(x, target_shape);
    y.backward();

    // 设置输出梯度为2
    auto gy = Tensor({2.0, 2.0, 2.0, 2.0}, Shape{4, 1});
    y.backward();

    auto gx_data = x.grad().to_vector<float>();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], 2.0, kTolerance);
    }
}

TEST_F(ReshapeOperatorTest, BackwardToSameShape)
{
    // 测试相同形状的反向传播
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
    Shape target_shape{2, 2};

    auto y = reshape(x, target_shape);
    y.backward();

    auto gx_data = x.grad().to_vector<float>();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], 1.0, kTolerance);
    }
}

TEST_F(ReshapeOperatorTest, BackwardToDifferentShape)
{
    // 测试不同形状的反向传播
    auto x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape{2, 3});
    Shape target_shape{3, 2};

    auto y = reshape(x, target_shape);
    y.backward();

    auto gx_data = x.grad().to_vector<float>();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], 1.0, kTolerance);
    }
}

// ==================== 边界情况测试 ====================

TEST_F(ReshapeOperatorTest, SingleElement)
{
    // 测试单元素张量
    auto x = Tensor({5.0}, Shape{1});
    Shape target_shape{1, 1};

    auto result = reshape(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    EXPECT_NEAR(result.item<float>(), 5.0, kTolerance);
}

TEST_F(ReshapeOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<data_t> data(100, 1.0);
    auto x = Tensor(data, Shape{10, 10});
    Shape target_shape{100};

    auto result = reshape(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data = result.to_vector<float>();

    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], 1.0, kTolerance);
    }
}

TEST_F(ReshapeOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    auto x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, Shape{2, 2, 2});
    Shape target_shape{4, 2};

    auto result = reshape(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

// ==================== 数值稳定性测试 ====================

TEST_F(ReshapeOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x = Tensor({1e10, 1e-10, 1e10, 1e-10}, Shape{2, 2});
    Shape target_shape{4};

    auto result = reshape(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {1e10, 1e-10, 1e10, 1e-10};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(ReshapeOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x = Tensor({0.1, 0.2, 0.3, 0.4}, Shape{2, 2});
    Shape target_shape{4};

    auto result = reshape(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {0.1, 0.2, 0.3, 0.4};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

// ==================== 特殊值测试 ====================

TEST_F(ReshapeOperatorTest, MixedSigns)
{
    // 测试混合符号
    auto x = Tensor({1.0, -2.0, 3.0, -4.0}, Shape{2, 2});
    Shape target_shape{4};

    auto result = reshape(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {1.0, -2.0, 3.0, -4.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(ReshapeOperatorTest, IdentityProperty)
{
    // 测试恒等性质：reshape(x, x.shape()) = x
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});

    auto result = reshape(x, x.shape());

    EXPECT_TRUE(tensorsEqual(result, x));
}

TEST_F(ReshapeOperatorTest, AssociativeProperty)
{
    // 测试结合性质：reshape(reshape(x, shape1), shape2) = reshape(x, shape2)
    auto x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape{2, 3});
    Shape shape1{3, 2};
    Shape shape2{6};

    auto result1 = reshape(reshape(x, shape1), shape2);
    auto result2 = reshape(x, shape2);

    EXPECT_TRUE(tensorsEqual(result1, result2));
}

TEST_F(ReshapeOperatorTest, CommutativeProperty)
{
    // 测试交换性质：reshape(x + y, shape) = reshape(x, shape) + reshape(y, shape)
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
    auto y = Tensor({5.0, 6.0, 7.0, 8.0}, Shape{2, 2});
    Shape target_shape{4};

    auto result1 = reshape(x + y, target_shape);
    auto result2 = reshape(x, target_shape) + reshape(y, target_shape);

    EXPECT_TRUE(tensorsEqual(result1, result2));
}

TEST_F(ReshapeOperatorTest, ElementCountValidation)
{
    // 测试元素数量验证
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});

    // 有效重塑
    Shape valid_shape{4};
    EXPECT_NO_THROW(reshape(x, valid_shape));

    // 无效重塑（元素数量不匹配）
    Shape invalid_shape{5};
    EXPECT_THROW(reshape(x, invalid_shape), std::exception);
}
