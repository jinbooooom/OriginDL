#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "origin.h"

using namespace origin;

class BroadcastToOperatorTest : public ::testing::Test
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

TEST_F(BroadcastToOperatorTest, ForwardBasic)
{
    // 测试基本广播运算（匹配libtorch的行主序行为）
    auto x = Tensor({1.0, 2.0}, Shape{2});
    Shape target_shape{2, 2};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data = result.to_vector<float>();
    // libtorch行主序：[1,2] expand到[2,2] = [[1,2],[1,2]] = [1,2,1,2]
    std::vector<data_t> expected = {1.0, 2.0, 1.0, 2.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(BroadcastToOperatorTest, ForwardScalar)
{
    // 测试标量广播
    auto x = Tensor({5.0}, Shape{1});
    Shape target_shape{3};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {5.0, 5.0, 5.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(BroadcastToOperatorTest, ForwardToSameShape)
{
    // 测试相同形状（应该不变）
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
    Shape target_shape{2, 2};

    auto result = broadcast_to(x, target_shape);

    EXPECT_TRUE(tensorsEqual(result, x));
}

TEST_F(BroadcastToOperatorTest, ForwardToLargerShape)
{
    // ===== 测试broadcast_to操作的基本功能 =====

    // 测试到更大形状
    auto x = Tensor({1.0, 2.0}, Shape{1, 2});
    Shape target_shape{3, 2};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {1.0, 2.0, 1.0, 2.0, 1.0, 2.0};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(BroadcastToOperatorTest, ForwardTo3D)
{
    // 测试到三维形状
    auto x = Tensor({1.0, 2.0}, Shape{1, 1, 2});
    Shape target_shape{2, 3, 2};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data = result.to_vector<float>();

    // 验证广播的正确性
    for (size_t i = 0; i < result_data.size(); i += 2)
    {
        EXPECT_NEAR(result_data[i], 1.0, kTolerance);
        EXPECT_NEAR(result_data[i + 1], 2.0, kTolerance);
    }
}

TEST_F(BroadcastToOperatorTest, ForwardZeroTensor)
{
    // 测试零张量
    auto x = Tensor({0.0, 0.0}, Shape{2});
    Shape target_shape{2, 2};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data = result.to_vector<float>();

    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], 0.0, kTolerance);
    }
}

// ==================== 反向传播测试 ====================

TEST_F(BroadcastToOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x = Tensor({1.0, 2.0}, Shape{2});
    Shape target_shape{2, 2};

    auto y = broadcast_to(x, target_shape);
    y.backward();

    // broadcast_to算子的梯度：∂y/∂x = sum_to(gy, x.shape())
    auto gx_data = x.grad().to_vector<float>();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], 2.0, kTolerance);  // 广播的梯度应该求和
    }
}

TEST_F(BroadcastToOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x = Tensor({1.0, 2.0}, Shape{2});
    Shape target_shape{2, 2};

    auto y = broadcast_to(x, target_shape);
    y.backward();

    // 设置输出梯度为2
    auto gy = Tensor({2.0, 2.0, 2.0, 2.0}, Shape{2, 2});
    y.backward();

    auto gx_data = x.grad().to_vector<float>();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], 4.0, kTolerance);  // 2 * 2 = 4
    }
}

TEST_F(BroadcastToOperatorTest, BackwardToSameShape)
{
    // 测试相同形状的反向传播
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
    Shape target_shape{2, 2};

    auto y = broadcast_to(x, target_shape);
    y.backward();

    auto gx_data = x.grad().to_vector<float>();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], 1.0, kTolerance);
    }
}

TEST_F(BroadcastToOperatorTest, BackwardToLargerShape)
{
    // 测试到更大形状的反向传播
    auto x = Tensor({5.0}, Shape{1});
    Shape target_shape{3};

    auto y = broadcast_to(x, target_shape);
    y.backward();

    auto gx_data = x.grad().to_vector<float>();

    EXPECT_EQ(gx_data.size(), 1U);
    EXPECT_NEAR(gx_data[0], 3.0, kTolerance);  // 广播的梯度应该求和
}

// ==================== 边界情况测试 ====================

TEST_F(BroadcastToOperatorTest, SingleElement)
{
    // 测试单元素张量
    auto x = Tensor({5.0}, Shape{1});
    Shape target_shape{1};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    EXPECT_NEAR(result.item<float>(), 5.0, kTolerance);
}

TEST_F(BroadcastToOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<data_t> data(10, 1.0);
    auto x = Tensor(data, Shape{10});
    Shape target_shape{10, 10};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data = result.to_vector<float>();

    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], 1.0, kTolerance);
    }
}

TEST_F(BroadcastToOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    auto x = Tensor({1.0, 2.0}, Shape{1, 1, 2});
    Shape target_shape{2, 2, 2};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data = result.to_vector<float>();

    for (size_t i = 0; i < result_data.size(); i += 2)
    {
        EXPECT_NEAR(result_data[i], 1.0, kTolerance);
        EXPECT_NEAR(result_data[i + 1], 2.0, kTolerance);
    }
}

// ==================== 数值稳定性测试 ====================

TEST_F(BroadcastToOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x = Tensor({1e10, 1e-10}, Shape{2});
    Shape target_shape{2, 2};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {1e10, 1e-10, 1e10, 1e-10};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(BroadcastToOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x = Tensor({0.1, 0.2}, Shape{2});
    Shape target_shape{2, 2};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data             = result.to_vector<float>();
    std::vector<data_t> expected = {0.1, 0.2, 0.1, 0.2};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

// ==================== 特殊值测试 ====================

TEST_F(BroadcastToOperatorTest, MixedSigns)
{
    // 测试混合符号
    auto x = Tensor({1.0, -2.0}, Shape{2});
    Shape target_shape{3, 2};

    auto result = broadcast_to(x, target_shape);

    EXPECT_EQ(result.shape(), target_shape);
    auto result_data = result.to_vector<float>();

    for (size_t i = 0; i < result_data.size(); i += 2)
    {
        EXPECT_NEAR(result_data[i], 1.0, kTolerance);
        EXPECT_NEAR(result_data[i + 1], -2.0, kTolerance);
    }
}

TEST_F(BroadcastToOperatorTest, IdentityProperty)
{
    // 测试恒等性质：broadcast_to(x, x.shape()) = x
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});

    auto result = broadcast_to(x, x.shape());

    EXPECT_TRUE(tensorsEqual(result, x));
}

TEST_F(BroadcastToOperatorTest, AssociativeProperty)
{
    // 测试结合性质：broadcast_to(broadcast_to(x, shape1), shape2) = broadcast_to(x, shape2)
    auto x = Tensor({1.0, 2.0}, Shape{2});
    Shape shape1{2, 2};
    Shape shape2{2, 2, 2};

    auto result1 = broadcast_to(broadcast_to(x, shape1), shape2);
    auto result2 = broadcast_to(x, shape2);

    EXPECT_TRUE(tensorsEqual(result1, result2));
}

TEST_F(BroadcastToOperatorTest, CommutativeProperty)
{
    // 测试交换性质：broadcast_to(x + y, shape) = broadcast_to(x, shape) + broadcast_to(y, shape)
    auto x = Tensor({1.0, 2.0}, Shape{2});
    auto y = Tensor({3.0, 4.0}, Shape{2});
    Shape target_shape{2, 2};

    auto result1 = broadcast_to(x + y, target_shape);
    auto result2 = broadcast_to(x, target_shape) + broadcast_to(y, target_shape);

    EXPECT_TRUE(tensorsEqual(result1, result2));
}
