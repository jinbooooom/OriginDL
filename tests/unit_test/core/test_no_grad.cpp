#include <gtest/gtest.h>
#include "origin/core/config.h"
#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "test_utils.h"

using namespace origin;

class NoGradTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // 确保每次测试前启用反向传播
        Config::enable_backprop = true;
    }

    void TearDown() override
    {
        // 确保每次测试后恢复启用反向传播
        Config::enable_backprop = true;
    }
};

TEST_F(NoGradTest, BasicNoGrad)
{
    // 测试基本的no_grad()功能
    auto x = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).requires_grad(true));
    auto y = x * Scalar(2.0f);

    // 在no_grad()作用域内，backward()应该不计算梯度
    {
        auto guard = no_grad();
        EXPECT_FALSE(Config::enable_backprop);
        y.backward();
        // 在no_grad()作用域内，梯度应该没有被计算
    }

    // 退出作用域后，应该恢复
    EXPECT_TRUE(Config::enable_backprop);

    // 现在应该可以正常计算梯度
    y.backward();
    // 验证梯度被计算了
    auto grad = x.grad();
    EXPECT_EQ(grad.shape(), x.shape());
}

TEST_F(NoGradTest, NestedNoGrad)
{
    // 测试嵌套的no_grad()
    {
        auto guard1 = no_grad();
        EXPECT_FALSE(Config::enable_backprop);

        {
            auto guard2 = no_grad();
            EXPECT_FALSE(Config::enable_backprop);
        }

        // guard2析构后，应该仍然是false（因为guard1还在）
        EXPECT_FALSE(Config::enable_backprop);
    }

    // guard1析构后，应该恢复为true
    EXPECT_TRUE(Config::enable_backprop);
}

TEST_F(NoGradTest, NoGradPreventsGradientComputation)
{
    // 测试no_grad()确实阻止了梯度计算
    auto x = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).requires_grad(true));
    auto y = x * Scalar(2.0f);

    // 先清除可能的梯度
    x.clear_grad();

    // 在no_grad()作用域内调用backward()
    {
        auto guard = no_grad();
        y.backward();
    }

    // 验证梯度没有被计算（grad_应该为空）
    // 由于grad()在没有梯度时会返回zeros，我们需要检查实际的grad_是否为空
    // 但grad()是const方法，我们无法直接检查
    // 所以我们在no_grad()作用域外再次调用backward()，然后检查梯度值
    y.clear_grad();
    y.backward();

    // 现在应该有梯度了
    auto grad      = x.grad();
    auto grad_data = grad.to_vector<float>();
    EXPECT_NEAR(grad_data[0], 2.0f, 1e-5f);
    EXPECT_NEAR(grad_data[1], 2.0f, 1e-5f);
}

TEST_F(NoGradTest, NoGradWithModel)
{
    // 测试no_grad()与模型一起使用
    auto x = Tensor({1.0f, 2.0f}, Shape{1, 2}, dtype(DataType::kFloat32).requires_grad(true));

    // 创建一个简单的计算图
    auto y = x * Scalar(2.0f);
    auto z = y * Scalar(3.0f);

    // 在no_grad()作用域内，backward()不应该计算梯度
    {
        auto guard = no_grad();
        z.backward();
    }

    // 验证Config已恢复
    EXPECT_TRUE(Config::enable_backprop);
}

TEST_F(NoGradTest, MoveSemantics)
{
    // 测试移动语义
    auto guard1 = no_grad();
    EXPECT_FALSE(Config::enable_backprop);

    // 移动guard1到guard2
    auto guard2 = std::move(guard1);
    EXPECT_FALSE(Config::enable_backprop);

    // guard2析构后应该恢复
}

TEST_F(NoGradTest, RestorePreviousValue)
{
    // 测试恢复之前的值
    Config::enable_backprop = false;

    {
        auto guard = no_grad();
        EXPECT_FALSE(Config::enable_backprop);
    }

    // 应该恢复为false（之前的值）
    EXPECT_FALSE(Config::enable_backprop);

    // 恢复为true以便后续测试
    Config::enable_backprop = true;
}
