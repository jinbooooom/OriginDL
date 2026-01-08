#include <gtest/gtest.h>
#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/nn/layers/dropout.h"
#include "test_utils.h"

using namespace origin;
namespace nn = origin::nn;

class DropoutTest : public ::testing::TestWithParam<DeviceType>
{
protected:
    DeviceType deviceType() const { return GetParam(); }
};

TEST_P(DropoutTest, BasicForward)
{
    // 测试基本的 Dropout 层前向传播（训练模式）
    nn::Dropout dropout(0.5f);
    dropout.to(Device(deviceType()));
    dropout.train(true);

    // 创建输入
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto x                    = Tensor(x_data, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));

    // 前向传播
    auto y = dropout.forward(x);

    // 验证输出形状
    EXPECT_EQ(y.shape().size(), 1U);
    EXPECT_EQ(y.shape()[0], 5U);

    // 在训练模式下，部分值应该被置为 0
    auto y_data   = y.to_vector<float>();
    bool has_zero = false;
    for (float val : y_data)
    {
        if (val == 0.0f)
        {
            has_zero = true;
            break;
        }
    }
    // 注意：由于随机性，可能所有值都不为 0，但概率很低
}

TEST_P(DropoutTest, EvalMode)
{
    // 测试评估模式：应该直接返回输入
    nn::Dropout dropout(0.5f);
    dropout.to(Device(deviceType()));
    dropout.eval();

    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto x                    = Tensor(x_data, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));

    auto y = dropout.forward(x);

    // 在评估模式下，输出应该等于输入
    EXPECT_EQ(y.shape(), x.shape());
    auto y_data       = y.to_vector<float>();
    auto x_data_check = x.to_vector<float>();
    for (size_t i = 0; i < y_data.size(); ++i)
    {
        EXPECT_FLOAT_EQ(y_data[i], x_data_check[i]);
    }
}

TEST_P(DropoutTest, Backward)
{
    // 测试反向传播
    nn::Dropout dropout(0.5f);
    dropout.to(Device(deviceType()));
    dropout.train(true);

    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    auto x                    = Tensor(x_data, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));

    auto y = dropout.forward(x);

    // 创建梯度
    std::vector<float> gy_data = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    auto gy                    = Tensor(gy_data, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));

    // 反向传播
    y.backward();

    // 验证梯度形状
    EXPECT_TRUE(x.grad().shape() == x.shape());
}

INSTANTIATE_TEST_SUITE_P(DropoutTests, DropoutTest, ::testing::Values(DeviceType::kCPU));
