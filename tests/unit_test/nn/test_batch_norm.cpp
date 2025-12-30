#include <gtest/gtest.h>
#include "origin/nn/layers/batch_norm1d.h"
#include "origin/nn/layers/batch_norm2d.h"
#include "origin/core/tensor.h"
#include "origin/core/operator.h"
#include "test_utils.h"

using namespace origin;
namespace nn = origin::nn;

class BatchNorm1dTest : public ::testing::TestWithParam<DeviceType>
{
protected:
    DeviceType deviceType() const { return GetParam(); }
};

TEST_P(BatchNorm1dTest, BasicForward)
{
    // 测试基本的 BatchNorm1d 层前向传播
    nn::BatchNorm1d bn(4, 1e-5f, 0.1f);
    bn.to(Device(deviceType()));
    bn.train(true);

    // 创建输入 (N=2, C=4)
    std::vector<float> x_data = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    auto x = Tensor(x_data, Shape{2, 4}, dtype(DataType::kFloat32).device(deviceType()));

    // 前向传播
    auto y = bn.forward(x);

    // 验证输出形状
    EXPECT_EQ(y.shape().size(), 2U);
    EXPECT_EQ(y.shape()[0], 2U);
    EXPECT_EQ(y.shape()[1], 4U);
}

TEST_P(BatchNorm1dTest, EvalMode)
{
    // 测试评估模式
    nn::BatchNorm1d bn(4, 1e-5f, 0.1f);
    bn.to(Device(deviceType()));
    bn.eval();

    std::vector<float> x_data = {
        1.0f, 2.0f, 3.0f, 4.0f,
        5.0f, 6.0f, 7.0f, 8.0f
    };
    auto x = Tensor(x_data, Shape{2, 4}, dtype(DataType::kFloat32).device(deviceType()));

    auto y = bn.forward(x);

    EXPECT_EQ(y.shape().size(), 2U);
    EXPECT_EQ(y.shape()[0], 2U);
    EXPECT_EQ(y.shape()[1], 4U);
}

class BatchNorm2dTest : public ::testing::TestWithParam<DeviceType>
{
protected:
    DeviceType deviceType() const { return GetParam(); }
};

TEST_P(BatchNorm2dTest, BasicForward)
{
    // 测试基本的 BatchNorm2d 层前向传播
    nn::BatchNorm2d bn(3, 1e-5f, 0.1f);
    bn.to(Device(deviceType()));
    bn.train(true);

    // 创建输入 (N=2, C=3, H=4, W=4)
    std::vector<float> x_data(2 * 3 * 4 * 4, 1.0f);
    auto x = Tensor(x_data, Shape{2, 3, 4, 4}, dtype(DataType::kFloat32).device(deviceType()));

    // 前向传播
    auto y = bn.forward(x);

    // 验证输出形状
    EXPECT_EQ(y.shape().size(), 4U);
    EXPECT_EQ(y.shape()[0], 2U);
    EXPECT_EQ(y.shape()[1], 3U);
    EXPECT_EQ(y.shape()[2], 4U);
    EXPECT_EQ(y.shape()[3], 4U);
}

TEST_P(BatchNorm2dTest, EvalMode)
{
    // 测试评估模式
    nn::BatchNorm2d bn(3, 1e-5f, 0.1f);
    bn.to(Device(deviceType()));
    bn.eval();

    std::vector<float> x_data(2 * 3 * 4 * 4, 1.0f);
    auto x = Tensor(x_data, Shape{2, 3, 4, 4}, dtype(DataType::kFloat32).device(deviceType()));

    auto y = bn.forward(x);

    EXPECT_EQ(y.shape().size(), 4U);
    EXPECT_EQ(y.shape()[0], 2U);
    EXPECT_EQ(y.shape()[1], 3U);
    EXPECT_EQ(y.shape()[2], 4U);
    EXPECT_EQ(y.shape()[3], 4U);
}

INSTANTIATE_TEST_SUITE_P(BatchNorm1dTests, BatchNorm1dTest, 
                         ::testing::Values(DeviceType::kCPU));

INSTANTIATE_TEST_SUITE_P(BatchNorm2dTests, BatchNorm2dTest, 
                         ::testing::Values(DeviceType::kCPU));

