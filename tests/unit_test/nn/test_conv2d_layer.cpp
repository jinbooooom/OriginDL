#include <gtest/gtest.h>
#include "origin/nn/layers/conv2d.h"
#include "origin/core/tensor.h"
#include "origin/core/operator.h"
#include "test_utils.h"

using namespace origin;

class Conv2dLayerTest : public ::testing::TestWithParam<DeviceType>
{
protected:
    DeviceType deviceType() const { return GetParam(); }
};

TEST_P(Conv2dLayerTest, BasicForward)
{
    // 测试基本的Conv2d层前向传播
    // 创建一个简单的Conv2d层: 1输入通道 -> 1输出通道，3x3卷积核
    Conv2d conv(1, 1, {3, 3}, {1, 1}, {1, 1}, true);
    
    // 确保层在正确的设备上
    conv.to(Device(deviceType()));
    
    // 创建输入 (N=1, C=1, H=5, W=5)
    std::vector<float> x_data(1 * 1 * 5 * 5, 1.0f);
    auto x = Tensor(x_data, Shape{1, 1, 5, 5}, dtype(DataType::kFloat32).device(deviceType()));
    
    // 前向传播
    auto y = conv.forward(x);
    
    // 验证输出形状 (N=1, OC=1, OH=5, OW=5) - 因为pad=1，所以输出尺寸不变
    EXPECT_EQ(y.shape().size(), 4U);
    EXPECT_EQ(y.shape()[0], 1U);  // batch_size
    EXPECT_EQ(y.shape()[1], 1U);  // output_channels
    EXPECT_EQ(y.shape()[2], 5U);  // height
    EXPECT_EQ(y.shape()[3], 5U);  // width
}

TEST_P(Conv2dLayerTest, MultipleChannels)
{
    // 测试多通道Conv2d层
    // 创建一个Conv2d层: 3输入通道 -> 64输出通道，3x3卷积核
    Conv2d conv(3, 64, {3, 3}, {1, 1}, {1, 1}, true);
    
    // 确保层在正确的设备上
    conv.to(Device(deviceType()));
    
    // 创建输入 (N=2, C=3, H=32, W=32)
    std::vector<float> x_data(2 * 3 * 32 * 32, 1.0f);
    auto x = Tensor(x_data, Shape{2, 3, 32, 32}, dtype(DataType::kFloat32).device(deviceType()));
    
    // 前向传播
    auto y = conv.forward(x);
    
    // 验证输出形状 (N=2, OC=64, OH=32, OW=32)
    EXPECT_EQ(y.shape().size(), 4U);
    EXPECT_EQ(y.shape()[0], 2U);   // batch_size
    EXPECT_EQ(y.shape()[1], 64U);  // output_channels
    EXPECT_EQ(y.shape()[2], 32U);  // height
    EXPECT_EQ(y.shape()[3], 32U);  // width
}

TEST_P(Conv2dLayerTest, WithStride)
{
    // 测试带步长的Conv2d层
    // 创建一个Conv2d层: 1输入通道 -> 1输出通道，3x3卷积核，stride=2
    Conv2d conv(1, 1, {3, 3}, {2, 2}, {1, 1}, true);
    
    // 确保层在正确的设备上
    conv.to(Device(deviceType()));
    
    // 创建输入 (N=1, C=1, H=5, W=5)
    std::vector<float> x_data(1 * 1 * 5 * 5, 1.0f);
    auto x = Tensor(x_data, Shape{1, 1, 5, 5}, dtype(DataType::kFloat32).device(deviceType()));
    
    // 前向传播
    auto y = conv.forward(x);
    
    // 验证输出形状 - stride=2，输出尺寸约为输入的一半
    EXPECT_EQ(y.shape().size(), 4U);
    EXPECT_EQ(y.shape()[0], 1U);  // batch_size
    EXPECT_EQ(y.shape()[1], 1U);  // output_channels
    // 输出高度和宽度应该是 (5 + 2*1 - 3) / 2 + 1 = 3
    EXPECT_EQ(y.shape()[2], 3U);  // height
    EXPECT_EQ(y.shape()[3], 3U);  // width
}

TEST_P(Conv2dLayerTest, WithoutBias)
{
    // 测试不使用偏置的Conv2d层
    Conv2d conv(1, 1, {3, 3}, {1, 1}, {1, 1}, false);
    
    // 确保层在正确的设备上
    conv.to(Device(deviceType()));
    
    // 创建输入
    std::vector<float> x_data(1 * 1 * 5 * 5, 1.0f);
    auto x = Tensor(x_data, Shape{1, 1, 5, 5}, dtype(DataType::kFloat32).device(deviceType()));
    
    // 前向传播
    auto y = conv.forward(x);
    
    // 验证输出形状
    EXPECT_EQ(y.shape().size(), 4U);
    EXPECT_EQ(y.shape()[0], 1U);
    EXPECT_EQ(y.shape()[1], 1U);
    
    // 验证bias()返回nullptr
    EXPECT_EQ(conv.bias(), nullptr);
}

TEST_P(Conv2dLayerTest, SingleValueConstructor)
{
    // 测试单值构造函数（kernel_size, stride, pad为单个值）
    Conv2d conv(1, 64, 3, 1, 1, true);
    
    // 确保层在正确的设备上
    conv.to(Device(deviceType()));
    
    // 创建输入
    std::vector<float> x_data(1 * 1 * 28 * 28, 1.0f);
    auto x = Tensor(x_data, Shape{1, 1, 28, 28}, dtype(DataType::kFloat32).device(deviceType()));
    
    // 前向传播
    auto y = conv.forward(x);
    
    // 验证输出形状
    EXPECT_EQ(y.shape().size(), 4U);
    EXPECT_EQ(y.shape()[0], 1U);
    EXPECT_EQ(y.shape()[1], 64U);
    EXPECT_EQ(y.shape()[2], 28U);  // pad=1，尺寸不变
    EXPECT_EQ(y.shape()[3], 28U);
}

TEST_P(Conv2dLayerTest, ParametersCollection)
{
    // 测试参数收集
    Conv2d conv(1, 64, {3, 3}, {1, 1}, {1, 1}, true);
    
    // 确保层在正确的设备上
    conv.to(Device(deviceType()));
    
    // 获取所有参数
    auto params = conv.parameters();
    
    // Conv2d层应该有weight和bias两个参数
    EXPECT_GE(params.size(), 1U);  // 至少有一个weight参数
    if (conv.bias() != nullptr)
    {
        EXPECT_GE(params.size(), 2U);  // 如果有bias，应该有两个参数
    }
    
    // 验证weight和bias访问器
    EXPECT_NE(conv.weight(), nullptr);
    if (conv.bias() != nullptr)
    {
        EXPECT_NE(conv.bias(), nullptr);
    }
}

TEST_P(Conv2dLayerTest, InvalidInputShape)
{
    // 测试无效的输入形状
    Conv2d conv(1, 64, {3, 3}, {1, 1}, {1, 1}, true);
    conv.to(Device(deviceType()));
    
    // 输入不是4D张量
    auto x = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    
    EXPECT_THROW({
        conv.forward(x);
    }, std::runtime_error);
}

TEST_P(Conv2dLayerTest, InvalidChannelMismatch)
{
    // 测试通道数不匹配
    Conv2d conv(1, 64, {3, 3}, {1, 1}, {1, 1}, true);
    conv.to(Device(deviceType()));
    
    // 输入通道数为3，但层期望1
    std::vector<float> x_data(1 * 3 * 5 * 5, 1.0f);
    auto x = Tensor(x_data, Shape{1, 3, 5, 5}, dtype(DataType::kFloat32).device(deviceType()));
    
    EXPECT_THROW({
        conv.forward(x);
    }, std::runtime_error);
}

TEST_P(Conv2dLayerTest, InvalidConstructorParams)
{
    // 测试无效的构造函数参数
    EXPECT_THROW({
        Conv2d conv(0, 64, {3, 3}, {1, 1}, {1, 1}, true);
    }, std::invalid_argument);
    
    EXPECT_THROW({
        Conv2d conv(1, 0, {3, 3}, {1, 1}, {1, 1}, true);
    }, std::invalid_argument);
    
    EXPECT_THROW({
        Conv2d conv(1, 64, {0, 3}, {1, 1}, {1, 1}, true);
    }, std::invalid_argument);
    
    EXPECT_THROW({
        Conv2d conv(1, 64, {3, 3}, {0, 1}, {1, 1}, true);
    }, std::invalid_argument);
    
    EXPECT_THROW({
        Conv2d conv(1, 64, {3, 3}, {1, 1}, {-1, 1}, true);
    }, std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(AllDevices, Conv2dLayerTest,
                         ::testing::Values(DeviceType::kCPU, DeviceType::kCUDA),
                         [](const ::testing::TestParamInfo<DeviceType> &info) {
                             return info.param == DeviceType::kCPU ? "CPU" : "CUDA";
                         });

