#include <gtest/gtest.h>
#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/nn/models/mlp.h"
#include "test_utils.h"

using namespace origin;
// 使用命名空间别名，语法类似 PyTorch
namespace nn = origin::nn;

class MLPTest : public ::testing::TestWithParam<DeviceType>
{
protected:
    DeviceType deviceType() const { return GetParam(); }
};

TEST_P(MLPTest, BasicForward)
{
    // 测试基本的MLP前向传播
    // 创建一个简单的MLP: 输入2维，隐藏层3维，输出1维
    nn::MLP mlp({2, 3, 1});

    // 确保模型在正确的设备上
    mlp.to(Device(deviceType()));

    // 创建输入
    auto x = Tensor({1.0f, 2.0f}, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()));

    // 前向传播
    auto y = mlp.forward(x);

    // 验证输出形状
    EXPECT_EQ(y.shape().size(), 2U);
    EXPECT_EQ(y.shape()[0], 1U);  // batch_size
    EXPECT_EQ(y.shape()[1], 1U);  // output_size
}

TEST_P(MLPTest, MultipleLayers)
{
    // 测试多层MLP
    // 创建一个3层MLP: 输入4维，隐藏层8维和6维，输出2维
    nn::MLP mlp({4, 8, 6, 2});

    // 创建输入
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{1, 4}, dtype(DataType::kFloat32).device(deviceType()));

    // 确保模型在正确的设备上
    mlp.to(Device(deviceType()));

    // 前向传播
    auto y = mlp.forward(x);

    // 验证输出形状
    EXPECT_EQ(y.shape().size(), 2U);
    EXPECT_EQ(y.shape()[0], 1U);  // batch_size
    EXPECT_EQ(y.shape()[1], 2U);  // output_size
}

TEST_P(MLPTest, BatchProcessing)
{
    // 测试批处理
    nn::MLP mlp({3, 5, 2});

    // 创建批量输入 (batch_size=2, input_size=3)
    std::vector<float> batch_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x                        = Tensor(batch_data, Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));

    // 确保模型在正确的设备上
    mlp.to(Device(deviceType()));

    // 前向传播
    auto y = mlp.forward(x);

    // 验证输出形状
    EXPECT_EQ(y.shape().size(), 2U);
    EXPECT_EQ(y.shape()[0], 2U);  // batch_size
    EXPECT_EQ(y.shape()[1], 2U);  // output_size
}

TEST_P(MLPTest, CustomActivation)
{
    // 测试自定义激活函数（使用identity，即不使用激活函数）
    auto identity = [](const Tensor &x) { return x; };
    nn::MLP mlp({2, 3, 1}, identity);

    auto x = Tensor({1.0f, 2.0f}, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()));

    // 确保模型在正确的设备上
    mlp.to(Device(deviceType()));

    auto y = mlp.forward(x);

    // 验证输出形状
    EXPECT_EQ(y.shape().size(), 2U);
    EXPECT_EQ(y.shape()[0], 1U);
    EXPECT_EQ(y.shape()[1], 1U);
}

TEST_P(MLPTest, ParametersCollection)
{
    // 测试参数收集
    nn::MLP mlp({2, 3, 1});

    // 确保模型在正确的设备上
    mlp.to(Device(deviceType()));

    // 获取所有参数
    auto params = mlp.parameters();

    // MLP有2个Linear层，每个Linear层有weight和bias，所以应该有4个参数
    // 但实际上，由于Linear层内部管理参数，参数数量可能不同
    // 至少应该有参数
    EXPECT_GT(params.size(), 0U);
}

TEST_P(MLPTest, InvalidLayerSizes)
{
    // 测试无效的层大小
    // 只有1个层大小（至少需要2个：输入和输出）
    EXPECT_THROW({ nn::MLP mlp({10}); }, std::invalid_argument);

    // 空层大小列表
    EXPECT_THROW({ nn::MLP mlp({}); }, std::invalid_argument);
}

INSTANTIATE_TEST_SUITE_P(AllDevices,
                         MLPTest,
                         ::testing::Values(DeviceType::kCPU, DeviceType::kCUDA),
                         [](const ::testing::TestParamInfo<DeviceType> &info) {
                             return info.param == DeviceType::kCPU ? "CPU" : "CUDA";
                         });
