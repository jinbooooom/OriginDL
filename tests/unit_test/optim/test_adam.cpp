#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../../common/device_test_base.h"
#include "../../common/gtest_utils.h"
#include "../../common/test_utils.h"
#include "origin.h"
#include "origin/nn/layers/linear.h"
#include "origin/nn/module.h"
#include "origin/optim/adam.h"

using namespace origin;
// 使用命名空间别名，语法类似 PyTorch
namespace nn = origin::nn;

/**
 * @brief Adam 优化器测试类（参数化版本）
 */
class AdamOptimizerTest : public origin::test::OperatorTestBase
{};

// ==================== 基本功能测试 ====================

TEST_P(AdamOptimizerTest, BasicStep)
{
    // 测试基本的参数更新
    auto model = Sequential();
    model.add(std::make_unique<nn::Linear>(1, 1, true));

    auto optimizer = Adam(model, 0.01f, 0.9f, 0.999f, 1e-8f);

    // 设置初始参数
    auto &linear = dynamic_cast<nn::Linear &>(model[0]);
    *linear.weight() =
        Parameter(Tensor({1.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true)));
    if (linear.bias() != nullptr)
    {
        *linear.bias() =
            Parameter(Tensor({0.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true)));
    }

    // 通过计算图设置梯度：创建一个简单的损失函数
    auto x      = Tensor({1.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));
    auto y      = model(x);
    auto target = Tensor({2.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));
    auto loss   = (y - target) * (y - target);
    loss.backward();

    // 执行一步更新
    optimizer.step();

    // 验证参数已更新（应该不等于初始值）
    auto w_after = linear.weight()->item<float>();
    EXPECT_NE(w_after, 1.0f);
}

TEST_P(AdamOptimizerTest, MultipleSteps)
{
    // 测试多步更新
    auto model = Sequential();
    model.add(std::make_unique<nn::Linear>(1, 1, true));

    auto optimizer = Adam(model, 0.01f);

    auto &linear = dynamic_cast<nn::Linear &>(model[0]);
    *linear.weight() =
        Parameter(Tensor({1.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true)));

    auto x      = Tensor({1.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({2.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));

    // 确保模型在正确的设备上
    model.to(Device(deviceType()));

    // 执行多步更新
    for (int i = 0; i < 5; ++i)
    {
        optimizer.zero_grad();
        auto y    = model(x);
        auto loss = (y - target) * (y - target);
        loss.backward();
        optimizer.step();
    }

    // 验证参数已更新
    auto w_after = linear.weight()->item<float>();
    EXPECT_NE(w_after, 1.0f);
}

TEST_P(AdamOptimizerTest, StateBuffers)
{
    // 测试状态缓冲区是否正确维护
    auto model = Sequential();
    model.add(std::make_unique<nn::Linear>(2, 1, true));

    auto optimizer = Adam(model, 0.01f, 0.9f, 0.999f, 1e-8f);

    auto &linear = dynamic_cast<nn::Linear &>(model[0]);
    // Linear(2, 1) 的权重形状是 (2, 1)，不是 (1, 2)
    *linear.weight() = Parameter(
        Tensor({1.0f, 2.0f}, Shape{2, 1}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true)));

    auto x      = Tensor({1.0f, 1.0f}, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({2.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));

    // 确保模型在正确的设备上
    model.to(Device(deviceType()));

    // 第一步：设置梯度并更新
    auto w_before = static_cast<const Tensor &>(*linear.weight()).to_vector<float>();
    auto y1       = model(x);
    auto loss1    = (y1 - target) * (y1 - target);
    loss1.backward();
    optimizer.step();
    auto w_after1 = static_cast<const Tensor &>(*linear.weight()).to_vector<float>();

    // 第二步：使用不同的梯度
    optimizer.zero_grad();
    auto y2    = model(x);
    auto loss2 = (y2 - target) * (y2 - target);
    loss2.backward();
    optimizer.step();
    auto w_after2 = static_cast<const Tensor &>(*linear.weight()).to_vector<float>();

    // 验证参数在每一步都有更新
    EXPECT_NE(w_after1[0], w_before[0]);
    EXPECT_NE(w_after2[0], w_after1[0]);
}

TEST_P(AdamOptimizerTest, DifferentLearningRates)
{
    // 测试不同学习率的影响
    auto model1 = Sequential();
    model1.add(std::make_unique<nn::Linear>(1, 1, false));
    auto model2 = Sequential();
    model2.add(std::make_unique<nn::Linear>(1, 1, false));

    auto optimizer1 = Adam(model1, 0.01f);
    auto optimizer2 = Adam(model2, 0.1f);

    auto &linear1 = dynamic_cast<nn::Linear &>(model1[0]);
    auto &linear2 = dynamic_cast<nn::Linear &>(model2[0]);

    *linear1.weight() =
        Parameter(Tensor({1.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true)));
    *linear2.weight() =
        Parameter(Tensor({1.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true)));

    // 确保模型在正确的设备上
    model1.to(Device(deviceType()));
    model2.to(Device(deviceType()));

    auto x      = Tensor({1.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({2.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));

    // 设置相同的梯度
    auto y1    = model1(x);
    auto loss1 = (y1 - target) * (y1 - target);
    loss1.backward();

    auto y2    = model2(x);
    auto loss2 = (y2 - target) * (y2 - target);
    loss2.backward();

    optimizer1.step();
    optimizer2.step();

    auto w1 = linear1.weight()->item<float>();
    auto w2 = linear2.weight()->item<float>();

    // 学习率大的应该更新更多
    EXPECT_GT(std::abs(w2 - 1.0f), std::abs(w1 - 1.0f));
}

TEST_P(AdamOptimizerTest, DefaultParameters)
{
    // 测试默认参数
    auto model = Sequential();
    model.add(std::make_unique<nn::Linear>(1, 1, false));
    auto optimizer = Adam(model, 0.01f);  // 使用默认的 beta1, beta2, eps

    auto &linear = dynamic_cast<nn::Linear &>(model[0]);
    *linear.weight() =
        Parameter(Tensor({1.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true)));

    // 确保模型在正确的设备上
    model.to(Device(deviceType()));

    auto x      = Tensor({1.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({2.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));
    auto y      = model(x);
    auto loss   = (y - target) * (y - target);
    loss.backward();

    optimizer.step();

    // 验证参数已更新
    auto w_after = linear.weight()->item<float>();
    EXPECT_NE(w_after, 1.0f);
}

TEST_P(AdamOptimizerTest, ZeroGradient)
{
    // 测试零梯度的情况（完美预测）
    auto model = Sequential();
    model.add(std::make_unique<nn::Linear>(1, 1, false));
    auto optimizer = Adam(model, 0.01f);

    auto &linear = dynamic_cast<nn::Linear &>(model[0]);
    *linear.weight() =
        Parameter(Tensor({1.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true)));

    // 确保模型在正确的设备上
    model.to(Device(deviceType()));

    // 创建一个完美预测的情况（y == target，梯度应该接近0）
    auto x      = Tensor({1.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));
    auto y      = model(x);
    auto target = y;  // 目标等于预测，损失为0
    auto loss   = (y - target) * (y - target);
    loss.backward();

    auto w_before = linear.weight()->item<float>();
    optimizer.step();
    auto w_after = linear.weight()->item<float>();

    // 零梯度时，参数应该不变（或变化很小，因为偏差修正）
    EXPECT_NEAR(w_after, w_before, 1e-4f);
}

TEST_P(AdamOptimizerTest, BiasCorrection)
{
    // 测试偏差修正（bias correction）
    // 第一步时，m_hat 和 v_hat 应该被放大
    auto model = Sequential();
    model.add(std::make_unique<nn::Linear>(1, 1, false));
    auto optimizer = Adam(model, 0.01f, 0.9f, 0.999f, 1e-8f);

    auto &linear = dynamic_cast<nn::Linear &>(model[0]);
    *linear.weight() =
        Parameter(Tensor({1.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true)));

    // 确保模型在正确的设备上
    model.to(Device(deviceType()));

    auto x      = Tensor({1.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({2.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));

    // 第一步
    auto y1    = model(x);
    auto loss1 = (y1 - target) * (y1 - target);
    loss1.backward();
    optimizer.step();
    auto w_after_step1 = linear.weight()->item<float>();

    // 第二步（偏差修正应该变小）
    optimizer.zero_grad();
    auto y2    = model(x);
    auto loss2 = (y2 - target) * (y2 - target);
    loss2.backward();
    optimizer.step();
    auto w_after_step2 = linear.weight()->item<float>();

    // 验证参数在更新
    EXPECT_NE(w_after_step1, 1.0f);
    EXPECT_NE(w_after_step2, w_after_step1);
}

TEST_P(AdamOptimizerTest, MultipleParameters)
{
    // 测试多个参数的情况
    auto model = Sequential();
    model.add(std::make_unique<nn::Linear>(2, 3, true));
    model.add(std::make_unique<nn::Linear>(3, 1, true));

    auto optimizer = Adam(model, 0.01f);

    // 确保模型在正确的设备上
    model.to(Device(deviceType()));

    // 设置参数需要梯度
    auto &linear1 = dynamic_cast<nn::Linear &>(model[0]);
    auto &linear2 = dynamic_cast<nn::Linear &>(model[1]);

    // 通过计算图设置梯度
    auto x      = Tensor({1.0f, 1.0f}, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({2.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));

    // 执行更新
    auto y    = model(x);
    auto loss = (y - target) * (y - target);
    loss.backward();
    optimizer.step();

    // 验证所有参数都已更新（通过检查参数值是否改变）
    // 获取第一个元素的初始值
    auto w1_before_data = static_cast<const Tensor &>(*linear1.weight()).to_vector<float>();
    auto w2_before_data = static_cast<const Tensor &>(*linear2.weight()).to_vector<float>();

    // 再次执行更新
    optimizer.zero_grad();
    auto y2    = model(x);
    auto loss2 = (y2 - target) * (y2 - target);
    loss2.backward();
    optimizer.step();

    auto w1_after_data = static_cast<const Tensor &>(*linear1.weight()).to_vector<float>();
    auto w2_after_data = static_cast<const Tensor &>(*linear2.weight()).to_vector<float>();

    // 验证参数已更新
    EXPECT_NE(w1_after_data[0], w1_before_data[0]);
    EXPECT_NE(w2_after_data[0], w2_before_data[0]);
}

// Instantiate test suite: automatically generate tests for CPU and available CUDA
INSTANTIATE_DEVICE_TEST_SUITE_P(AdamOptimizerTest);
