#include <gtest/gtest.h>
#include "origin/optim/hooks.h"
#include "origin/optim/sgd.h"
#include "origin/nn/layers/linear.h"
#include "origin/nn/module.h"
#include "origin/core/tensor.h"
#include "origin/core/operator.h"
#include "origin/mat/scalar.h"
#include "test_utils.h"

using namespace origin;

class WeightDecayHookTest : public ::testing::TestWithParam<DeviceType>
{
protected:
    DeviceType deviceType() const { return GetParam(); }
};

TEST_P(WeightDecayHookTest, BasicHook)
{
    // 测试基本的权重衰减Hook
    auto model = Sequential();
    model.add(std::make_unique<Linear>(1, 1, false));
    
    auto optimizer = SGD(model, 0.01f, 0.0f, 0.0f, false);  // 不使用内置的weight_decay
    
    // 注册WeightDecay Hook
    WeightDecay weight_decay(0.1f);
    optimizer.register_hook(weight_decay.hook());
    
    auto &linear = dynamic_cast<Linear &>(model[0]);
    *linear.weight() = Parameter(Tensor({1.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true)));
    
    // 确保模型在正确的设备上
    model.to(Device(deviceType()));
    
    // 通过计算图设置梯度
    auto x = Tensor({1.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({2.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));
    auto y = model(x);
    auto loss = (y - target) * (y - target);
    loss.backward();
    
    // 执行一步更新（Hook会在step()中执行）
    optimizer.step();
    
    // 验证参数已更新（如果Hook正确执行，参数更新应该包含权重衰减的影响）
    auto w_after = linear.weight()->item<float>();
    EXPECT_NE(w_after, 1.0f);
}

TEST_P(WeightDecayHookTest, HookModifiesGradient)
{
    // 测试Hook是否正确修改梯度
    auto model = Sequential();
    model.add(std::make_unique<Linear>(1, 1, false));
    
    auto optimizer = SGD(model, 0.01f, 0.0f, 0.0f, false);
    
    // 注册WeightDecay Hook
    WeightDecay weight_decay(0.1f);
    optimizer.register_hook(weight_decay.hook());
    
    auto &linear = dynamic_cast<Linear &>(model[0]);
    *linear.weight() = Parameter(Tensor({1.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true)));
    
    // 确保模型在正确的设备上
    model.to(Device(deviceType()));
    
    // 通过计算图设置梯度
    auto x = Tensor({1.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({2.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));
    auto y = model(x);
    auto loss = (y - target) * (y - target);
    loss.backward();
    
    // 获取原始梯度
    auto grad_before = linear.weight()->grad().item<float>();
    
    // 手动执行Hook（模拟optimizer.step()中的Hook执行）
    std::vector<Parameter *> params = {linear.weight()};
    weight_decay.hook()(params);
    
    // 获取修改后的梯度
    auto grad_after = linear.weight()->grad().item<float>();
    
    // 验证梯度已被修改：grad_after = grad_before + rate * param
    // rate = 0.1, param = 1.0, 所以 grad_after = grad_before + 0.1
    float expected_grad = grad_before + 0.1f * 1.0f;
    EXPECT_NEAR(grad_after, expected_grad, 1e-5f);
}

TEST_P(WeightDecayHookTest, DifferentRates)
{
    // 测试不同权重衰减率的影响
    auto model1 = Sequential();
    model1.add(std::make_unique<Linear>(1, 1, false));
    auto model2 = Sequential();
    model2.add(std::make_unique<Linear>(1, 1, false));
    
    auto optimizer1 = SGD(model1, 0.01f, 0.0f, 0.0f, false);
    auto optimizer2 = SGD(model2, 0.01f, 0.0f, 0.0f, false);
    
    // 注册不同权重的WeightDecay Hook
    WeightDecay weight_decay1(0.1f);
    WeightDecay weight_decay2(0.2f);
    optimizer1.register_hook(weight_decay1.hook());
    optimizer2.register_hook(weight_decay2.hook());
    
    auto &linear1 = dynamic_cast<Linear &>(model1[0]);
    auto &linear2 = dynamic_cast<Linear &>(model2[0]);
    
    *linear1.weight() = Parameter(Tensor({1.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true)));
    *linear2.weight() = Parameter(Tensor({1.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true)));
    
    // 确保模型在正确的设备上
    model1.to(Device(deviceType()));
    model2.to(Device(deviceType()));
    
    auto x = Tensor({1.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({2.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));
    
    // 设置相同的梯度
    auto y1 = model1(x);
    auto loss1 = (y1 - target) * (y1 - target);
    loss1.backward();
    
    auto y2 = model2(x);
    auto loss2 = (y2 - target) * (y2 - target);
    loss2.backward();
    
    // 手动执行Hook
    std::vector<Parameter *> params1 = {linear1.weight()};
    std::vector<Parameter *> params2 = {linear2.weight()};
    weight_decay1.hook()(params1);
    weight_decay2.hook()(params2);
    
    // 获取修改后的梯度
    auto grad1 = linear1.weight()->grad().item<float>();
    auto grad2 = linear2.weight()->grad().item<float>();
    
    // 验证权重衰减率大的梯度修改更多
    // grad2应该比grad1多0.1（因为rate2 - rate1 = 0.1）
    EXPECT_GT(grad2, grad1);
    EXPECT_NEAR(grad2 - grad1, 0.1f, 1e-5f);
}

TEST_P(WeightDecayHookTest, MultipleParameters)
{
    // 测试多个参数的情况
    auto model = Sequential();
    model.add(std::make_unique<Linear>(2, 3, true));
    model.add(std::make_unique<Linear>(3, 1, true));
    
    auto optimizer = SGD(model, 0.01f, 0.0f, 0.0f, false);
    
    // 注册WeightDecay Hook
    WeightDecay weight_decay(0.1f);
    optimizer.register_hook(weight_decay.hook());
    
    auto &linear1 = dynamic_cast<Linear &>(model[0]);
    auto &linear2 = dynamic_cast<Linear &>(model[1]);
    
    // 确保模型在正确的设备上
    model.to(Device(deviceType()));
    
    auto x = Tensor({1.0f, 1.0f}, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({2.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));
    
    // 通过计算图设置梯度
    auto y = model(x);
    auto loss = (y - target) * (y - target);
    loss.backward();
    
    // 手动执行Hook
    std::vector<Parameter *> params = {linear1.weight(), linear2.weight()};
    if (linear1.bias() != nullptr)
    {
        params.push_back(linear1.bias());
    }
    if (linear2.bias() != nullptr)
    {
        params.push_back(linear2.bias());
    }
    weight_decay.hook()(params);
    
    // 验证所有参数的梯度都被修改（通过检查参数已更新）
    auto w1_before = linear1.weight()->to_vector<float>()[0];
    auto w2_before = linear2.weight()->to_vector<float>()[0];
    
    optimizer.step();
    
    auto w1_after = linear1.weight()->to_vector<float>()[0];
    auto w2_after = linear2.weight()->to_vector<float>()[0];
    
    // 验证参数已更新
    EXPECT_NE(w1_after, w1_before);
    EXPECT_NE(w2_after, w2_before);
}

INSTANTIATE_TEST_SUITE_P(AllDevices, WeightDecayHookTest,
                         ::testing::Values(DeviceType::kCPU, DeviceType::kCUDA),
                         [](const ::testing::TestParamInfo<DeviceType> &info) {
                             return info.param == DeviceType::kCPU ? "CPU" : "CUDA";
                         });

