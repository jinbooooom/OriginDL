#include <arrayfire.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "origin.h"
#include "origin/utils/log.h"

using namespace origin;

class LinearRegressionTest : public ::testing::Test
{
protected:
    // 精度忍受常量
    static constexpr double kTolerance = 0.1;  // 线性回归允许较大的误差
    static constexpr double kExpectedW = 2.0;  // 期望的权重
    static constexpr double kExpectedB = 5.0;  // 期望的偏置
    
    void SetUp() override
    {
        // 测试前的设置
        // 初始化ArrayFire后端
        try
        {
            af::setBackend(AF_BACKEND_CPU);
        }
        catch (const af::exception &e)
        {
            // 忽略错误，继续测试
        }
        
        // 设置随机种子确保可重复性
        af::setSeed(0);
    }

    void TearDown() override
    {
        // 测试后的清理
    }

    // 预测函数（带偏置）
    Tensor Predict(const Tensor &x, const Tensor &w, const Tensor &b)
    {
        auto y = origin::mat_mul(x, w) + b;
        return y;
    }

    // 均方误差损失函数
    Tensor MSE(const Tensor &x0, const Tensor &x1)
    {
        auto diff       = x0 - x1;
        auto sum_result = origin::sum(origin::pow(diff, 2));
        // 使用除法算子而不是直接创建Tensor，确保有正确的creator_
        auto elements = Tensor::constant(diff.elements(), sum_result.shape());
        auto result   = sum_result / elements;
        return result;
    }
};

// 线性回归收敛性测试
TEST_F(LinearRegressionTest, ConvergeToExpectedValues)
{
    // 设置随机种子
    af::setSeed(0);

    // 生成随机数据
    size_t input_size = 100;
    auto x = Tensor::randn(Shape{input_size, 1});
    // 设置一个噪声，使真实值在预测结果附近
    auto noise = Tensor::randn(Shape{input_size, 1}) * 0.1;
    // 生成真实标签：y = x * 2.0 + 5.0 + noise
    auto y = x * kExpectedW + kExpectedB + noise;

    // 初始化权重和偏置
    auto w = Tensor::constant(0, Shape{1, 1});
    auto b = Tensor::constant(0, Shape{1, 1});

    // 设置学习率和迭代次数
    data_t lr = 0.1;
    int iters = 200;

    // 训练
    for (int i = 0; i < iters; i++)
    {
        // 清零梯度
        w.clear_grad();
        b.clear_grad();

        auto y_pred = Predict(x, w, b);
        auto loss = MSE(y, y_pred);

        // 反向传播
        loss.backward();

        // 更新参数 - 使用算子而不是直接修改data()
        w = w - lr * w.grad();
        b = b - lr * b.grad();

        // 打印结果
#if 0
        float loss_val = loss.to_vector()[0];
        float w_val = w.to_vector()[0];
        float b_val = b.to_vector()[0];
        logi("Iteration {}: loss = {:.6f}, w = {:.6f}, b = {:.6f}", i, loss_val, w_val, b_val);
#endif
    }

    // 验证权重是否收敛到期望值
    float final_w = w.to_vector()[0];
    float final_b = b.to_vector()[0];
    
    EXPECT_NEAR(final_w, kExpectedW, kTolerance) 
        << "Weight w should converge to " << kExpectedW << ", but got " << final_w;
    
    // 验证偏置是否收敛到期望值
    EXPECT_NEAR(final_b, kExpectedB, kTolerance) 
        << "Bias b should converge to " << kExpectedB << ", but got " << final_b;
}
