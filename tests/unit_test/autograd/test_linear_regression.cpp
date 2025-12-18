#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "origin.h"
#include "origin/utils/log.h"
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"

using namespace origin;
/**
 * @brief 线性回归测试类（参数化版本）
 * @details 使用参数化测试，自动为CPU和CUDA生成测试用例
 *          无GPU环境只运行CPU测试，有GPU环境运行CPU+CUDA测试
 */
class LinearRegressionTest : public origin::test::OperatorTestBase
{
protected:
    // 精度忍受常量
    static constexpr double kTolerance = 0.1;   // 线性回归允许较大的误差
    static constexpr float kExpectedW  = 2.0f;  // 期望的权重
    static constexpr float kExpectedB  = 5.0f;  // 期望的偏置

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
        auto sum_result = origin::sum(origin::pow(diff, 2.0f));
        // 使用除法算子而不是直接创建Tensor，确保有正确的creator_
        // 确保elements张量在正确的设备上（使用sum_result的设备，而不是deviceType()）
        auto elements = Tensor(static_cast<float>(diff.elements()), sum_result.shape(), 
                                     dtype(DataType::kFloat32).device(sum_result.device()));
        auto result   = sum_result / elements;
        return result;
    }
};

// 线性回归收敛性测试
TEST_P(LinearRegressionTest, ConvergeToExpectedValues)
{
    // 生成随机数据
    size_t input_size = 256; // 数据量足够大，cuda才有优势，如 256000，不然 cuda 的耗时都花在 launch kernel 上了
    auto x            = Tensor::randn(Shape{input_size, 1}, dtype(DataType::kFloat32).device(deviceType()));
    // 设置一个噪声，使真实值在预测结果附近
    auto noise = Tensor::randn(Shape{input_size, 1}, dtype(DataType::kFloat32).device(deviceType())) * 0.1f;
    // 生成真实标签：y = x * 2.0 + 5.0 + noise
    auto y = x * kExpectedW + kExpectedB + noise;

    // 初始化权重和偏置
    auto w = Tensor(0.0f, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    auto b = Tensor(0.0f, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    // 设置学习率和迭代次数
    float lr  = 0.1f;
    int iters = 200;

    // 训练
    for (int i = 0; i < iters; i++)
    {
        // 清零梯度
        w.clear_grad();
        b.clear_grad();

        auto y_pred = Predict(x, w, b);
        auto loss   = MSE(y, y_pred);

        // 反向传播
        loss.backward();

        // 更新参数 - 使用算子而不是直接修改data()
        w = w - lr * w.grad();
        b = b - lr * b.grad();

        // 打印结果
#if 0
        float loss_val = loss.item<float>();
        float w_val = w.item<float>();
        float b_val = b.item<float>();
        logi("Iteration {}: loss = {:.6f}, w = {:.6f}, b = {:.6f}", i, loss_val, w_val, b_val);
#endif
    }

    // 验证权重是否收敛到期望值
    float final_w = w.item<float>();
    float final_b = b.item<float>();

    EXPECT_NEAR(final_w, kExpectedW, kTolerance)
        << "Weight w should converge to " << kExpectedW << ", but got " << final_w;

    // 验证偏置是否收敛到期望值
    EXPECT_NEAR(final_b, kExpectedB, kTolerance)
        << "Bias b should converge to " << kExpectedB << ", but got " << final_b;
}

// 实例化测试套件：自动为CPU和可用CUDA生成测试
INSTANTIATE_DEVICE_TEST_SUITE_P(LinearRegressionTest);
