#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "origin.h"
#include "origin/utils/log.h"

using namespace origin;

/**
 * @brief CUDA线性回归自动微分测试类
 * @details 测试CUDA张量的线性回归自动微分功能
 */
class CudaLinearRegressionTest : public ::testing::Test
{
protected:
    // 精度忍受常量
    static constexpr double kTolerance = 0.1;   // 线性回归允许较大的误差
    static constexpr float kExpectedW  = 2.0f;  // 期望的权重
    static constexpr float kExpectedB  = 5.0f;  // 期望的偏置

    void SetUp() override {}

    void TearDown() override
    {
        // 清理CUDA资源
        cudaDeviceSynchronize();
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
        auto sum_result = origin::sum(origin::pow(diff, 2.0f));
        // 使用除法算子而不是直接创建Tensor，确保有正确的creator_
        auto elements =
            Tensor(static_cast<float>(diff.elements()), sum_result.shape(), dtype(Float32).device(sum_result.device()));
        auto result = sum_result / elements;
        return result;
    }
};

/**
 * @brief CUDA线性回归收敛性测试（基本功能测试）
 * @details 测试完整的训练流程，验证参数是否能收敛到期望值
 *
 * 测试目的：
 * 1. 验证前向传播：y = x * w + b 计算正确
 * 2. 验证损失函数：MSE损失计算正确
 * 3. 验证反向传播：梯度计算和参数更新正确
 * 4. 验证收敛性：经过多轮训练后参数收敛到真实值附近
 *
 * 这是最核心的功能测试，确保整个自动微分训练流程正常工作
 */
TEST_F(CudaLinearRegressionTest, ConvergeToExpectedValues)
{
    // 生成随机数据
    size_t input_size = 100;
    auto x            = Tensor::randn(Shape{input_size, 1}, dtype(DataType::kFloat32).device(kCUDA));
    // 设置一个噪声，使真实值在预测结果附近
    auto noise = Tensor::randn(Shape{input_size, 1}, dtype(DataType::kFloat32).device(kCUDA)) * 0.1f;
    // 生成真实标签：y = x * 2.0 + 5.0 + noise
    auto y = x * kExpectedW + kExpectedB + noise;

    // 初始化权重和偏置
    auto w = Tensor(0.0f, Shape{1, 1}, dtype(Float32).device(kCUDA));
    auto b = Tensor(0.0f, Shape{1, 1}, dtype(Float32).device(kCUDA));

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
    }

    // 验证权重是否收敛到期望值
    float final_w = w.to_vector<float>()[0];
    float final_b = b.to_vector<float>()[0];

    EXPECT_NEAR(final_w, kExpectedW, kTolerance)
        << "Weight w should converge to " << kExpectedW << ", but got " << final_w;

    // 验证偏置是否收敛到期望值
    EXPECT_NEAR(final_b, kExpectedB, kTolerance)
        << "Bias b should converge to " << kExpectedB << ", but got " << final_b;
}

/**
 * @brief CUDA线性回归梯度计算测试
 * @details 专门测试反向传播中梯度计算的正确性
 *
 * 测试目的：
 * 1. 验证梯度张量形状正确：确保梯度张量与参数张量形状匹配
 * 2. 验证梯度数值非零：对于非最优参数，梯度应该不为零
 * 3. 验证梯度计算机制：确保backward()函数能正确计算梯度
 * 4. 验证梯度存储：确保梯度能正确存储在参数的grad()中
 *
 * 这个测试专注于梯度计算环节，帮助快速定位反向传播问题
 */
TEST_F(CudaLinearRegressionTest, GradientComputation)
{
    // 创建简单的测试数据
    auto x = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{4, 1}, dtype(Float32).device(kCUDA));
    auto y = Tensor(std::vector<float>{7.0f, 9.0f, 11.0f, 13.0f}, Shape{4, 1}, dtype(Float32).device(kCUDA));

    // 初始化参数
    auto w = Tensor(1.0f, Shape{1, 1}, dtype(Float32).device(kCUDA));
    auto b = Tensor(1.0f, Shape{1, 1}, dtype(Float32).device(kCUDA));

    // 前向传播
    auto y_pred = Predict(x, w, b);
    auto loss   = MSE(y, y_pred);

    // 反向传播
    loss.backward();

    // 检查梯度是否存在
    EXPECT_TRUE(w.grad().shape() == Shape({1, 1}));
    EXPECT_TRUE(b.grad().shape() == Shape({1, 1}));

    // 检查梯度不为零（对于非最优参数）
    auto w_grad = w.grad().to_vector<float>();
    auto b_grad = b.grad().to_vector<float>();

    EXPECT_GT(std::abs(w_grad[0]), 1e-6);
    EXPECT_GT(std::abs(b_grad[0]), 1e-6);
}

/**
 * @brief CUDA线性回归数值稳定性测试
 * @details 测试在极端条件下（大学习率）的数值稳定性和鲁棒性
 *
 * 测试目的：
 * 1. 验证大学习率下的稳定性：确保参数不会发散或产生NaN/Inf值
 * 2. 验证数值有限性：每个训练步骤后参数都保持有限值
 * 3. 验证边界条件收敛：即使在不利条件下仍能收敛到合理范围
 * 4. 验证鲁棒性：确保算法对超参数变化有一定的容忍度
 *
 * 这个测试确保算法在实际使用中的稳定性和可靠性
 */
TEST_F(CudaLinearRegressionTest, NumericalStability)
{
    // 使用随机数据和较大学习率测试数值稳定性
    size_t input_size = 50;
    auto x            = Tensor::randn(Shape{input_size, 1}, dtype(DataType::kFloat32).device(kCUDA));
    auto noise        = Tensor::randn(Shape{input_size, 1}, dtype(DataType::kFloat32).device(kCUDA)) * 0.01f;
    auto y            = x * kExpectedW + kExpectedB + noise;

    // 初始化参数
    auto w = Tensor(0.0f, Shape{1, 1}, dtype(Float32).device(kCUDA));
    auto b = Tensor(0.0f, Shape{1, 1}, dtype(Float32).device(kCUDA));

    float lr  = 0.5f;  // 较大的学习率
    int iters = 100;

    // 训练并检查数值稳定性
    for (int i = 0; i < iters; i++)
    {
        w.clear_grad();
        b.clear_grad();

        auto y_pred = Predict(x, w, b);
        auto loss   = MSE(y, y_pred);
        loss.backward();

        w = w - lr * w.grad();
        b = b - lr * b.grad();

        // 检查参数是否保持有限值
        auto w_data = w.to_vector<float>();
        auto b_data = b.to_vector<float>();
        EXPECT_TRUE(std::isfinite(w_data[0]));
        EXPECT_TRUE(std::isfinite(b_data[0]));
    }

    // 验证最终收敛性
    float final_w = w.to_vector<float>()[0];
    float final_b = b.to_vector<float>()[0];
    EXPECT_NEAR(final_w, kExpectedW, kTolerance * 2);
    EXPECT_NEAR(final_b, kExpectedB, kTolerance * 2);
}
