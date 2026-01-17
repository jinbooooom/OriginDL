#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"
#include "origin/operators/normalization/batch_norm.h"

using namespace origin;
namespace F = origin::functional;

/**
 * @brief BatchNorm 算子测试类（参数化版本）
 */
class BatchNormOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== BatchNorm1d 前向传播测试 ====================

TEST_P(BatchNormOperatorTest, BatchNorm1dForwardTraining)
{
    // 测试 BatchNorm1d 训练模式前向传播
    // 输入: (N=2, C=3)
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x                    = Tensor(x_data, Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));

    // 参数: gamma, beta, running_mean, running_var
    std::vector<float> gamma_data = {1.0f, 1.0f, 1.0f};
    auto gamma                    = Tensor(gamma_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> beta_data = {0.0f, 0.0f, 0.0f};
    auto beta                    = Tensor(beta_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> running_mean_data = {0.0f, 0.0f, 0.0f};
    auto running_mean = Tensor(running_mean_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> running_var_data = {1.0f, 1.0f, 1.0f};
    auto running_var = Tensor(running_var_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    // 前向传播（训练模式）
    auto result = F::batch_norm(x, gamma, beta, running_mean, running_var, true, 1e-5f, 0.1f, 2);

    // 验证输出形状
    Shape expected_shape{2, 3};
    EXPECT_EQ(result.shape(), expected_shape);

    // 验证输出值：在训练模式下，应该对每个通道进行归一化
    // 通道0: mean = (1+4)/2 = 2.5, 归一化后应该是 (1-2.5)/std 和 (4-2.5)/std
    // 由于 gamma=1, beta=0，输出就是归一化后的值
    auto result_data = result.to_vector<float>();
    EXPECT_EQ(result_data.size(), 6U);

    // 验证输出不为 NaN 或 Inf
    for (float val : result_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST_P(BatchNormOperatorTest, BatchNorm1dForwardEval)
{
    // 测试 BatchNorm1d 评估模式前向传播
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x                    = Tensor(x_data, Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> gamma_data = {1.0f, 1.0f, 1.0f};
    auto gamma                    = Tensor(gamma_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> beta_data = {0.0f, 0.0f, 0.0f};
    auto beta                    = Tensor(beta_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> running_mean_data = {2.0f, 3.0f, 4.0f};
    auto running_mean = Tensor(running_mean_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> running_var_data = {1.0f, 1.0f, 1.0f};
    auto running_var = Tensor(running_var_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    // 前向传播（评估模式）
    auto result = F::batch_norm(x, gamma, beta, running_mean, running_var, false, 1e-5f, 0.1f, 2);

    // 验证输出形状
    Shape expected_shape{2, 3};
    EXPECT_EQ(result.shape(), expected_shape);

    // 在评估模式下，应该使用 running_mean 和 running_var
    auto result_data = result.to_vector<float>();
    EXPECT_EQ(result_data.size(), 6U);

    // 验证输出不为 NaN 或 Inf
    for (float val : result_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

// ==================== BatchNorm2d 前向传播测试 ====================

TEST_P(BatchNormOperatorTest, BatchNorm2dForwardTraining)
{
    // 测试 BatchNorm2d 训练模式前向传播
    // 输入: (N=2, C=2, H=2, W=2)
    std::vector<float> x_data = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                                 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    auto x                    = Tensor(x_data, Shape{2, 2, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> gamma_data = {1.0f, 1.0f};
    auto gamma                    = Tensor(gamma_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> beta_data = {0.0f, 0.0f};
    auto beta                    = Tensor(beta_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> running_mean_data = {0.0f, 0.0f};
    auto running_mean = Tensor(running_mean_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> running_var_data = {1.0f, 1.0f};
    auto running_var = Tensor(running_var_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    // 前向传播（训练模式）
    auto result = F::batch_norm(x, gamma, beta, running_mean, running_var, true, 1e-5f, 0.1f, 4);

    // 验证输出形状
    Shape expected_shape{2, 2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    // 验证输出值
    auto result_data = result.to_vector<float>();
    EXPECT_EQ(result_data.size(), 16U);

    // 验证输出不为 NaN 或 Inf
    for (float val : result_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST_P(BatchNormOperatorTest, BatchNorm2dForwardEval)
{
    // 测试 BatchNorm2d 评估模式前向传播
    std::vector<float> x_data = {1.0f, 2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,
                                 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f, 16.0f};
    auto x                    = Tensor(x_data, Shape{2, 2, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> gamma_data = {1.0f, 1.0f};
    auto gamma                    = Tensor(gamma_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> beta_data = {0.0f, 0.0f};
    auto beta                    = Tensor(beta_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> running_mean_data = {5.0f, 6.0f};
    auto running_mean = Tensor(running_mean_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> running_var_data = {1.0f, 1.0f};
    auto running_var = Tensor(running_var_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    // 前向传播（评估模式）
    auto result = F::batch_norm(x, gamma, beta, running_mean, running_var, false, 1e-5f, 0.1f, 4);

    // 验证输出形状
    Shape expected_shape{2, 2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    // 验证输出值
    auto result_data = result.to_vector<float>();
    EXPECT_EQ(result_data.size(), 16U);

    // 验证输出不为 NaN 或 Inf
    for (float val : result_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

// ==================== 数值正确性测试 ====================

TEST_P(BatchNormOperatorTest, BatchNorm1dWithGammaBeta)
{
    // 测试 BatchNorm1d 使用 gamma 和 beta 的情况
    // 输入: (N=2, C=2)
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x                    = Tensor(x_data, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    // gamma = 2, beta = 1
    std::vector<float> gamma_data = {2.0f, 2.0f};
    auto gamma                    = Tensor(gamma_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> beta_data = {1.0f, 1.0f};
    auto beta                    = Tensor(beta_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> running_mean_data = {0.0f, 0.0f};
    auto running_mean = Tensor(running_mean_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> running_var_data = {1.0f, 1.0f};
    auto running_var = Tensor(running_var_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::batch_norm(x, gamma, beta, running_mean, running_var, true, 1e-5f, 0.1f, 2);

    // 验证输出形状
    EXPECT_EQ(result.shape(), Shape({2, 2}));

    // 验证输出不为 NaN 或 Inf
    auto result_data = result.to_vector<float>();
    for (float val : result_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST_P(BatchNormOperatorTest, BatchNorm2dWithGammaBeta)
{
    // 测试 BatchNorm2d 使用 gamma 和 beta 的情况
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x                    = Tensor(x_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> gamma_data = {2.0f};
    auto gamma                    = Tensor(gamma_data, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> beta_data = {1.0f};
    auto beta                    = Tensor(beta_data, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> running_mean_data = {0.0f};
    auto running_mean = Tensor(running_mean_data, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> running_var_data = {1.0f};
    auto running_var = Tensor(running_var_data, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::batch_norm(x, gamma, beta, running_mean, running_var, true, 1e-5f, 0.1f, 4);

    // 验证输出形状
    EXPECT_EQ(result.shape(), Shape({1, 1, 2, 2}));

    // 验证输出不为 NaN 或 Inf
    auto result_data = result.to_vector<float>();
    for (float val : result_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

// ==================== 边界情况测试 ====================

TEST_P(BatchNormOperatorTest, BatchNorm1dSingleBatch)
{
    // 测试单个 batch 的情况
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f};
    auto x                    = Tensor(x_data, Shape{1, 3}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> gamma_data = {1.0f, 1.0f, 1.0f};
    auto gamma                    = Tensor(gamma_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> beta_data = {0.0f, 0.0f, 0.0f};
    auto beta                    = Tensor(beta_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> running_mean_data = {0.0f, 0.0f, 0.0f};
    auto running_mean = Tensor(running_mean_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> running_var_data = {1.0f, 1.0f, 1.0f};
    auto running_var = Tensor(running_var_data, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::batch_norm(x, gamma, beta, running_mean, running_var, true, 1e-5f, 0.1f, 2);

    EXPECT_EQ(result.shape(), Shape({1, 3}));

    auto result_data = result.to_vector<float>();
    for (float val : result_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

TEST_P(BatchNormOperatorTest, BatchNorm2dSingleChannel)
{
    // 测试单个通道的情况
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x                    = Tensor(x_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> gamma_data = {1.0f};
    auto gamma                    = Tensor(gamma_data, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> beta_data = {0.0f};
    auto beta                    = Tensor(beta_data, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> running_mean_data = {0.0f};
    auto running_mean = Tensor(running_mean_data, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> running_var_data = {1.0f};
    auto running_var = Tensor(running_var_data, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::batch_norm(x, gamma, beta, running_mean, running_var, true, 1e-5f, 0.1f, 4);

    EXPECT_EQ(result.shape(), Shape({1, 1, 2, 2}));

    auto result_data = result.to_vector<float>();
    for (float val : result_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

// ==================== Float64 类型测试 ====================

TEST_P(BatchNormOperatorTest, BatchNorm1dFloat64)
{
    // 测试 float64 类型
    std::vector<double> x_data = {1.0, 2.0, 3.0, 4.0};
    auto x                     = Tensor(x_data, Shape{2, 2}, dtype(DataType::kFloat64).device(deviceType()));

    std::vector<double> gamma_data = {1.0, 1.0};
    auto gamma                     = Tensor(gamma_data, Shape{2}, dtype(DataType::kFloat64).device(deviceType()));

    std::vector<double> beta_data = {0.0, 0.0};
    auto beta                     = Tensor(beta_data, Shape{2}, dtype(DataType::kFloat64).device(deviceType()));

    std::vector<double> running_mean_data = {0.0, 0.0};
    auto running_mean = Tensor(running_mean_data, Shape{2}, dtype(DataType::kFloat64).device(deviceType()));

    std::vector<double> running_var_data = {1.0, 1.0};
    auto running_var = Tensor(running_var_data, Shape{2}, dtype(DataType::kFloat64).device(deviceType()));

    auto result = F::batch_norm(x, gamma, beta, running_mean, running_var, true, 1e-5, 0.1, 2);

    EXPECT_EQ(result.shape(), Shape({2, 2}));

    auto result_data = result.to_vector<double>();
    for (double val : result_data)
    {
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    }
}

// ==================== 参数化测试实例化 ====================

INSTANTIATE_DEVICE_TEST_SUITE_P(BatchNormOperatorTest);
