#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../../common/device_test_base.h"
#include "../../common/gtest_utils.h"
#include "../../common/test_utils.h"
#include "origin.h"

using namespace origin;

/**
 * @brief SoftmaxCrossEntropy 算子测试类（参数化版本）
 */
class SoftmaxCrossEntropyOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 前向传播测试 ====================

TEST_P(SoftmaxCrossEntropyOperatorTest, ForwardBasic)
{
    // 测试基本 softmax 交叉熵运算
    // 输入: x = [[1.0, 2.0, 3.0]], target = [2]
    // softmax(x) ≈ [0.090, 0.245, 0.665]
    // loss = -log(0.665) ≈ 0.408
    auto x      = Tensor({1.0f, 2.0f, 3.0f}, Shape{1, 3}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({2}, Shape{1}, dtype(DataType::kInt32).device(deviceType()));

    auto result = softmax_cross_entropy(x, target);

    Shape expected_shape{};  // 标量
    EXPECT_EQ(result.shape(), expected_shape);

    // 验证损失值在合理范围内（应该大于 0）
    float loss_value = result.item<float>();
    EXPECT_GT(loss_value, 0.0f);
    EXPECT_LT(loss_value, 10.0f);  // 合理的上界
}

TEST_P(SoftmaxCrossEntropyOperatorTest, ForwardPerfectPrediction)
{
    // 测试完美预测的情况
    // 如果 softmax 输出在正确类别上为 1.0，损失应该接近 0
    // 使用很大的 logits 值来近似完美预测
    auto x      = Tensor({-10.0f, -10.0f, 10.0f}, Shape{1, 3}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({2}, Shape{1}, dtype(DataType::kInt32).device(deviceType()));

    auto result = softmax_cross_entropy(x, target);

    float loss_value = result.item<float>();
    EXPECT_NEAR(loss_value, 0.0f, 0.1f);  // 应该接近 0
}

TEST_P(SoftmaxCrossEntropyOperatorTest, ForwardBatch)
{
    // 测试批处理
    // x: (2, 3), target: (2,)
    auto x = Tensor({1.0f, 2.0f, 3.0f, 3.0f, 2.0f, 1.0f}, Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({2, 0}, Shape{2}, dtype(DataType::kInt32).device(deviceType()));

    auto result = softmax_cross_entropy(x, target);

    Shape expected_shape{};  // 标量
    EXPECT_EQ(result.shape(), expected_shape);

    float loss_value = result.item<float>();
    EXPECT_GT(loss_value, 0.0f);
    EXPECT_LT(loss_value, 10.0f);
}

TEST_P(SoftmaxCrossEntropyOperatorTest, ForwardUniformDistribution)
{
    // 测试均匀分布的情况
    // 如果所有 logits 相等，softmax 输出均匀分布，损失应该接近 log(C)
    auto x      = Tensor({1.0f, 1.0f, 1.0f}, Shape{1, 3}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({1}, Shape{1}, dtype(DataType::kInt32).device(deviceType()));

    auto result = softmax_cross_entropy(x, target);

    float loss_value    = result.item<float>();
    float expected_loss = std::log(3.0f);  // -log(1/3) = log(3)
    EXPECT_NEAR(loss_value, expected_loss, 0.01f);
}

// ==================== 反向传播测试 ====================

TEST_P(SoftmaxCrossEntropyOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x =
        Tensor({1.0f, 2.0f, 3.0f}, Shape{1, 3}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    auto target = Tensor({2}, Shape{1}, dtype(DataType::kInt32).device(deviceType()));

    auto loss = softmax_cross_entropy(x, target);
    loss.backward();

    // 验证梯度形状
    EXPECT_EQ(x.grad().shape(), x.shape());

    // 验证梯度值在合理范围内
    auto grad_data = x.grad().to_vector<float>();
    for (float g : grad_data)
    {
        EXPECT_GE(g, -1.0f);
        EXPECT_LE(g, 1.0f);
    }
}

TEST_P(SoftmaxCrossEntropyOperatorTest, BackwardBatch)
{
    // 测试批处理的反向传播
    auto x      = Tensor({1.0f, 2.0f, 3.0f, 3.0f, 2.0f, 1.0f}, Shape{2, 3},
                         dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    auto target = Tensor({2, 0}, Shape{2}, dtype(DataType::kInt32).device(deviceType()));

    auto loss = softmax_cross_entropy(x, target);
    loss.backward();

    // 验证梯度形状
    EXPECT_EQ(x.grad().shape(), x.shape());

    // 验证梯度值在合理范围内
    auto grad_data = x.grad().to_vector<float>();
    for (float g : grad_data)
    {
        EXPECT_GE(g, -1.0f);
        EXPECT_LE(g, 1.0f);
    }
}

TEST_P(SoftmaxCrossEntropyOperatorTest, BackwardGradientCheck)
{
    // 测试梯度数值正确性
    // 对于 softmax_cross_entropy，梯度应该是 (softmax(x) - one_hot(target)) / N
    auto x =
        Tensor({1.0f, 2.0f, 3.0f}, Shape{1, 3}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    auto target = Tensor({2}, Shape{1}, dtype(DataType::kInt32).device(deviceType()));

    auto loss = softmax_cross_entropy(x, target);
    loss.backward();

    // 计算 softmax
    auto p      = softmax(x, -1);
    auto p_data = p.to_vector<float>();

    // 创建 one_hot
    std::vector<float> one_hot = {0.0f, 0.0f, 1.0f};

    // 计算期望梯度: (softmax(x) - one_hot) / N
    std::vector<float> expected_grad(3);
    for (size_t i = 0; i < 3; ++i)
    {
        expected_grad[i] = (p_data[i] - one_hot[i]) / 1.0f;  // N = 1
    }

    auto grad_data = x.grad().to_vector<float>();
    for (size_t i = 0; i < 3; ++i)
    {
        EXPECT_NEAR(grad_data[i], expected_grad[i], 0.01f);
    }
}

// ==================== 边界情况测试 ====================

TEST_P(SoftmaxCrossEntropyOperatorTest, SingleClass)
{
    // 测试单类别（虽然不太常见，但应该能处理）
    auto x      = Tensor({1.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({0}, Shape{1}, dtype(DataType::kInt32).device(deviceType()));

    auto result = softmax_cross_entropy(x, target);

    // 对于单类别，softmax 输出为 1.0，损失为 -log(1) = 0
    float loss_value = result.item<float>();
    EXPECT_NEAR(loss_value, 0.0f, 0.01f);
}

TEST_P(SoftmaxCrossEntropyOperatorTest, LargeBatch)
{
    // 测试大批量
    size_t N = 10;
    size_t C = 5;
    std::vector<float> x_data(N * C);
    std::vector<int32_t> target_data(N);

    for (size_t i = 0; i < N; ++i)
    {
        for (size_t j = 0; j < C; ++j)
        {
            x_data[i * C + j] = static_cast<float>(j);
        }
        target_data[i] = static_cast<int32_t>(i % C);
    }

    auto x      = Tensor(x_data, Shape{N, C}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor(target_data, Shape{N}, dtype(DataType::kInt32).device(deviceType()));

    auto result = softmax_cross_entropy(x, target);

    float loss_value = result.item<float>();
    EXPECT_GT(loss_value, 0.0f);
    EXPECT_LT(loss_value, 10.0f);
}

TEST_P(SoftmaxCrossEntropyOperatorTest, ManyClasses)
{
    // 测试多类别
    size_t C = 10;
    std::vector<float> x_data(C);
    for (size_t i = 0; i < C; ++i)
    {
        x_data[i] = static_cast<float>(i);
    }

    auto x      = Tensor(x_data, Shape{1, C}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({5}, Shape{1}, dtype(DataType::kInt32).device(deviceType()));

    auto result = softmax_cross_entropy(x, target);

    float loss_value = result.item<float>();
    EXPECT_GT(loss_value, 0.0f);
    EXPECT_LT(loss_value, 10.0f);
}

// ==================== 错误处理测试 ====================

TEST_P(SoftmaxCrossEntropyOperatorTest, InvalidTargetIndex)
{
    // 测试无效的目标索引
    auto x      = Tensor({1.0f, 2.0f, 3.0f}, Shape{1, 3}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({5}, Shape{1}, dtype(DataType::kInt32).device(deviceType()));  // 索引超出范围

    EXPECT_THROW(softmax_cross_entropy(x, target), std::exception);
}

TEST_P(SoftmaxCrossEntropyOperatorTest, ShapeMismatch)
{
    // 测试形状不匹配
    auto x      = Tensor({1.0f, 2.0f, 3.0f}, Shape{1, 3}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({0, 1}, Shape{2}, dtype(DataType::kInt32).device(deviceType()));  // batch size 不匹配

    EXPECT_THROW(softmax_cross_entropy(x, target), std::exception);
}

// Instantiate test suite: automatically generate tests for CPU and available CUDA
INSTANTIATE_DEVICE_TEST_SUITE_P(SoftmaxCrossEntropyOperatorTest);
