#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../../common/device_test_base.h"
#include "../../common/gtest_utils.h"
#include "../../common/test_utils.h"
#include "origin.h"
#include "origin/utils/metrics.h"

using namespace origin;

/**
 * @brief Accuracy 函数测试类（参数化版本）
 */
class AccuracyTest : public origin::test::OperatorTestBase
{};

// ==================== 基本测试 ====================

TEST_P(AccuracyTest, PerfectAccuracy)
{
    // 测试完美准确率（所有预测都正确）
    // y: [[0.1, 0.9], [0.8, 0.2]] -> argmax: [1, 0]
    // target: [1, 0]
    auto y      = Tensor({0.1f, 0.9f, 0.8f, 0.2f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({1, 0}, Shape{2}, dtype(DataType::kInt32).device(deviceType()));

    auto result = accuracy(y, target);

    Shape expected_shape{};  // 标量
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 1.0f, 0.001f);
}

TEST_P(AccuracyTest, ZeroAccuracy)
{
    // 测试零准确率（所有预测都错误）
    // y: [[0.1, 0.9], [0.8, 0.2]] -> argmax: [1, 0]
    // target: [0, 1]
    auto y      = Tensor({0.1f, 0.9f, 0.8f, 0.2f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({0, 1}, Shape{2}, dtype(DataType::kInt32).device(deviceType()));

    auto result = accuracy(y, target);

    EXPECT_NEAR(result.item<float>(), 0.0f, 0.001f);
}

TEST_P(AccuracyTest, PartialAccuracy)
{
    // 测试部分准确率（一半正确）
    // y: [[0.1, 0.9], [0.8, 0.2]] -> argmax: [1, 0]
    // target: [1, 1]
    auto y      = Tensor({0.1f, 0.9f, 0.8f, 0.2f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({1, 1}, Shape{2}, dtype(DataType::kInt32).device(deviceType()));

    auto result = accuracy(y, target);

    EXPECT_NEAR(result.item<float>(), 0.5f, 0.001f);
}

TEST_P(AccuracyTest, SingleSample)
{
    // 测试单个样本
    auto y      = Tensor({0.1f, 0.9f}, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({1}, Shape{1}, dtype(DataType::kInt32).device(deviceType()));

    auto result = accuracy(y, target);

    EXPECT_NEAR(result.item<float>(), 1.0f, 0.001f);
}

TEST_P(AccuracyTest, ThreeClasses)
{
    // 测试三个类别
    // y: [[0.1, 0.2, 0.7], [0.5, 0.3, 0.2]] -> argmax: [2, 0]
    // target: [2, 0]
    auto y = Tensor({0.1f, 0.2f, 0.7f, 0.5f, 0.3f, 0.2f}, Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({2, 0}, Shape{2}, dtype(DataType::kInt32).device(deviceType()));

    auto result = accuracy(y, target);

    EXPECT_NEAR(result.item<float>(), 1.0f, 0.001f);
}

TEST_P(AccuracyTest, LargeBatch)
{
    // 测试大批量
    size_t N = 10;
    size_t C = 5;
    std::vector<float> y_data(N * C);
    std::vector<int32_t> target_data(N);

    // 创建数据：每个样本的预测类别等于其索引模 C
    for (size_t i = 0; i < N; ++i)
    {
        int32_t pred_class = static_cast<int32_t>(i % C);
        for (size_t j = 0; j < C; ++j)
        {
            // 在预测类别处设置较大的值
            y_data[i * C + j] = (j == pred_class) ? 0.9f : 0.1f;
        }
        target_data[i] = pred_class;  // 目标也设置为相同，所以准确率应该是 1.0
    }

    auto y      = Tensor(y_data, Shape{N, C}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor(target_data, Shape{N}, dtype(DataType::kInt32).device(deviceType()));

    auto result = accuracy(y, target);

    EXPECT_NEAR(result.item<float>(), 1.0f, 0.001f);
}

TEST_P(AccuracyTest, ManyClasses)
{
    // 测试多类别
    size_t C = 10;
    std::vector<float> y_data(C);
    // 在索引 5 处设置最大值
    for (size_t i = 0; i < C; ++i)
    {
        y_data[i] = (i == 5) ? 0.9f : 0.1f;
    }

    auto y      = Tensor(y_data, Shape{1, C}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({5}, Shape{1}, dtype(DataType::kInt32).device(deviceType()));

    auto result = accuracy(y, target);

    EXPECT_NEAR(result.item<float>(), 1.0f, 0.001f);
}

// ==================== 边界情况测试 ====================

TEST_P(AccuracyTest, TieBreaking)
{
    // 测试平局情况（多个类别有相同的最大值）
    // 当有平局时，argmax 应该选择第一个最大值
    // y: [0.5, 0.5, 0.3] -> argmax: 0（第一个最大值）
    auto y      = Tensor({0.5f, 0.5f, 0.3f}, Shape{1, 3}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({0}, Shape{1}, dtype(DataType::kInt32).device(deviceType()));

    auto result = accuracy(y, target);

    EXPECT_NEAR(result.item<float>(), 1.0f, 0.001f);
}

TEST_P(AccuracyTest, SingleClass)
{
    // 测试单类别（虽然不太常见，但应该能处理）
    auto y      = Tensor({1.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({0}, Shape{1}, dtype(DataType::kInt32).device(deviceType()));

    auto result = accuracy(y, target);

    EXPECT_NEAR(result.item<float>(), 1.0f, 0.001f);
}

// ==================== 错误处理测试 ====================

TEST_P(AccuracyTest, ShapeMismatch)
{
    // 测试形状不匹配
    auto y      = Tensor({0.1f, 0.9f}, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({0, 1}, Shape{2}, dtype(DataType::kInt32).device(deviceType()));  // batch size 不匹配

    EXPECT_THROW(accuracy(y, target), std::exception);
}

TEST_P(AccuracyTest, InvalidTargetIndex)
{
    // 测试无效的目标索引（应该仍然计算，但该样本不计入正确数）
    // y: [[0.1, 0.9]] -> argmax: 1
    // target: [5] (超出范围)
    auto y      = Tensor({0.1f, 0.9f}, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto target = Tensor({5}, Shape{1}, dtype(DataType::kInt32).device(deviceType()));  // 索引超出范围

    auto result = accuracy(y, target);

    // 由于目标索引超出范围，该样本不应该被计入正确数，准确率应该是 0
    EXPECT_NEAR(result.item<float>(), 0.0f, 0.001f);
}

// Instantiate test suite: automatically generate tests for CPU and available CUDA
INSTANTIATE_DEVICE_TEST_SUITE_P(AccuracyTest);
