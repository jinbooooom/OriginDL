#include <gtest/gtest.h>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"
#include "origin/operators/shape/split.h"

using namespace origin;
namespace F = origin::functional;

/**
 * @brief Split 算子测试类（参数化版本）
 */
class SplitOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 前向传播测试：按大小列表分割 ====================

TEST_P(SplitOperatorTest, ForwardBasic)
{
    // 测试基本分割：在 dim=0 上按大小列表分割
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{4}, dtype(DataType::kFloat32).device(deviceType()));

    auto results = F::split(x, {2, 2}, 0);

    EXPECT_EQ(results.size(), 2U);
    EXPECT_EQ(results[0].shape(), Shape({2}));
    EXPECT_EQ(results[1].shape(), Shape({2}));

    std::vector<float> expected_data0 = {1.0f, 2.0f};
    std::vector<float> expected_data1 = {3.0f, 4.0f};
    auto expected0 = Tensor(expected_data0, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto expected1 = Tensor(expected_data1, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[0], expected0, origin::test::TestTolerance::kDefault);
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[1], expected1, origin::test::TestTolerance::kDefault);
}

TEST_P(SplitOperatorTest, ForwardDim1)
{
    // 测试在 dim=1 上分割
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{1, 4}, dtype(DataType::kFloat32).device(deviceType()));

    auto results = F::split(x, {2, 2}, 1);

    EXPECT_EQ(results.size(), 2U);
    EXPECT_EQ(results[0].shape(), Shape({1, 2}));
    EXPECT_EQ(results[1].shape(), Shape({1, 2}));

    std::vector<float> expected_data0 = {1.0f, 2.0f};
    std::vector<float> expected_data1 = {3.0f, 4.0f};
    auto expected0 = Tensor(expected_data0, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto expected1 = Tensor(expected_data1, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[0], expected0, origin::test::TestTolerance::kDefault);
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[1], expected1, origin::test::TestTolerance::kDefault);
}

TEST_P(SplitOperatorTest, ForwardMultipleSplits)
{
    // 测试分割成多个部分
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));

    auto results = F::split(x, {1, 2, 2}, 0);

    EXPECT_EQ(results.size(), 3U);
    EXPECT_EQ(results[0].shape(), Shape({1}));
    EXPECT_EQ(results[1].shape(), Shape({2}));
    EXPECT_EQ(results[2].shape(), Shape({2}));

    std::vector<float> expected_data0 = {1.0f};
    std::vector<float> expected_data1 = {2.0f, 3.0f};
    std::vector<float> expected_data2 = {4.0f, 5.0f};
    auto expected0 = Tensor(expected_data0, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    auto expected1 = Tensor(expected_data1, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto expected2 = Tensor(expected_data2, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[0], expected0, origin::test::TestTolerance::kDefault);
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[1], expected1, origin::test::TestTolerance::kDefault);
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[2], expected2, origin::test::TestTolerance::kDefault);
}

TEST_P(SplitOperatorTest, ForwardTwoDimensional)
{
    // 测试 2D 张量分割
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));

    // 在 dim=0 上分割
    auto results0 = F::split(x, {1, 1}, 0);
    EXPECT_EQ(results0.size(), 2U);
    EXPECT_EQ(results0[0].shape(), Shape({1, 3}));
    EXPECT_EQ(results0[1].shape(), Shape({1, 3}));

    // 在 dim=1 上分割
    auto results1 = F::split(x, {1, 2}, 1);
    EXPECT_EQ(results1.size(), 2U);
    EXPECT_EQ(results1[0].shape(), Shape({2, 1}));
    EXPECT_EQ(results1[1].shape(), Shape({2, 2}));
}

// ==================== 前向传播测试：按固定大小分割 ====================

TEST_P(SplitOperatorTest, ForwardFixedSize)
{
    // 测试按固定大小分割
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));

    auto results = F::split(x, 2, 0);

    EXPECT_EQ(results.size(), 3U);  // 5 / 2 = 2.5，向上取整为 3
    EXPECT_EQ(results[0].shape(), Shape({2}));
    EXPECT_EQ(results[1].shape(), Shape({2}));
    EXPECT_EQ(results[2].shape(), Shape({1}));  // 最后一个可能不完整

    std::vector<float> expected_data0 = {1.0f, 2.0f};
    std::vector<float> expected_data1 = {3.0f, 4.0f};
    std::vector<float> expected_data2 = {5.0f};
    auto expected0 = Tensor(expected_data0, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto expected1 = Tensor(expected_data1, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto expected2 = Tensor(expected_data2, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[0], expected0, origin::test::TestTolerance::kDefault);
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[1], expected1, origin::test::TestTolerance::kDefault);
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[2], expected2, origin::test::TestTolerance::kDefault);
}

TEST_P(SplitOperatorTest, ForwardFixedSizeExact)
{
    // 测试按固定大小分割（正好整除）
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{4}, dtype(DataType::kFloat32).device(deviceType()));

    auto results = F::split(x, 2, 0);

    EXPECT_EQ(results.size(), 2U);
    EXPECT_EQ(results[0].shape(), Shape({2}));
    EXPECT_EQ(results[1].shape(), Shape({2}));
}

// ==================== 反向传播测试 ====================

TEST_P(SplitOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{4}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto results = F::split(x, {2, 2}, 0);
    
    // 对每个结果求和并反向传播
    Tensor sum_result = results[0] + results[1];
    sum_result.backward();

    // 验证梯度形状
    EXPECT_EQ(x.grad().shape(), x.shape());

    // 梯度应该是全1（因为每个元素都被使用了一次）
    auto expected_grad = Tensor::ones(Shape{4}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(SplitOperatorTest, BackwardDim1)
{
    // 测试在 dim=1 上的反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{1, 4}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto results = F::split(x, {2, 2}, 1);
    
    Tensor sum_result = results[0] + results[1];
    sum_result.backward();

    // 验证梯度形状
    EXPECT_EQ(x.grad().shape(), x.shape());
}

TEST_P(SplitOperatorTest, BackwardFixedSize)
{
    // 测试按固定大小分割的反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, Shape{5}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto results = F::split(x, 2, 0);
    
    // 对所有结果求和
    Tensor sum_result = results[0];
    for (size_t i = 1; i < results.size(); ++i)
    {
        sum_result = sum_result + results[i];
    }
    sum_result.backward();

    // 验证梯度形状
    EXPECT_EQ(x.grad().shape(), x.shape());
}

// ==================== 边界情况测试 ====================

TEST_P(SplitOperatorTest, SingleSplit)
{
    // 测试分割成单个部分
    auto x = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto results = F::split(x, {3}, 0);

    EXPECT_EQ(results.size(), 1U);
    EXPECT_EQ(results[0].shape(), Shape({3}));
    
    // 应该和输入相同
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[0], x, origin::test::TestTolerance::kDefault);
}

TEST_P(SplitOperatorTest, UnequalSplits)
{
    // 测试不等大小的分割
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));

    auto results = F::split(x, {1, 3, 1}, 0);

    EXPECT_EQ(results.size(), 3U);
    EXPECT_EQ(results[0].shape(), Shape({1}));
    EXPECT_EQ(results[1].shape(), Shape({3}));
    EXPECT_EQ(results[2].shape(), Shape({1}));
}

// Instantiate test suite: automatically generate tests for CPU and available CUDA
INSTANTIATE_DEVICE_TEST_SUITE_P(SplitOperatorTest);
