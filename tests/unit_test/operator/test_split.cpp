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

TEST_P(SplitOperatorTest, ForwardSplitSizes)
{
    // 测试各种按大小列表分割的场景
    auto tolerance = origin::test::TestTolerance::kDefault;
    
    // 1. 1D 张量，dim=0，等分
    {
        auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{4}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, {2, 2}, 0);
        EXPECT_EQ(results.size(), 2U);
        EXPECT_EQ(results[0].shape(), Shape({2}));
        EXPECT_EQ(results[1].shape(), Shape({2}));
        auto expected0 = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
        auto expected1 = Tensor({3.0f, 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[0], expected0, tolerance);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[1], expected1, tolerance);
    }
    
    // 2. 1D 张量，多个不等分
    {
        auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, {1, 2, 2}, 0);
        EXPECT_EQ(results.size(), 3U);
        EXPECT_EQ(results[0].shape(), Shape({1}));
        EXPECT_EQ(results[1].shape(), Shape({2}));
        EXPECT_EQ(results[2].shape(), Shape({2}));
        auto expected0 = Tensor({1.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
        auto expected1 = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
        auto expected2 = Tensor({4.0f, 5.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[0], expected0, tolerance);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[1], expected1, tolerance);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[2], expected2, tolerance);
    }
    
    // 3. 2D 张量，dim=0
    {
        auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, {1, 1}, 0);
        EXPECT_EQ(results.size(), 2U);
        EXPECT_EQ(results[0].shape(), Shape({1, 3}));
        EXPECT_EQ(results[1].shape(), Shape({1, 3}));
        // 验证数据内容：第一行和第二行
        auto expected0 = Tensor({1.0f, 2.0f, 3.0f}, Shape{1, 3}, dtype(DataType::kFloat32).device(deviceType()));
        auto expected1 = Tensor({4.0f, 5.0f, 6.0f}, Shape{1, 3}, dtype(DataType::kFloat32).device(deviceType()));
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[0], expected0, tolerance);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[1], expected1, tolerance);
    }
    
    // 4. 2D 张量，dim=1
    {
        auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{1, 4}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, {2, 2}, 1);
        EXPECT_EQ(results.size(), 2U);
        EXPECT_EQ(results[0].shape(), Shape({1, 2}));
        EXPECT_EQ(results[1].shape(), Shape({1, 2}));
        auto expected0 = Tensor({1.0f, 2.0f}, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()));
        auto expected1 = Tensor({3.0f, 4.0f}, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()));
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[0], expected0, tolerance);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[1], expected1, tolerance);
    }
    
    // 5. 3D 张量，dim=0
    {
        auto x = Tensor::ones(Shape{2, 3, 4}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, {1, 1}, 0);
        EXPECT_EQ(results.size(), 2U);
        EXPECT_EQ(results[0].shape(), Shape({1, 3, 4}));
        EXPECT_EQ(results[1].shape(), Shape({1, 3, 4}));
    }
    
    // 6. 3D 张量，dim=1
    {
        auto x = Tensor::ones(Shape{2, 6, 4}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, {2, 2, 2}, 1);
        EXPECT_EQ(results.size(), 3U);
        EXPECT_EQ(results[0].shape(), Shape({2, 2, 4}));
        EXPECT_EQ(results[1].shape(), Shape({2, 2, 4}));
        EXPECT_EQ(results[2].shape(), Shape({2, 2, 4}));
    }
    
    // 7. 3D 张量，dim=2
    {
        auto x = Tensor::ones(Shape{2, 3, 6}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, {3, 3}, 2);
        EXPECT_EQ(results.size(), 2U);
        EXPECT_EQ(results[0].shape(), Shape({2, 3, 3}));
        EXPECT_EQ(results[1].shape(), Shape({2, 3, 3}));
    }
    
    // 8. 4D 张量，dim=0
    {
        auto x = Tensor::ones(Shape{4, 2, 3, 4}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, {2, 2}, 0);
        EXPECT_EQ(results.size(), 2U);
        EXPECT_EQ(results[0].shape(), Shape({2, 2, 3, 4}));
        EXPECT_EQ(results[1].shape(), Shape({2, 2, 3, 4}));
    }
    
    // 9. 4D 张量，dim=1
    {
        auto x = Tensor::ones(Shape{2, 4, 3, 4}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, {1, 1, 2}, 1);
        EXPECT_EQ(results.size(), 3U);
        EXPECT_EQ(results[0].shape(), Shape({2, 1, 3, 4}));
        EXPECT_EQ(results[1].shape(), Shape({2, 1, 3, 4}));
        EXPECT_EQ(results[2].shape(), Shape({2, 2, 3, 4}));
    }
    
    // 10. 4D 张量，dim=3
    {
        auto x = Tensor::ones(Shape{2, 3, 4, 6}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, {2, 2, 2}, 3);
        EXPECT_EQ(results.size(), 3U);
        EXPECT_EQ(results[0].shape(), Shape({2, 3, 4, 2}));
        EXPECT_EQ(results[1].shape(), Shape({2, 3, 4, 2}));
        EXPECT_EQ(results[2].shape(), Shape({2, 3, 4, 2}));
    }
    
    // 11. 单个分割（边界情况）
    {
        auto x = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, {3}, 0);
        EXPECT_EQ(results.size(), 1U);
        EXPECT_EQ(results[0].shape(), Shape({3}));
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[0], x, tolerance);
    }
}

// ==================== 前向传播测试：按固定大小分割 ====================

TEST_P(SplitOperatorTest, ForwardFixedSize)
{
    // 测试按固定大小分割的各种场景
    auto tolerance = origin::test::TestTolerance::kDefault;
    
    // 1. 1D 张量，不能整除
    {
        auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, 2, 0);
        EXPECT_EQ(results.size(), 3U);  // 5 / 2 = 2.5，向上取整为 3
        EXPECT_EQ(results[0].shape(), Shape({2}));
        EXPECT_EQ(results[1].shape(), Shape({2}));
        EXPECT_EQ(results[2].shape(), Shape({1}));
        auto expected0 = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
        auto expected1 = Tensor({3.0f, 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
        auto expected2 = Tensor({5.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[0], expected0, tolerance);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[1], expected1, tolerance);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[2], expected2, tolerance);
    }
    
    // 2. 1D 张量，正好整除
    {
        auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{4}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, 2, 0);
        EXPECT_EQ(results.size(), 2U);
        EXPECT_EQ(results[0].shape(), Shape({2}));
        EXPECT_EQ(results[1].shape(), Shape({2}));
    }
    
    // 3. 2D 张量，dim=1
    {
        auto x = Tensor::ones(Shape{2, 6}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, 2, 1);
        EXPECT_EQ(results.size(), 3U);
        EXPECT_EQ(results[0].shape(), Shape({2, 2}));
        EXPECT_EQ(results[1].shape(), Shape({2, 2}));
        EXPECT_EQ(results[2].shape(), Shape({2, 2}));
    }
    
    // 4. 3D 张量，dim=2
    {
        auto x = Tensor::ones(Shape{2, 3, 8}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, 3, 2);
        EXPECT_EQ(results.size(), 3U);  // 8 / 3 = 2.67，向上取整为 3
        EXPECT_EQ(results[0].shape(), Shape({2, 3, 3}));
        EXPECT_EQ(results[1].shape(), Shape({2, 3, 3}));
        EXPECT_EQ(results[2].shape(), Shape({2, 3, 2}));
    }
    
    // 5. 4D 张量，dim=3
    {
        auto x = Tensor::ones(Shape{2, 3, 4, 9}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, 4, 3);
        EXPECT_EQ(results.size(), 3U);  // 9 / 4 = 2.25，向上取整为 3
        EXPECT_EQ(results[0].shape(), Shape({2, 3, 4, 4}));
        EXPECT_EQ(results[1].shape(), Shape({2, 3, 4, 4}));
        EXPECT_EQ(results[2].shape(), Shape({2, 3, 4, 1}));
    }
}

// ==================== 反向传播测试 ====================

TEST_P(SplitOperatorTest, Backward)
{
    // 测试反向传播的各种场景
    auto tolerance = origin::test::TestTolerance::kDefault;
    
    // 1. 1D 张量，按大小列表分割
    {
        auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{4}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
        auto results = F::split(x, {2, 2}, 0);
        Tensor sum_result = results[0] + results[1];
        sum_result.backward();
        EXPECT_EQ(x.grad().shape(), x.shape());
        auto expected_grad = Tensor::ones(Shape{4}, dtype(DataType::kFloat32).device(deviceType()));
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, tolerance);
    }
    
    // 2. 2D 张量，dim=1
    {
        auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{1, 4}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
        auto results = F::split(x, {2, 2}, 1);
        Tensor sum_result = results[0] + results[1];
        sum_result.backward();
        EXPECT_EQ(x.grad().shape(), x.shape());
    }
    
    // 3. 按固定大小分割
    {
        auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, Shape{5}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
        auto results = F::split(x, 2, 0);
        Tensor sum_result = results[0];
        for (size_t i = 1; i < results.size(); ++i) {
            sum_result = sum_result + results[i];
        }
        sum_result.backward();
        EXPECT_EQ(x.grad().shape(), x.shape());
    }
    
    // 4. 3D 张量，dim=1
    {
        auto x = Tensor::ones(Shape{2, 6, 4}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
        auto results = F::split(x, {2, 2, 2}, 1);
        Tensor sum_result = results[0] + results[1] + results[2];
        sum_result.backward();
        EXPECT_EQ(x.grad().shape(), x.shape());
    }
    
    // 5. 4D 张量，dim=2
    {
        auto x = Tensor::ones(Shape{2, 3, 6, 4}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
        auto results = F::split(x, {3, 3}, 2);
        Tensor sum_result = results[0] + results[1];
        sum_result.backward();
        EXPECT_EQ(x.grad().shape(), x.shape());
    }
}

// ==================== 测试不同的调用方式 ====================

TEST_P(SplitOperatorTest, DifferentCallStyles)
{
    // 测试不同的调用方式：初始化列表、vector、C 数组
    auto tolerance = origin::test::TestTolerance::kDefault;
    
    // 1. 初始化列表调用
    {
        auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{4}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, {2, 2}, 0);
        EXPECT_EQ(results.size(), 2U);
        EXPECT_EQ(results[0].shape(), Shape({2}));
        EXPECT_EQ(results[1].shape(), Shape({2}));
        auto expected0 = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
        auto expected1 = Tensor({3.0f, 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[0], expected0, tolerance);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[1], expected1, tolerance);
    }
    
    // 2. std::vector<size_t> 调用
    {
        auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{4}, dtype(DataType::kFloat32).device(deviceType()));
        std::vector<size_t> split_sizes = {2, 2};
        auto results = F::split(x, split_sizes, 0);
        EXPECT_EQ(results.size(), 2U);
        EXPECT_EQ(results[0].shape(), Shape({2}));
        EXPECT_EQ(results[1].shape(), Shape({2}));
    }
    
    // 3. C 数组调用
    {
        auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));
        size_t split_sizes[] = {1, 2, 2};
        auto results = F::split(x, split_sizes, 0);
        EXPECT_EQ(results.size(), 3U);
        EXPECT_EQ(results[0].shape(), Shape({1}));
        EXPECT_EQ(results[1].shape(), Shape({2}));
        EXPECT_EQ(results[2].shape(), Shape({2}));
        auto expected0 = Tensor({1.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
        auto expected1 = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
        auto expected2 = Tensor({4.0f, 5.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[0], expected0, tolerance);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[1], expected1, tolerance);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(results[2], expected2, tolerance);
    }
}

// Instantiate test suite: automatically generate tests for CPU and available CUDA
INSTANTIATE_DEVICE_TEST_SUITE_P(SplitOperatorTest);
