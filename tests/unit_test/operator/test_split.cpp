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
        auto x       = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{4}, dtype(DataType::kFloat32).device(deviceType()));
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
        auto x       = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));
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
        auto x =
            Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));
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
        auto x       = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{1, 4}, dtype(DataType::kFloat32).device(deviceType()));
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
        auto x       = Tensor::ones(Shape{2, 3, 4}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, {1, 1}, 0);
        EXPECT_EQ(results.size(), 2U);
        EXPECT_EQ(results[0].shape(), Shape({1, 3, 4}));
        EXPECT_EQ(results[1].shape(), Shape({1, 3, 4}));
    }

    // 6. 3D 张量，dim=1
    {
        auto x       = Tensor::ones(Shape{2, 6, 4}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, {2, 2, 2}, 1);
        EXPECT_EQ(results.size(), 3U);
        EXPECT_EQ(results[0].shape(), Shape({2, 2, 4}));
        EXPECT_EQ(results[1].shape(), Shape({2, 2, 4}));
        EXPECT_EQ(results[2].shape(), Shape({2, 2, 4}));
    }

    // 7. 3D 张量，dim=2
    {
        auto x       = Tensor::ones(Shape{2, 3, 6}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, {3, 3}, 2);
        EXPECT_EQ(results.size(), 2U);
        EXPECT_EQ(results[0].shape(), Shape({2, 3, 3}));
        EXPECT_EQ(results[1].shape(), Shape({2, 3, 3}));
    }

    // 8. 4D 张量，dim=0
    {
        auto x       = Tensor::ones(Shape{4, 2, 3, 4}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, {2, 2}, 0);
        EXPECT_EQ(results.size(), 2U);
        EXPECT_EQ(results[0].shape(), Shape({2, 2, 3, 4}));
        EXPECT_EQ(results[1].shape(), Shape({2, 2, 3, 4}));
    }

    // 9. 4D 张量，dim=1
    {
        auto x       = Tensor::ones(Shape{2, 4, 3, 4}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, {1, 1, 2}, 1);
        EXPECT_EQ(results.size(), 3U);
        EXPECT_EQ(results[0].shape(), Shape({2, 1, 3, 4}));
        EXPECT_EQ(results[1].shape(), Shape({2, 1, 3, 4}));
        EXPECT_EQ(results[2].shape(), Shape({2, 2, 3, 4}));
    }

    // 10. 4D 张量，dim=3
    {
        auto x       = Tensor::ones(Shape{2, 3, 4, 6}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, {2, 2, 2}, 3);
        EXPECT_EQ(results.size(), 3U);
        EXPECT_EQ(results[0].shape(), Shape({2, 3, 4, 2}));
        EXPECT_EQ(results[1].shape(), Shape({2, 3, 4, 2}));
        EXPECT_EQ(results[2].shape(), Shape({2, 3, 4, 2}));
    }

    // 11. 单个分割（边界情况）
    {
        auto x       = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
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
        auto x       = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));
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
        auto x       = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{4}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, 2, 0);
        EXPECT_EQ(results.size(), 2U);
        EXPECT_EQ(results[0].shape(), Shape({2}));
        EXPECT_EQ(results[1].shape(), Shape({2}));
    }

    // 3. 2D 张量，dim=1
    {
        auto x       = Tensor::ones(Shape{2, 6}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, 2, 1);
        EXPECT_EQ(results.size(), 3U);
        EXPECT_EQ(results[0].shape(), Shape({2, 2}));
        EXPECT_EQ(results[1].shape(), Shape({2, 2}));
        EXPECT_EQ(results[2].shape(), Shape({2, 2}));
    }

    // 4. 3D 张量，dim=2
    {
        auto x       = Tensor::ones(Shape{2, 3, 8}, dtype(DataType::kFloat32).device(deviceType()));
        auto results = F::split(x, 3, 2);
        EXPECT_EQ(results.size(), 3U);  // 8 / 3 = 2.67，向上取整为 3
        EXPECT_EQ(results[0].shape(), Shape({2, 3, 3}));
        EXPECT_EQ(results[1].shape(), Shape({2, 3, 3}));
        EXPECT_EQ(results[2].shape(), Shape({2, 3, 2}));
    }

    // 5. 4D 张量，dim=3
    {
        auto x       = Tensor::ones(Shape{2, 3, 4, 9}, dtype(DataType::kFloat32).device(deviceType()));
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
        auto x            = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{4},
                                   dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
        auto results      = F::split(x, {2, 2}, 0);
        Tensor sum_result = results[0] + results[1];
        sum_result.backward();
        EXPECT_EQ(x.grad().shape(), x.shape());
        auto expected_grad = Tensor::ones(Shape{4}, dtype(DataType::kFloat32).device(deviceType()));
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, tolerance);
    }

    // 2. 2D 张量，dim=1
    {
        auto x            = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{1, 4},
                                   dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
        auto results      = F::split(x, {2, 2}, 1);
        Tensor sum_result = results[0] + results[1];
        sum_result.backward();
        EXPECT_EQ(x.grad().shape(), x.shape());
    }

    // 3. 按固定大小分割
    {
        auto x            = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, Shape{5},
                                   dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
        auto results      = F::split(x, 2, 0);
        Tensor sum_result = results[0];
        for (size_t i = 1; i < results.size(); ++i)
        {
            sum_result = sum_result + results[i];
        }
        sum_result.backward();
        EXPECT_EQ(x.grad().shape(), x.shape());
    }

    // 4. 3D 张量，dim=1
    {
        auto x       = Tensor::ones(Shape{2, 6, 4}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
        auto results = F::split(x, {2, 2, 2}, 1);
        Tensor sum_result = results[0] + results[1] + results[2];
        sum_result.backward();
        EXPECT_EQ(x.grad().shape(), x.shape());
    }

    // 5. 4D 张量，dim=2
    {
        auto x = Tensor::ones(Shape{2, 3, 6, 4}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
        auto results      = F::split(x, {3, 3}, 2);
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
        auto x       = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{4}, dtype(DataType::kFloat32).device(deviceType()));
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
        auto results                    = F::split(x, split_sizes, 0);
        EXPECT_EQ(results.size(), 2U);
        EXPECT_EQ(results[0].shape(), Shape({2}));
        EXPECT_EQ(results[1].shape(), Shape({2}));
    }

    // 3. C 数组调用
    {
        auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));
        size_t split_sizes[] = {1, 2, 2};
        auto results         = F::split(x, split_sizes, 0);
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

// ==================== 高维张量测试 ====================

TEST_P(SplitOperatorTest, HighDimensional)
{
    // 测试高维张量分割（6维，每个维度都大于1）
    // 形状: [2, 3, 4, 2, 3, 2] 在 dim=2 上分割
    // 输入: [2, 3, 4, 2, 3, 2] (C=4)
    // 输出1: [2, 3, 2, 2, 3, 2] (C=2)
    // 输出2: [2, 3, 2, 2, 3, 2] (C=2)
    // 转换为3维: [M=2*3=6, C=4, N=2*3*2=12] -> [M=6, C=2, N=12] 和 [M=6, C=2, N=12]

    // 创建输入数据：输入有 2*3*4*2*3*2 = 288 个元素
    std::vector<float> input_data(288);
    for (size_t i = 0; i < 288; ++i)
    {
        input_data[i] = static_cast<float>(i);
    }

    auto x = Tensor(input_data, Shape{2, 3, 4, 2, 3, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto results = F::split(x, {2, 2}, 2);

    // 验证输出数量
    EXPECT_EQ(results.size(), 2U);

    // 验证输出形状
    Shape expected_shape1{2, 3, 2, 2, 3, 2};
    Shape expected_shape2{2, 3, 2, 2, 3, 2};
    EXPECT_EQ(results[0].shape(), expected_shape1);
    EXPECT_EQ(results[1].shape(), expected_shape2);

    // 验证输出数据：应该先包含输入的前半部分，然后是后半部分
    // 由于在 dim=2 上分割，每个 chunk (对应 M=2*3=6 个 chunk) 应该先包含前 C=2 个通道，然后是后 C=2 个通道
    auto result0_data = results[0].to_vector<float>();
    auto result1_data = results[1].to_vector<float>();
    EXPECT_EQ(result0_data.size(), 144);  // 2*3*2*2*3*2 = 144
    EXPECT_EQ(result1_data.size(), 144);  // 2*3*2*2*3*2 = 144

    // 完整验证：验证所有元素
    // M=6, C=2, N=12 (2*3*2)
    // 每个 chunk 有 C*N = 2*12 = 24 个元素
    const size_t M = 6;
    const size_t C = 2;
    const size_t N = 12;

    for (size_t m_idx = 0; m_idx < M; ++m_idx)
    {
        size_t input_chunk_start   = m_idx * 4 * N;  // 输入中每个 chunk 有 4*12 = 48 个元素
        size_t result0_chunk_start = m_idx * C * N;  // result0 中每个 chunk 有 2*12 = 24 个元素
        size_t result1_chunk_start = m_idx * C * N;  // result1 中每个 chunk 有 2*12 = 24 个元素

        // 验证 result0 的 24 个元素（输入的前 C=2 个通道）
        for (size_t i = 0; i < C * N; ++i)
        {
            EXPECT_FLOAT_EQ(result0_data[result0_chunk_start + i], static_cast<float>(input_chunk_start + i))
                << "m_idx=" << m_idx << ", result0_chunk_start=" << result0_chunk_start << ", i=" << i;
        }

        // 验证 result1 的 24 个元素（输入的后 C=2 个通道）
        for (size_t i = 0; i < C * N; ++i)
        {
            EXPECT_FLOAT_EQ(result1_data[result1_chunk_start + i], static_cast<float>(input_chunk_start + C * N + i))
                << "m_idx=" << m_idx << ", result1_chunk_start=" << result1_chunk_start << ", i=" << i;
        }
    }
}

TEST_P(SplitOperatorTest, HighDimensionalDim0)
{
    // 测试高维张量在最左边维度（dim=0）上分割（边界测试）
    // 形状: [4, 2, 3, 2, 3, 2] 在 dim=0 上分割
    // 输入: [4, 2, 3, 2, 3, 2] (A=4)
    // 输出1: [2, 2, 3, 2, 3, 2] (A=2)
    // 输出2: [2, 2, 3, 2, 3, 2] (A=2)
    // 转换为3维: [M=1, C=4, N=2*3*2*3*2=72] -> [M=1, C=2, N=72] 和 [M=1, C=2, N=72]

    // 创建输入数据：输入有 4*2*3*2*3*2 = 288 个元素
    std::vector<float> input_data(288);
    for (size_t i = 0; i < 288; ++i)
    {
        input_data[i] = static_cast<float>(i);
    }

    auto x = Tensor(input_data, Shape{4, 2, 3, 2, 3, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto results = F::split(x, {2, 2}, 0);

    // 验证输出形状
    Shape expected_shape{2, 2, 3, 2, 3, 2};
    EXPECT_EQ(results[0].shape(), expected_shape);
    EXPECT_EQ(results[1].shape(), expected_shape);

    // 验证输出数据
    // 在 dim=0 上分割，M=1，所以只有一个 chunk
    // 输出应该先包含输入的前 144 个元素，然后是后 144 个元素
    auto result0_data = results[0].to_vector<float>();
    auto result1_data = results[1].to_vector<float>();
    EXPECT_EQ(result0_data.size(), 144);  // 2*2*3*2*3*2 = 144
    EXPECT_EQ(result1_data.size(), 144);  // 2*2*3*2*3*2 = 144

    // 验证前 144 个元素来自输入的前半部分
    for (size_t i = 0; i < 144; ++i)
    {
        EXPECT_FLOAT_EQ(result0_data[i], static_cast<float>(i));
    }

    // 验证接下来的 144 个元素来自输入的后半部分
    for (size_t i = 0; i < 144; ++i)
    {
        EXPECT_FLOAT_EQ(result1_data[i], static_cast<float>(i + 144));
    }
}

TEST_P(SplitOperatorTest, HighDimensionalDim5)
{
    // 测试高维张量在最右边维度（dim=5）上分割（边界测试）
    // 形状: [2, 3, 2, 3, 2, 4] 在 dim=5 上分割
    // 输入: [2, 3, 2, 3, 2, 4] (F=4)
    // 输出1: [2, 3, 2, 3, 2, 2] (F=2)
    // 输出2: [2, 3, 2, 3, 2, 2] (F=2)
    // 转换为3维: [M=2*3*2*3*2=72, C=4, N=1] -> [M=72, C=2, N=1] 和 [M=72, C=2, N=1]

    // 创建输入数据：输入有 2*3*2*3*2*4 = 288 个元素
    std::vector<float> input_data(288);
    for (size_t i = 0; i < 288; ++i)
    {
        input_data[i] = static_cast<float>(i);
    }

    auto x = Tensor(input_data, Shape{2, 3, 2, 3, 2, 4}, dtype(DataType::kFloat32).device(deviceType()));

    auto results = F::split(x, {2, 2}, 5);

    // 验证输出形状
    Shape expected_shape{2, 3, 2, 3, 2, 2};
    EXPECT_EQ(results[0].shape(), expected_shape);
    EXPECT_EQ(results[1].shape(), expected_shape);

    // 验证输出数据
    // 在 dim=5 上分割，N=1，所以每个 chunk 只有 C 个元素
    // M=72，所以有 72 个 chunk
    // 每个 chunk 先包含输入的前 2 个元素，然后是后 2 个元素
    auto result0_data = results[0].to_vector<float>();
    auto result1_data = results[1].to_vector<float>();
    EXPECT_EQ(result0_data.size(), 144);  // 2*3*2*3*2*2 = 144
    EXPECT_EQ(result1_data.size(), 144);  // 2*3*2*3*2*2 = 144

    // 完整验证：验证所有元素
    // 对于每个 chunk (m_idx = 0 到 71)，每个 chunk 有 2 个元素
    // chunk 内布局: [input[m_idx*4], input[m_idx*4+1]] 和 [input[m_idx*4+2], input[m_idx*4+3]]
    for (size_t m_idx = 0; m_idx < 72; ++m_idx)
    {
        size_t input_chunk_start   = m_idx * 4;  // 输入中每个 chunk 有 4 个元素
        size_t result0_chunk_start = m_idx * 2;  // result0 中每个 chunk 有 2 个元素
        size_t result1_chunk_start = m_idx * 2;  // result1 中每个 chunk 有 2 个元素

        // 验证 result0 的两个元素（输入的前 2 个元素）
        EXPECT_FLOAT_EQ(result0_data[result0_chunk_start], static_cast<float>(input_chunk_start))
            << "m_idx=" << m_idx << ", result0_chunk_start=" << result0_chunk_start;
        EXPECT_FLOAT_EQ(result0_data[result0_chunk_start + 1], static_cast<float>(input_chunk_start + 1))
            << "m_idx=" << m_idx << ", result0_chunk_start=" << result0_chunk_start;

        // 验证 result1 的两个元素（输入的后 2 个元素）
        EXPECT_FLOAT_EQ(result1_data[result1_chunk_start], static_cast<float>(input_chunk_start + 2))
            << "m_idx=" << m_idx << ", result1_chunk_start=" << result1_chunk_start;
        EXPECT_FLOAT_EQ(result1_data[result1_chunk_start + 1], static_cast<float>(input_chunk_start + 3))
            << "m_idx=" << m_idx << ", result1_chunk_start=" << result1_chunk_start;
    }
}

// Instantiate test suite: automatically generate tests for CPU and available CUDA
INSTANTIATE_DEVICE_TEST_SUITE_P(SplitOperatorTest);
