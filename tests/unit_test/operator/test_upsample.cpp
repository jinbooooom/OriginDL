#include <gtest/gtest.h>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"
#include "origin/operators/nn/upsample.h"

using namespace origin;
namespace F = origin::functional;

/**
 * @brief Upsample 算子测试类（参数化版本）
 */
class UpsampleOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 前向传播测试（最近邻） ====================

TEST_P(UpsampleOperatorTest, ForwardNearestBasic)
{
    // mode=nearest：2x2 -> 4x4，每个输入像素复制到 2x2 区域
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};  // 输入 [[1,2],[3,4]]
    auto x                    = Tensor(x_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::upsample(x, "nearest", {2.0f, 2.0f});

    Shape expected_shape{1, 1, 4, 4};
    EXPECT_EQ(result.shape(), expected_shape);

    // 最近邻：输出角与输入一致，每个输入像素填满对应 2x2 块
    // [1, 2]     [1, 1, 2, 2]
    // [3, 4]  -> [1, 1, 2, 2]
    //            [3, 3, 4, 4]
    //            [3, 3, 4, 4]
    std::vector<float> expected_data = {1.0f, 1.0f, 2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 2.0f,
                                        3.0f, 3.0f, 4.0f, 4.0f, 3.0f, 3.0f, 4.0f, 4.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(UpsampleOperatorTest, ForwardNearestScaleFactor)
{
    // mode=nearest：不同缩放因子，仅检查形状
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x                    = Tensor(x_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::upsample(x, "nearest", {3.0f, 3.0f});

    Shape expected_shape{1, 1, 6, 6};
    EXPECT_EQ(result.shape(), expected_shape);
}

TEST_P(UpsampleOperatorTest, ForwardNearestSingleChannel)
{
    // mode=nearest：1x1 -> 2x2，单点复制为 4 个
    std::vector<float> x_data = {1.0f};
    auto x                    = Tensor(x_data, Shape{1, 1, 1, 1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::upsample(x, "nearest", {2.0f, 2.0f});

    Shape expected_shape{1, 1, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    std::vector<float> expected_data = {1.0f, 1.0f, 1.0f, 1.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(UpsampleOperatorTest, ForwardNearestMultiChannel)
{
    // mode=nearest：多通道，形状正确即可
    std::vector<float> x_data = {1.0f, 2.0f};
    auto x                    = Tensor(x_data, Shape{1, 2, 1, 1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::upsample(x, "nearest", {2.0f, 2.0f});

    EXPECT_EQ(result.shape(), Shape({1, 2, 2, 2}));
}

// ==================== 反向传播测试（最近邻） ====================

TEST_P(UpsampleOperatorTest, BackwardNearest)
{
    // mode=nearest：梯度为输出梯度的下采样累加，默认 gy=1 时 gx 每个位置应为 4
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x = Tensor(x_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::upsample(x, "nearest", {2.0f, 2.0f});
    y.backward();

    EXPECT_EQ(x.grad().shape(), x.shape());
    auto gx_data = x.grad().to_vector<float>();
    for (size_t i = 0; i < gx_data.size(); ++i)
        EXPECT_NEAR(gx_data[i], 4.0f, origin::test::TestTolerance::kDefault);
}

// ==================== 边界情况测试 ====================

TEST_P(UpsampleOperatorTest, SingleElement)
{
    // 测试单元素上采样
    std::vector<float> x_data = {5.0f};
    auto x                    = Tensor(x_data, Shape{1, 1, 1, 1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::upsample(x, "nearest", {2.0f, 2.0f});

    Shape expected_shape{1, 1, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    // 所有值都应该是 5.0
    std::vector<float> expected_data = {5.0f, 5.0f, 5.0f, 5.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// Instantiate test suite: automatically generate tests for CPU and available CUDA
INSTANTIATE_DEVICE_TEST_SUITE_P(UpsampleOperatorTest);
