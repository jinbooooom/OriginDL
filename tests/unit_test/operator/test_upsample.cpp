#include <gtest/gtest.h>
#include <vector>
#include "origin.h"
#include "origin/operators/nn/upsample.h"
#include "../../common/device_test_base.h"
#include "../../common/gtest_utils.h"
#include "../../common/test_utils.h"

using namespace origin;
namespace F = origin::functional;

/**
 * @brief Upsample 算子测试类（参数化版本）
 */
class UpsampleOperatorTest : public origin::test::OperatorTestBase
{
};

// ==================== 前向传播测试 ====================

TEST_P(UpsampleOperatorTest, ForwardBasic)
{
    // 测试基本上采样：2x2 -> 4x4 (scale_factor=2)
    // 输入: (1, 1, 2, 2)
    // 输出: (1, 1, 4, 4)
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x = Tensor(x_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::upsample(x, "nearest", {2.0f, 2.0f});

    Shape expected_shape{1, 1, 4, 4};
    EXPECT_EQ(result.shape(), expected_shape);
    
    // 最近邻上采样：每个像素复制到 2x2 区域
    // [1, 2]     [1, 1, 2, 2]
    // [3, 4]  -> [1, 1, 2, 2]
    //            [3, 3, 4, 4]
    //            [3, 3, 4, 4]
    std::vector<float> expected_data = {
        1.0f, 1.0f, 2.0f, 2.0f,
        1.0f, 1.0f, 2.0f, 2.0f,
        3.0f, 3.0f, 4.0f, 4.0f,
        3.0f, 3.0f, 4.0f, 4.0f
    };
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(UpsampleOperatorTest, ForwardScaleFactor)
{
    // 测试不同的缩放因子
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x = Tensor(x_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::upsample(x, "nearest", {3.0f, 3.0f});

    // 2 * 3 = 6
    Shape expected_shape{1, 1, 6, 6};
    EXPECT_EQ(result.shape(), expected_shape);
}

TEST_P(UpsampleOperatorTest, ForwardSingleChannel)
{
    // 测试单通道上采样
    std::vector<float> x_data = {1.0f};
    auto x = Tensor(x_data, Shape{1, 1, 1, 1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::upsample(x, "nearest", {2.0f, 2.0f});

    Shape expected_shape{1, 1, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    
    // 所有值都应该是 1.0
    std::vector<float> expected_data = {1.0f, 1.0f, 1.0f, 1.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(UpsampleOperatorTest, ForwardMultiChannel)
{
    // 测试多通道上采样
    // 输入: (1, 2, 1, 1) - 1个batch，2个通道，1x1图像
    std::vector<float> x_data = {
        1.0f,  // 通道1
        2.0f   // 通道2
    };
    auto x = Tensor(x_data, Shape{1, 2, 1, 1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::upsample(x, "nearest", {2.0f, 2.0f});

    Shape expected_shape{1, 2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
}

// ==================== 反向传播测试 ====================

TEST_P(UpsampleOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x = Tensor(x_data, Shape{1, 1, 2, 2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::upsample(x, "nearest", {2.0f, 2.0f});
    y.backward();

    // 验证梯度形状
    EXPECT_EQ(x.grad().shape(), x.shape());
    
    // 对于最近邻上采样，梯度应该被下采样回原始形状
    // 每个输入像素的梯度是它对应的所有输出像素梯度的和
    auto gx_data = x.grad().to_vector<float>();
    // 由于输出梯度是全1（默认），每个输入像素的梯度应该是 4（对应 2x2 区域）
    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], 4.0f, origin::test::TestTolerance::kDefault);
    }
}

// ==================== 边界情况测试 ====================

TEST_P(UpsampleOperatorTest, SingleElement)
{
    // 测试单元素上采样
    std::vector<float> x_data = {5.0f};
    auto x = Tensor(x_data, Shape{1, 1, 1, 1}, dtype(DataType::kFloat32).device(deviceType()));

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

