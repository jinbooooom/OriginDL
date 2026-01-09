#include <gtest/gtest.h>
#include <vector>
#include "origin.h"
#include "origin/operators/custom/linear.h"
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"

using namespace origin;
namespace F = origin::functional;

/**
 * @brief Linear 算子测试类（参数化版本）
 */
class LinearOperatorTest : public origin::test::OperatorTestBase
{
};

// ==================== 前向传播测试 ====================

TEST_P(LinearOperatorTest, ForwardBasic)
{
    // 测试基本线性层操作
    // 输入: (2, 3) - batch_size=2, in_features=3
    // 权重: (4, 3) - out_features=4, in_features=3
    // 偏置: (4,)
    // 期望输出: (2, 4)

    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x = Tensor(x_data, Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> weight_data = {1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    auto weight = Tensor(weight_data, Shape{4, 3}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> bias_data = {0.0f, 0.0f, 0.0f, 0.0f};
    auto bias = Tensor(bias_data, Shape{4}, dtype(DataType::kFloat32).device(deviceType()));

    functional::LinearOp op(3, 4, true);
    auto result = op.forward({x, weight, bias})[0];

    Shape expected_shape{2, 4};
    EXPECT_EQ(result.shape(), expected_shape);

    // 手动计算期望结果：
    // x * weight^T = [[1,2,3], [4,5,6]] * [[1,0,0,1], [0,1,0,1], [0,0,1,1]]
    // = [[1,2,3,6], [4,5,6,15]]
    std::vector<float> expected_data = {1.0f, 2.0f, 3.0f, 6.0f, 4.0f, 5.0f, 6.0f, 15.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(LinearOperatorTest, ForwardWithBias)
{
    // 测试带偏置的线性层
    std::vector<float> x_data = {1.0f, 2.0f};
    auto x = Tensor(x_data, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> weight_data = {1.0f, 0.0f, 0.0f, 1.0f};
    auto weight = Tensor(weight_data, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> bias_data = {1.0f, 2.0f};
    auto bias = Tensor(bias_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    functional::LinearOp op(2, 2, true);
    auto result = op.forward({x, weight, bias})[0];

    Shape expected_shape{1, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    // x * weight^T + bias = [1,2] * [[1,0], [0,1]] + [1,2] = [1,2] + [1,2] = [2,4]
    std::vector<float> expected_data = {2.0f, 4.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(LinearOperatorTest, ForwardNoBias)
{
    // 测试无偏置的线性层
    std::vector<float> x_data = {1.0f, 2.0f};
    auto x = Tensor(x_data, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()));

    std::vector<float> weight_data = {1.0f, 0.0f, 0.0f, 1.0f};
    auto weight = Tensor(weight_data, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    functional::LinearOp op(2, 2, false);
    auto result = op.forward({x, weight})[0];

    Shape expected_shape{1, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    // x * weight^T = [1,2] * [[1,0], [0,1]] = [1,2]
    std::vector<float> expected_data = {1.0f, 2.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(LinearOperatorTest, Forward3DInput)
{
    // 测试3D输入（会被展平）
    // 输入: (2, 2, 3) -> 展平为 (2, 6)
    std::vector<float> x_data(12);
    for (size_t i = 0; i < 12; ++i) {
        x_data[i] = static_cast<float>(i);
    }
    auto x = Tensor(x_data, Shape{2, 2, 3}, dtype(DataType::kFloat32).device(deviceType()));

    // 权重: (4, 6) - out_features=4, in_features=6
    // 需要 4 * 6 = 24 个元素
    std::vector<float> weight_data(24);
    for (size_t i = 0; i < 24; ++i) {
        weight_data[i] = 1.0f;
    }
    auto weight = Tensor(weight_data, Shape{4, 6}, dtype(DataType::kFloat32).device(deviceType()));

    functional::LinearOp op(6, 4, false);
    auto result = op.forward({x, weight})[0];

    Shape expected_shape{2, 4};
    EXPECT_EQ(result.shape(), expected_shape);
    
    // 验证输出形状正确
    EXPECT_EQ(result.shape()[0], 2U);
    EXPECT_EQ(result.shape()[1], 4U);
}

INSTANTIATE_DEVICE_TEST_SUITE_P(LinearOperatorTest);

