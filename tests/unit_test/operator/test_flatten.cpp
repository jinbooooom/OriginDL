#include <gtest/gtest.h>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"
#include "origin/operators/shape/flatten.h"

using namespace origin;
namespace F = origin::functional;

/**
 * @brief Flatten 算子测试类（参数化版本）
 */
class FlattenOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 前向传播测试 ====================

TEST_P(FlattenOperatorTest, ForwardBasic)
{
    // 测试基本展平操作
    // 输入: (1, 2, 3) - 3D张量
    // start_dim=1, end_dim=-1
    // 期望输出: (1, 6)

    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    auto x                    = Tensor(x_data, Shape{1, 2, 3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::flatten(x, 1, -1);

    Shape expected_shape{1, 6};
    EXPECT_EQ(result.shape(), expected_shape);

    // 数据应该保持不变
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x.reshape(expected_shape),
                                                origin::test::TestTolerance::kDefault);
}

TEST_P(FlattenOperatorTest, Forward2D)
{
    // 测试2D张量展平
    std::vector<float> x_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto x                    = Tensor(x_data, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::flatten(x, 0, -1);

    Shape expected_shape{4};
    EXPECT_EQ(result.shape(), expected_shape);

    // 数据应该保持不变
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x.reshape(expected_shape),
                                                origin::test::TestTolerance::kDefault);
}

TEST_P(FlattenOperatorTest, Forward4D)
{
    // 测试4D张量展平（ResNet 常用）
    std::vector<float> x_data(24);
    for (size_t i = 0; i < 24; ++i)
    {
        x_data[i] = static_cast<float>(i);
    }
    auto x = Tensor(x_data, Shape{1, 2, 3, 4}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::flatten(x, 1, -1);

    Shape expected_shape{1, 24};
    EXPECT_EQ(result.shape(), expected_shape);

    // 数据应该保持不变
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x.reshape(expected_shape),
                                                origin::test::TestTolerance::kDefault);
}

TEST_P(FlattenOperatorTest, ForwardPartial)
{
    // 测试部分维度展平
    std::vector<float> x_data(12);
    for (size_t i = 0; i < 12; ++i)
    {
        x_data[i] = static_cast<float>(i);
    }
    auto x = Tensor(x_data, Shape{2, 2, 3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::flatten(x, 1, 2);

    Shape expected_shape{2, 6};
    EXPECT_EQ(result.shape(), expected_shape);

    // 数据应该保持不变
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x.reshape(expected_shape),
                                                origin::test::TestTolerance::kDefault);
}

INSTANTIATE_DEVICE_TEST_SUITE_P(FlattenOperatorTest);
