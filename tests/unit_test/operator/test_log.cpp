#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "origin.h"
#include "../../common/device_test_base.h"
#include "../../common/gtest_utils.h"
#include "../../common/test_utils.h"

using namespace origin;

/**
 * @brief 对数算子测试类（参数化版本）
 */
class LogOperatorTest : public origin::test::OperatorTestBase
{
};

// ==================== 前向传播测试 ====================

TEST_P(LogOperatorTest, ForwardBasic)
{
    // 测试基本对数运算
    auto x = Tensor({1.0f, std::exp(1.0f), std::exp(2.0f)}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = log(x);

    Shape expected_shape{3};
    EXPECT_EQ(result.shape(), expected_shape);
    std::vector<float> expected_data = {0.0f, 1.0f, 2.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(LogOperatorTest, ForwardOne)
{
    // 测试 log(1) = 0
    auto x = Tensor::ones(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = log(x);

    auto expected = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(LogOperatorTest, ForwardSmallValues)
{
    // 测试小值
    auto x = Tensor({0.1f, 0.5f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = log(x);

    std::vector<float> expected_data = {static_cast<float>(std::log(0.1)), static_cast<float>(std::log(0.5))};
    auto expected = Tensor(expected_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(LogOperatorTest, ForwardLargeValues)
{
    // 测试大值
    auto x = Tensor({std::exp(5.0f), std::exp(10.0f)}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = log(x);

    std::vector<float> expected_data = {5.0f, 10.0f};
    auto expected = Tensor(expected_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_NEAR(result, expected, 1e-3, 1e3);
}

// ==================== 反向传播测试 ====================

TEST_P(LogOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    // log(x) 的梯度：∂y/∂x = 1/x
    auto x = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = log(x);
    y.backward();

    // 梯度应该是 1/x
    std::vector<float> expected_grad_data = {1.0f / 2.0f, 1.0f / 3.0f};
    auto expected_grad = Tensor(expected_grad_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(LogOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x = Tensor({2.0f, 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = log(x);
    y.backward();

    // 梯度应该是 1/x
    std::vector<float> expected_grad_data = {1.0f / 2.0f, 1.0f / 4.0f};
    auto expected_grad = Tensor(expected_grad_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(LogOperatorTest, BackwardOne)
{
    // 测试 log(1) 的梯度
    auto x = Tensor::ones(Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = log(x);
    y.backward();

    // log(1) = 0，梯度是 1/1 = 1
    auto expected_grad = Tensor::ones(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

// ==================== 边界情况测试 ====================

TEST_P(LogOperatorTest, SingleElement)
{
    // 测试单元素张量
    auto x = Tensor({std::exp(1.0f)}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = log(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 1.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(LogOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<float> data(100);
    for (size_t i = 0; i < 100; ++i)
    {
        data[i] = std::exp(1.0f);
    }
    auto x = Tensor(data, Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = log(x);

    Shape expected_shape{10, 10};
    EXPECT_EQ(result.shape(), expected_shape);
    
    auto expected = Tensor(std::vector<float>(100, 1.0f), Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(LogOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    std::vector<float> data;
    for (int i = 0; i < 8; ++i)
    {
        data.push_back(std::exp(static_cast<float>(i)));
    }
    auto x = Tensor(data, Shape{2, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = log(x);

    Shape expected_shape{2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    
    std::vector<float> expected_data;
    for (int i = 0; i < 8; ++i)
    {
        expected_data.push_back(static_cast<float>(i));
    }
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_NEAR(result, expected, 1e-3, 1e3);
}

// ==================== 数值稳定性测试 ====================

TEST_P(LogOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x = Tensor({std::exp(0.1f), std::exp(0.2f)}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = log(x);

    std::vector<float> expected_data = {0.1f, 0.2f};
    auto expected = Tensor(expected_data, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(LogOperatorTest, IdentityProperty)
{
    // 测试恒等性质：log(exp(x)) = x
    auto x = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto exp_x = exp(x);
    auto result = log(exp_x);

    origin::test::GTestUtils::EXPECT_TENSORS_NEAR(result, x, 1e-5, 1e5);
}

TEST_P(LogOperatorTest, MonotonicProperty)
{
    // 测试单调性：如果 x1 < x2，则 log(x1) < log(x2) (当 x1, x2 > 0)
    auto x1 = Tensor({1.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    auto x2 = Tensor({2.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto y1 = log(x1);
    auto y2 = log(x2);

    EXPECT_LT(y1.item<float>(), y2.item<float>());
}

// 实例化测试套件：自动为CPU和可用CUDA生成测试
INSTANTIATE_DEVICE_TEST_SUITE_P(LogOperatorTest);

