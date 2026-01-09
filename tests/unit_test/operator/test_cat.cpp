#include <gtest/gtest.h>
#include <vector>
#include "origin.h"
#include "origin/operators/shape/cat.h"
#include "../../common/device_test_base.h"
#include "../../common/gtest_utils.h"
#include "../../common/test_utils.h"

using namespace origin;
namespace F = origin::functional;

/**
 * @brief Cat 算子测试类（参数化版本）
 */
class CatOperatorTest : public origin::test::OperatorTestBase
{
};

// ==================== 前向传播测试 ====================

TEST_P(CatOperatorTest, ForwardBasic)
{
    // 测试基本拼接：在 dim=0 上拼接
    auto x0 = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({3.0f, 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::cat({x0, x1}, 0);

    Shape expected_shape{4};
    EXPECT_EQ(result.shape(), expected_shape);
    
    std::vector<float> expected_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(CatOperatorTest, ForwardDim1)
{
    // 测试在 dim=1 上拼接
    auto x0 = Tensor({1.0f, 2.0f}, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({3.0f, 4.0f}, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::cat({x0, x1}, 1);

    Shape expected_shape{1, 4};
    EXPECT_EQ(result.shape(), expected_shape);
    
    std::vector<float> expected_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(CatOperatorTest, ForwardMultipleTensors)
{
    // 测试拼接多个张量
    auto x0 = Tensor({1.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({2.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    auto x2 = Tensor({3.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::cat({x0, x1, x2}, 0);

    Shape expected_shape{3};
    EXPECT_EQ(result.shape(), expected_shape);
    
    std::vector<float> expected_data = {1.0f, 2.0f, 3.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(CatOperatorTest, ForwardTwoDimensional)
{
    // 测试 2D 张量拼接
    auto x0 = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    // 在 dim=0 上拼接
    auto result0 = F::cat({x0, x1}, 0);
    Shape expected_shape0{4, 2};
    EXPECT_EQ(result0.shape(), expected_shape0);
    
    // 在 dim=1 上拼接
    auto result1 = F::cat({x0, x1}, 1);
    Shape expected_shape1{2, 4};
    EXPECT_EQ(result1.shape(), expected_shape1);
}

TEST_P(CatOperatorTest, ForwardSingleTensor)
{
    // 测试单个张量（应该直接返回）
    auto x = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::cat({x}, 0);

    // 应该和输入相同
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x, origin::test::TestTolerance::kDefault);
}

// ==================== 反向传播测试 ====================

TEST_P(CatOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x0 = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    auto x1 = Tensor({3.0f, 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::cat({x0, x1}, 0);
    y.backward();

    // 验证梯度形状
    EXPECT_EQ(x0.grad().shape(), x0.shape());
    EXPECT_EQ(x1.grad().shape(), x1.shape());
    
    // 梯度应该被分割回各个输入
    // 由于输出梯度是全1（默认），每个输入的梯度应该是全1
    auto expected_grad = Tensor::ones(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x0.grad(), expected_grad, origin::test::TestTolerance::kDefault);
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x1.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(CatOperatorTest, BackwardDim1)
{
    // 测试在 dim=1 上的反向传播
    auto x0 = Tensor({1.0f, 2.0f}, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    auto x1 = Tensor({3.0f, 4.0f}, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::cat({x0, x1}, 1);
    y.backward();

    // 验证梯度形状
    EXPECT_EQ(x0.grad().shape(), x0.shape());
    EXPECT_EQ(x1.grad().shape(), x1.shape());
}

// ==================== 边界情况测试 ====================

TEST_P(CatOperatorTest, SingleElement)
{
    // 测试单元素张量拼接
    auto x0 = Tensor({5.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({6.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::cat({x0, x1}, 0);

    Shape expected_shape{2};
    EXPECT_EQ(result.shape(), expected_shape);
    
    std::vector<float> expected_data = {5.0f, 6.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(CatOperatorTest, DifferentSizes)
{
    // 测试不同大小的张量拼接
    auto x0 = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({3.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::cat({x0, x1}, 0);

    Shape expected_shape{3};
    EXPECT_EQ(result.shape(), expected_shape);
    
    std::vector<float> expected_data = {1.0f, 2.0f, 3.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// Instantiate test suite: automatically generate tests for CPU and available CUDA
INSTANTIATE_DEVICE_TEST_SUITE_P(CatOperatorTest);

