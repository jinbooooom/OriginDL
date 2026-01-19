#include <gtest/gtest.h>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"

using namespace origin;
namespace F = origin::functional;

/**
 * @brief 加法算子测试类（参数化版本）
 * @details 使用参数化测试，自动为CPU和CUDA生成测试用例
 *          无GPU环境只运行CPU测试，有GPU环境运行CPU+CUDA测试
 */
class AddOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 前向传播测试 ====================

TEST_P(AddOperatorTest, ForwardBasic)
{
    // 测试基本加法运算
    Shape shape{2, 2};
    // 使用origindl API显式指定设备类型
    auto x0 = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, shape, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({5.0f, 6.0f, 7.0f, 8.0f}, shape, dtype(DataType::kFloat32).device(deviceType()));

    // 验证输入张量在正确的设备上
    origin::test::GTestUtils::EXPECT_TENSOR_DEVICE(x0, deviceType());
    origin::test::GTestUtils::EXPECT_TENSOR_DEVICE(x1, deviceType());

    auto result = F::add(x0, x1);

    // 验证结果张量在正确的设备上
    origin::test::GTestUtils::EXPECT_TENSOR_DEVICE(result, deviceType());

    EXPECT_EQ(result.shape(), shape);
    auto expected = Tensor({6.0f, 8.0f, 10.0f, 12.0f}, shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(AddOperatorTest, ForwardOperatorOverload)
{
    // 测试运算符重载
    Shape shape{2};
    auto x0 = Tensor({1.0f, 2.0f}, shape, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({3.0f, 4.0f}, shape, dtype(DataType::kFloat32).device(deviceType()));

    auto result = x0 + x1;

    EXPECT_EQ(result.shape(), shape);
    auto expected = Tensor({4.0f, 6.0f}, shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(AddOperatorTest, ForwardScalarTensor)
{
    // 测试标量与张量的加法
    auto x       = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    float scalar = 5.0f;

    auto result1 = x + scalar;
    auto result2 = scalar + x;

    EXPECT_EQ(result1.shape(), Shape{3});
    EXPECT_EQ(result2.shape(), Shape{3});

    auto expected = Tensor({6.0f, 7.0f, 8.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result1, expected, origin::test::TestTolerance::kDefault);
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result2, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(AddOperatorTest, ForwardZeroTensor)
{
    // 测试零张量加法
    auto x0 = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::add(x0, x1);

    // 结果应该等于x0
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x0, origin::test::TestTolerance::kDefault);
}

TEST_P(AddOperatorTest, ForwardNegativeValues)
{
    // 测试负值加法
    auto x0 = Tensor({-1.0f, -2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({3.0f, 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::add(x0, x1);

    auto expected = Tensor({2.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 反向传播测试 ====================

TEST_P(AddOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x0 = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    auto x1 = Tensor({3.0f, 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::add(x0, x1);
    y.backward();

    // 加法算子的梯度应该都是1
    auto expected_grad = Tensor::ones(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x0.grad(), expected_grad, origin::test::TestTolerance::kDefault);
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x1.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(AddOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x0 = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    auto x1 = Tensor({1.0f, 1.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::add(x0, x1);
    y.backward();

    // 加法算子的梯度：∂y/∂x0 = 1, ∂y/∂x1 = 1
    auto expected_grad = Tensor::ones(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x0.grad(), expected_grad, origin::test::TestTolerance::kDefault);
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x1.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(AddOperatorTest, BackwardDifferentShapes)
{
    // 测试不同形状的张量加法反向传播
    auto x0 = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    auto x1 = Tensor({3.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::add(x0, x1);
    y.backward();

    // 梯度应该正确广播
    auto gx0_data = x0.grad().to_vector<float>();
    auto gx1_data = x1.grad().to_vector<float>();

    EXPECT_EQ(gx0_data.size(), 2U);
    EXPECT_EQ(gx1_data.size(), 1U);

    // x0的梯度应该是全1
    auto expected_gx0 = Tensor::ones(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x0.grad(), expected_gx0, origin::test::TestTolerance::kDefault);

    // x1的梯度应该是2（广播后的梯度）
    EXPECT_NEAR(gx1_data[0], 2.0f, origin::test::TestTolerance::kDefault);
}

// ==================== 边界情况测试 ====================

TEST_P(AddOperatorTest, SingleElement)
{
    // 测试单元素张量
    auto x0 = Tensor({5.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({3.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::add(x0, x1);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 8.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(AddOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<float> data1(100, 1.0f);
    std::vector<float> data2(100, 2.0f);
    auto x0 = Tensor(data1, Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor(data2, Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::add(x0, x1);

    Shape expected_shape{10, 10};
    EXPECT_EQ(result.shape(), expected_shape);

    // 验证所有元素都是3.0
    auto expected =
        Tensor(std::vector<float>(100, 3.0f), Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(AddOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    auto x0 = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2, 2},
                     dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f, 0.7f, 0.8f}, Shape{2, 2, 2},
                     dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::add(x0, x1);

    Shape expected_shape{2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    // 期望值：x0[i] + x1[i]
    auto expected = Tensor({1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f}, Shape{2, 2, 2},
                           dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 数值稳定性测试 ====================

TEST_P(AddOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x0 = Tensor({1e-10f, 1e10f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({1e-10f, 1e10f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::add(x0, x1);

    auto expected = Tensor({2e-10f, 2e10f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(AddOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x0 = Tensor({0.1f, 0.2f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({0.3f, 0.4f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::add(x0, x1);

    auto expected = Tensor({0.4f, 0.6f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 原地操作测试 ====================

TEST_P(AddOperatorTest, InplaceBasic)
{
    // 测试基本原地加法运算
    Shape shape{2, 2};
    auto x0 = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, shape, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({5.0f, 6.0f, 7.0f, 8.0f}, shape, dtype(DataType::kFloat32).device(deviceType()));

    // 保存原始值用于验证
    auto x0_copy = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, shape, dtype(DataType::kFloat32).device(deviceType()));

    // 执行原地操作
    F::add_(x0, x1);

    // 验证结果
    auto expected = Tensor({6.0f, 8.0f, 10.0f, 12.0f}, shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x0, expected, origin::test::TestTolerance::kDefault);

    // 验证 x1 没有被修改
    auto x1_expected = Tensor({5.0f, 6.0f, 7.0f, 8.0f}, shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x1, x1_expected, origin::test::TestTolerance::kDefault);
}

TEST_P(AddOperatorTest, InplaceZeroTensor)
{
    // 测试零张量原地加法
    auto x0 = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto x0_original = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    F::add_(x0, x1);

    // 结果应该等于x0的原始值
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x0, x0_original, origin::test::TestTolerance::kDefault);
}

TEST_P(AddOperatorTest, InplaceNegativeValues)
{
    // 测试负值原地加法
    auto x0 = Tensor({-1.0f, -2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({3.0f, 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    F::add_(x0, x1);

    auto expected = Tensor({2.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x0, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 类型提升测试 ====================

TEST_P(AddOperatorTest, TypePromotionFloat32Float64)
{
    // 测试 float32 + float64 → float64
    auto x0 = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({5.0, 6.0, 7.0, 8.0}, Shape{2, 2}, dtype(DataType::kFloat64).device(deviceType()));

    auto result = F::add(x0, x1);

    // 结果类型应该是 float64
    EXPECT_EQ(result.dtype(), DataType::kFloat64);
    EXPECT_EQ(result.shape(), Shape({2, 2}));

    // 验证值正确
    auto expected = Tensor({6.0, 8.0, 10.0, 12.0}, Shape{2, 2}, dtype(DataType::kFloat64).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(AddOperatorTest, TypePromotionFloat64Float32)
{
    // 测试 float64 + float32 → float64（float64优先级更高）
    auto x0 = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2}, dtype(DataType::kFloat64).device(deviceType()));
    auto x1 = Tensor({5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::add(x0, x1);

    // 结果类型应该是 float64
    EXPECT_EQ(result.dtype(), DataType::kFloat64);
    EXPECT_EQ(result.shape(), Shape({2, 2}));
}

TEST_P(AddOperatorTest, TypePromotionInt32Float32)
{
    // 测试 int32 + float32 → float32（浮点优先级高于整数）
    auto x0 = Tensor({1, 2, 3, 4}, Shape{2, 2}, dtype(DataType::kInt32).device(deviceType()));
    auto x1 = Tensor({5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::add(x0, x1);

    // 结果类型应该是 float32
    EXPECT_EQ(result.dtype(), DataType::kFloat32);
    EXPECT_EQ(result.shape(), Shape({2, 2}));

    // 验证值正确（整数转换为浮点数）
    auto expected = Tensor({6.0f, 8.0f, 10.0f, 12.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(AddOperatorTest, TypePromotionFloat32Int32)
{
    // 测试 float32 + int32 → float32（浮点优先级高于整数）
    auto x0 = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({5, 6, 7, 8}, Shape{2, 2}, dtype(DataType::kInt32).device(deviceType()));

    auto result = F::add(x0, x1);

    // 结果类型应该是 float32
    EXPECT_EQ(result.dtype(), DataType::kFloat32);
    EXPECT_EQ(result.shape(), Shape({2, 2}));
}

TEST_P(AddOperatorTest, TypePromotionInt32Int64)
{
    // 测试 int32 + int64 → int64
    auto x0 = Tensor({1, 2, 3, 4}, Shape{2, 2}, dtype(DataType::kInt32).device(deviceType()));
    auto x1 = Tensor({5L, 6L, 7L, 8L}, Shape{2, 2}, dtype(DataType::kInt64).device(deviceType()));

    auto result = F::add(x0, x1);

    // 结果类型应该是 int64
    EXPECT_EQ(result.dtype(), DataType::kInt64);
    EXPECT_EQ(result.shape(), Shape({2, 2}));

    // 验证值正确（通过转换为float32来验证，因为EXPECT_TENSORS_EQ不支持int64）
    auto result_f32 = result.to(DataType::kFloat32);
    auto expected_f32 = Tensor({6.0f, 8.0f, 10.0f, 12.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result_f32, expected_f32, origin::test::TestTolerance::kDefault);
}

TEST_P(AddOperatorTest, TypePromotionInt64Int32)
{
    // 测试 int64 + int32 → int64
    auto x0 = Tensor({1L, 2L, 3L, 4L}, Shape{2, 2}, dtype(DataType::kInt64).device(deviceType()));
    auto x1 = Tensor({5, 6, 7, 8}, Shape{2, 2}, dtype(DataType::kInt32).device(deviceType()));

    auto result = F::add(x0, x1);

    // 结果类型应该是 int64
    EXPECT_EQ(result.dtype(), DataType::kInt64);
    EXPECT_EQ(result.shape(), Shape({2, 2}));
}

TEST_P(AddOperatorTest, TypePromotionInt64Double)
{
    // 测试 int64 + double → double（浮点优先级高于整数）
    auto x0 = Tensor({1L, 2L, 3L, 4L}, Shape{2, 2}, dtype(DataType::kInt64).device(deviceType()));
    auto x1 = Tensor({5.5, 6.5, 7.5, 8.5}, Shape{2, 2}, dtype(DataType::kFloat64).device(deviceType()));

    auto result = F::add(x0, x1);

    // 结果类型应该是 float64
    EXPECT_EQ(result.dtype(), DataType::kFloat64);
    EXPECT_EQ(result.shape(), Shape({2, 2}));

    // 验证值正确
    auto expected = Tensor({6.5, 8.5, 10.5, 12.5}, Shape{2, 2}, dtype(DataType::kFloat64).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(AddOperatorTest, TypePromotionSameType)
{
    // 测试相同类型不提升：float32 + float32 → float32
    auto x0 = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::add(x0, x1);

    // 结果类型应该是 float32（不提升）
    EXPECT_EQ(result.dtype(), DataType::kFloat32);
    EXPECT_EQ(result.shape(), Shape({2, 2}));
}

TEST_P(AddOperatorTest, TypePromotionBackward)
{
    // 测试类型提升后的反向传播
    auto x0 = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    auto x1 = Tensor({3.0, 4.0}, Shape{2}, dtype(DataType::kFloat64).device(deviceType()).requires_grad(true));

    auto y = F::add(x0, x1);
    y.backward();

    // 结果类型应该是 float64
    EXPECT_EQ(y.dtype(), DataType::kFloat64);

    // 梯度类型应该和输出类型一致（提升后的类型）
    EXPECT_EQ(x0.grad().dtype(), DataType::kFloat64);
    EXPECT_EQ(x1.grad().dtype(), DataType::kFloat64);

    // 梯度值应该都是1
    auto expected_grad = Tensor::ones(Shape{2}, dtype(DataType::kFloat64).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x0.grad(), expected_grad, origin::test::TestTolerance::kDefault);
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x1.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

// 实例化测试套件：自动为CPU和可用CUDA生成测试
INSTANTIATE_DEVICE_TEST_SUITE_P(AddOperatorTest);
