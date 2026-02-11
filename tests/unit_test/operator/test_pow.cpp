#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"

using namespace origin;
namespace F = origin::functional;
/**
 * @brief 幂算子测试类（参数化版本）
 */
class PowOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 前向传播测试 ====================

TEST_P(PowOperatorTest, ForwardBasic)
{
    // 测试基本幂运算
    auto x       = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = 2;

    auto result = F::pow(x, Scalar(exponent));

    Shape expected_shape{2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto expected = Tensor({4.0f, 9.0f}, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, ForwardOperatorOverload)
{
    // 测试运算符重载
    auto x       = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = 3;

    auto result = x ^ exponent;

    Shape expected_shape{2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto expected = Tensor({8.0f, 27.0f}, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, ForwardZeroExponent)
{
    // 测试零指数
    auto x       = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = 0;

    auto result = F::pow(x, Scalar(exponent));

    auto expected = Tensor::ones(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, ForwardNegativeExponent)
{
    // 测试负指数
    auto x       = Tensor({2.0f, 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = -1;

    auto result = F::pow(x, Scalar(exponent));

    auto expected = Tensor({0.5f, 0.25f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, ForwardZeroBase)
{
    // 测试零底数
    auto x       = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = 2;

    auto result = F::pow(x, Scalar(exponent));

    auto expected = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 反向传播测试 ====================

TEST_P(PowOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x       = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    int exponent = 2;

    auto y = F::pow(x, Scalar(exponent));
    y.backward();

    // 幂算子的梯度：∂y/∂x = exponent * x^(exponent-1)
    auto expected_grad = Tensor({4.0f, 6.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x       = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    int exponent = 3;

    auto y = F::pow(x, Scalar(exponent));
    y.backward();

    // 梯度会累积
    auto expected_grad = Tensor({12.0f, 27.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, BackwardZeroExponent)
{
    // 测试零指数的梯度
    auto x       = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    int exponent = 0;

    auto y = F::pow(x, Scalar(exponent));
    y.backward();

    auto expected_grad = Tensor::zeros(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, BackwardNegativeExponent)
{
    // 测试负指数的梯度
    auto x       = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    int exponent = -2;

    auto y = F::pow(x, Scalar(exponent));
    y.backward();

    // 梯度：-2 * x^(-3) = -2 / x^3
    auto expected_grad =
        Tensor({-2.0f / 8.0f, -2.0f / 27.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

// ==================== 边界情况测试 ====================

TEST_P(PowOperatorTest, SingleElement)
{
    // 测试单元素张量
    auto x       = Tensor({2.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = 3;

    auto result = F::pow(x, Scalar(exponent));

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 8.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<float> data(100, 2.0f);
    auto x       = Tensor(data, Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = 2;

    auto result = F::pow(x, Scalar(exponent));

    Shape expected_shape{10, 10};
    EXPECT_EQ(result.shape(), expected_shape);

    auto expected =
        Tensor(std::vector<float>(100, 4.0f), Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    auto x       = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2, 2},
                          dtype(DataType::kFloat32).device(deviceType()));
    int exponent = 2;

    auto result = F::pow(x, Scalar(exponent));

    Shape expected_shape{2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    std::vector<float> expected_data;
    for (int i = 1; i <= 8; ++i)
    {
        float val = static_cast<float>(i);
        expected_data.push_back(val * val);
    }
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 数值稳定性测试 ====================

TEST_P(PowOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x       = Tensor({1e-3f, 1e5f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = 2;

    auto result = F::pow(x, Scalar(exponent));

    auto expected = Tensor({1e-6f, 1e10f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_NEAR(result, expected, 1e-3, 1e9);
}

TEST_P(PowOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x       = Tensor({0.1f, 0.2f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = 3;

    auto result = F::pow(x, Scalar(exponent));

    auto expected = Tensor({0.001f, 0.008f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 特殊值测试 ====================

TEST_P(PowOperatorTest, DifferentExponents)
{
    // 测试不同指数
    auto x = Tensor({2.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    // 测试指数为1
    auto result1 = F::pow(x, Scalar(1));
    EXPECT_NEAR(result1.item<float>(), 2.0f, origin::test::TestTolerance::kDefault);

    // 测试指数为2
    auto result2 = F::pow(x, Scalar(2));
    EXPECT_NEAR(result2.item<float>(), 4.0f, origin::test::TestTolerance::kDefault);

    // 测试指数为3
    auto result3 = F::pow(x, Scalar(3));
    EXPECT_NEAR(result3.item<float>(), 8.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, IdentityProperty)
{
    // 测试恒等性质：x^1 = x
    auto x = Tensor({2.0f, 3.0f, 4.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::pow(x, Scalar(1));

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, ZeroPowerProperty)
{
    // 测试零幂性质：x^0 = 1
    auto x = Tensor({2.0f, 3.0f, 4.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::pow(x, Scalar(0));

    auto expected = Tensor::ones(Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, LargeExponent)
{
    // 测试大指数
    auto x       = Tensor({1.1f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = 10;

    auto result = F::pow(x, Scalar(exponent));

    EXPECT_NEAR(result.item<float>(), static_cast<float>(std::pow(1.1, 10)), origin::test::TestTolerance::kDefault);
}

// ==================== 负数底数测试 ====================

TEST_P(PowOperatorTest, NegativeBaseWithIntegerExponent)
{
    // 测试负数底数 + 整数指数 → 正常
    auto x       = Tensor({-2.0f, -3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = 2;

    auto result = F::pow(x, Scalar(exponent));

    // 负数底数的整数次幂应该正常计算
    // (-2)^2 = 4, (-3)^2 = 9
    auto expected = Tensor({4.0f, 9.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, NegativeBaseWithPositiveIntegerExponent)
{
    // 测试负数底数 + 正整数指数
    auto x       = Tensor({-2.0f, -3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = 3;

    auto result = F::pow(x, Scalar(exponent));

    // (-2)^3 = -8, (-3)^3 = -27
    auto expected = Tensor({-8.0f, -27.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, NegativeBaseWithNegativeIntegerExponent)
{
    // 测试负数底数 + 负整数指数
    auto x       = Tensor({-2.0f, -4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    int exponent = -2;

    auto result = F::pow(x, Scalar(exponent));

    // (-2)^(-2) = 1/4 = 0.25, (-4)^(-2) = 1/16 = 0.0625
    auto expected = Tensor({0.25f, 0.0625f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, NegativeBaseWithNonIntegerExponent)
{
    // 测试负数底数 + 非整数指数 → NaN
    auto x          = Tensor({-2.0f, -3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    double exponent = 2.5;  // 非整数指数

    auto result = F::pow(x, Scalar(exponent));

    // 负数底数的非整数次幂应该产生 NaN
    // 结果可能被提升为 float64，这里统一转换到 float32 再检查
    auto result_data = result.to_vector<float>();
    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_TRUE(std::isnan(result_data[i]))
            << "Element " << i << " should be NaN for negative base with non-integer exponent";
    }
}

TEST_P(PowOperatorTest, NegativeBaseWithFloatExponent)
{
    // 测试负数底数 + 浮点数指数（非整数）→ NaN
    auto x         = Tensor({-2.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    float exponent = 2.5f;  // 浮点数非整数指数

    auto result = F::pow(x, Scalar(exponent));

    // 应该产生 NaN
    float result_value = result.item<float>();
    EXPECT_TRUE(std::isnan(result_value)) << "Result should be NaN for negative base with float non-integer exponent";
}

TEST_P(PowOperatorTest, NegativeBaseWithHalfExponent)
{
    // 测试负数底数 + 0.5指数（平方根）→ NaN
    auto x          = Tensor({-4.0f, -9.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    double exponent = 0.5;  // 平方根

    auto result = F::pow(x, Scalar(exponent));

    auto result_data = result.to_vector<float>();
    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_TRUE(std::isnan(result_data[i]))
            << "Element " << i << " should be NaN for negative base with 0.5 exponent";
    }
}

// ==================== 浮点数指数测试 ====================

TEST_P(PowOperatorTest, FloatExponentPositiveBase)
{
    // 测试正数底数 + 浮点数指数
    auto x          = Tensor({4.0f, 9.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    double exponent = 2.5;

    auto result = F::pow(x, Scalar(exponent));

    // 4^2.5 = 4^2 * 4^0.5 = 16 * 2 = 32
    // 9^2.5 = 9^2 * 9^0.5 = 81 * 3 = 243
    auto expected = Tensor({32.0f, 243.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_NEAR(result, expected, 1e-3, 1e-2);
}

TEST_P(PowOperatorTest, FloatExponentWithOperator)
{
    // 测试使用运算符的浮点数指数
    auto x          = Tensor({2.0f, 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    double exponent = 2.5;

    auto result = x ^ exponent;

    // 验证结果不为 NaN（正数底数）；to_vector<float>() 内部会做类型转换
    auto result_data = result.to_vector<float>();
    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_FALSE(std::isnan(result_data[i])) << "Element " << i << " should not be NaN for positive base";
        EXPECT_TRUE(std::isfinite(result_data[i])) << "Element " << i << " should be finite";
    }
}

TEST_P(PowOperatorTest, FloatExponentSmallValue)
{
    // 测试小数值的浮点数指数
    auto x          = Tensor({2.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    double exponent = 0.5;  // 平方根

    auto result = F::pow(x, Scalar(exponent));

    // 2^0.5 = sqrt(2) ≈ 1.414
    EXPECT_NEAR(result.item<float>(), static_cast<float>(std::sqrt(2.0)), origin::test::TestTolerance::kDefault);
}

TEST_P(PowOperatorTest, FloatExponentNegativeValue)
{
    // 测试负浮点数指数
    auto x          = Tensor({4.0f, 9.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    double exponent = -0.5;  // 负平方根

    auto result = F::pow(x, Scalar(exponent));

    // 4^(-0.5) = 1/sqrt(4) = 0.5
    // 9^(-0.5) = 1/sqrt(9) = 1/3 ≈ 0.333
    auto expected = Tensor({0.5f, 1.0f / 3.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_NEAR(result, expected, 1e-3, 1e-4);
}

TEST_P(PowOperatorTest, MixedPositiveNegativeBaseWithFloatExponent)
{
    // 测试混合正负底数 + 浮点数指数
    auto x          = Tensor({2.0f, -2.0f, 4.0f, -4.0f}, Shape{4}, dtype(DataType::kFloat32).device(deviceType()));
    double exponent = 2.5;

    auto result = F::pow(x, Scalar(exponent));

    // 结果可能被提升为 float64，to_vector<float>() 内部会做类型转换
    auto result_data = result.to_vector<float>();

    // 正数底数应该正常计算
    EXPECT_FALSE(std::isnan(result_data[0])) << "Positive base should not produce NaN";
    EXPECT_FALSE(std::isnan(result_data[2])) << "Positive base should not produce NaN";

    // 负数底数应该产生 NaN
    EXPECT_TRUE(std::isnan(result_data[1])) << "Negative base should produce NaN";
    EXPECT_TRUE(std::isnan(result_data[3])) << "Negative base should produce NaN";
}

// ==================== 综合测试 ====================

TEST_P(PowOperatorTest, NegativeBaseIntegerVsNonInteger)
{
    // 对比测试：负数底数的整数指数 vs 非整数指数
    auto x_neg = Tensor({-2.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    // 整数指数：应该正常
    auto result_int = F::pow(x_neg, Scalar(2));
    EXPECT_FALSE(std::isnan(result_int.item<float>())) << "Integer exponent should work with negative base";
    EXPECT_NEAR(result_int.item<float>(), 4.0f, origin::test::TestTolerance::kDefault);

    // 非整数指数：应该产生 NaN
    auto result_non_int = F::pow(x_neg, Scalar(2.5f));
    EXPECT_TRUE(std::isnan(result_non_int.item<float>()))
        << "Non-integer exponent should produce NaN with negative base";
}

// ==================== 类型提升测试 ====================

TEST_P(PowOperatorTest, TypePromotion)
{
    // 测试 pow 操作的类型提升规则，包括 pow 的特殊规则（整数→float32）

    // 1. int32 tensor + int32 exponent → 应提升到 float32（pow 的特殊规则）
    {
        auto x      = Tensor({2, 3, 4}, Shape{3}, dtype(DataType::kInt32).device(deviceType()));
        int32_t exp = 2;
        auto result = F::pow(x, Scalar(exp));
        EXPECT_EQ(result.dtype(), DataType::kFloat32) << "int32 + int32 should promote to float32 (pow special rule)";
        auto expected = Tensor({4.0f, 9.0f, 16.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
    }

    // 2. int32 tensor + float32 exponent → 应提升到 float32
    {
        auto x      = Tensor({2, 3, 4}, Shape{3}, dtype(DataType::kInt32).device(deviceType()));
        float exp   = 2.5f;
        auto result = F::pow(x, Scalar(exp));
        EXPECT_EQ(result.dtype(), DataType::kFloat32) << "int32 + float32 should promote to float32";
        // 验证值：2^2.5 ≈ 5.657, 3^2.5 ≈ 15.588, 4^2.5 = 32.0
        auto result_data = result.to_vector<float>();
        EXPECT_NEAR(result_data[0], std::pow(2.0, 2.5), 1e-3);
        EXPECT_NEAR(result_data[1], std::pow(3.0, 2.5), 1e-3);
        EXPECT_NEAR(result_data[2], 32.0f, 1e-3);
    }

    // 3. float32 tensor + int32 exponent → 应保持 float32
    {
        auto x      = Tensor({2.0f, 3.0f, 4.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
        int32_t exp = 2;
        auto result = F::pow(x, Scalar(exp));
        EXPECT_EQ(result.dtype(), DataType::kFloat32) << "float32 + int32 should stay float32";
        auto expected = Tensor({4.0f, 9.0f, 16.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
    }

    // 4. float64 tensor + int32 exponent → 应保持 float64
    {
        auto x      = Tensor({2.0, 3.0, 4.0}, Shape{3}, dtype(DataType::kFloat64).device(deviceType()));
        int32_t exp = 2;
        auto result = F::pow(x, Scalar(exp));
        EXPECT_EQ(result.dtype(), DataType::kFloat64) << "float64 + int32 should stay float64";
        auto expected = Tensor({4.0, 9.0, 16.0}, Shape{3}, dtype(DataType::kFloat64).device(deviceType()));
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
    }

    // 5. int32 tensor + float64 exponent → 应提升到 float64
    {
        auto x      = Tensor({2, 3, 4}, Shape{3}, dtype(DataType::kInt32).device(deviceType()));
        double exp  = 2.5;
        auto result = F::pow(x, Scalar(exp));
        EXPECT_EQ(result.dtype(), DataType::kFloat64) << "int32 + float64 should promote to float64";
        // 验证值
        auto result_data = result.to_vector<double>();
        EXPECT_NEAR(result_data[0], std::pow(2.0, 2.5), 1e-6);
        EXPECT_NEAR(result_data[1], std::pow(3.0, 2.5), 1e-6);
        EXPECT_NEAR(result_data[2], 32.0, 1e-6);
    }

    // 6. int64 tensor + int64 exponent → 应提升到 float32（pow 的特殊规则）
    {
        auto x      = Tensor({2L, 3L, 4L}, Shape{3}, dtype(DataType::kInt64).device(deviceType()));
        int64_t exp = 2L;
        auto result = F::pow(x, Scalar(exp));
        EXPECT_EQ(result.dtype(), DataType::kFloat32) << "int64 + int64 should promote to float32 (pow special rule)";
        auto expected = Tensor({4.0f, 9.0f, 16.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
    }
}

// 实例化测试套件：自动为CPU和可用CUDA生成测试
INSTANTIATE_DEVICE_TEST_SUITE_P(PowOperatorTest);
