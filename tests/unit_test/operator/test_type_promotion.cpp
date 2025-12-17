#include <gtest/gtest.h>
#include <vector>
#include "origin.h"
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"

using namespace origin;

/**
 * @brief 类型提升测试类（参数化版本）
 * @details 使用参数化测试，自动为CPU和CUDA生成测试用例
 *          测试标量运算中的类型提升规则
 */
class TypePromotionTest : public origin::test::OperatorTestBase
{
};

// ==================== 浮点类型提升测试 ====================

TEST_P(TypePromotionTest, Float32PlusDouble)
{
    // 测试 float32 + double → float64
    auto a = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    double scalar = 2.0;
    
    auto result = a + scalar;
    
    // 结果类型应该是 float64
    EXPECT_EQ(result.dtype(), DataType::kFloat64);
    EXPECT_EQ(result.shape(), Shape({2, 2}));
    
    // 验证值正确
    auto expected = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat64).device(deviceType())) + 2.0;
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(TypePromotionTest, DoublePlusFloat32)
{
    // 测试 double + float32 → float64
    auto a = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat64).device(deviceType()));
    float scalar = 2.0f;
    
    auto result = a + scalar;
    
    // 结果类型应该是 float64
    EXPECT_EQ(result.dtype(), DataType::kFloat64);
}

TEST_P(TypePromotionTest, Float32PlusFloat32)
{
    // 测试 float32 + float32 → float32（相同类型，不提升）
    auto a = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    float scalar = 2.0f;
    
    auto result = a + scalar;
    
    // 结果类型应该是 float32
    EXPECT_EQ(result.dtype(), DataType::kFloat32);
}

// ==================== 整数类型提升测试 ====================

TEST_P(TypePromotionTest, Int32PlusInt64)
{
    // 测试 int32 + int64 → int64
    auto a = Tensor({1, 2, 3, 4}, Shape{2, 2}, dtype(DataType::kInt32).device(deviceType()));
    int64_t scalar = 2L;
    
    auto result = a + scalar;
    
    // 结果类型应该是 int64
    EXPECT_EQ(result.dtype(), DataType::kInt64);
    EXPECT_EQ(result.shape(), Shape({2, 2}));
}

TEST_P(TypePromotionTest, Int64PlusInt32)
{
    // 测试 int64 + int32 → int64
    auto a = Tensor({1L, 2L, 3L, 4L}, Shape{2, 2}, dtype(DataType::kInt64).device(deviceType()));
    int32_t scalar = 2;
    
    auto result = a + scalar;
    
    // 结果类型应该是 int64
    EXPECT_EQ(result.dtype(), DataType::kInt64);
}

TEST_P(TypePromotionTest, Int32PlusInt32)
{
    // 测试 int32 + int32 → int32（相同类型，不提升）
    auto a = Tensor({1, 2, 3, 4}, Shape{2, 2}, dtype(DataType::kInt32).device(deviceType()));
    int32_t scalar = 2;
    
    auto result = a + scalar;
    
    // 结果类型应该是 int32
    EXPECT_EQ(result.dtype(), DataType::kInt32);
}

// ==================== 浮点优先级高于整数测试 ====================

TEST_P(TypePromotionTest, Int32PlusFloat32)
{
    // 测试 int32 + float32 → float32（浮点优先级高于整数）
    auto a = Tensor({1, 2, 3, 4}, Shape{2, 2}, dtype(DataType::kInt32).device(deviceType()));
    float scalar = 2.5f;
    
    auto result = a + scalar;
    
    // 结果类型应该是 float32
    EXPECT_EQ(result.dtype(), DataType::kFloat32);
    EXPECT_EQ(result.shape(), Shape({2, 2}));
    
    // 验证值正确（整数转换为浮点数）
    auto expected = Tensor({3.5f, 4.5f, 5.5f, 6.5f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(TypePromotionTest, Float32PlusInt32)
{
    // 测试 float32 + int32 → float32（浮点优先级高于整数）
    auto a = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    int32_t scalar = 2;
    
    auto result = a + scalar;
    
    // 结果类型应该是 float32
    EXPECT_EQ(result.dtype(), DataType::kFloat32);
}

TEST_P(TypePromotionTest, Int64PlusDouble)
{
    // 测试 int64 + double → double（浮点优先级高于整数）
    auto a = Tensor({1L, 2L, 3L, 4L}, Shape{2, 2}, dtype(DataType::kInt64).device(deviceType()));
    double scalar = 2.5;
    
    auto result = a + scalar;
    
    // 结果类型应该是 float64
    EXPECT_EQ(result.dtype(), DataType::kFloat64);
}

// ==================== 减法运算类型提升测试 ====================

TEST_P(TypePromotionTest, SubFloat32Double)
{
    // 测试 float32 - double → float64
    auto a = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    double scalar = 2.0;
    
    auto result = a - scalar;
    
    EXPECT_EQ(result.dtype(), DataType::kFloat64);
}

TEST_P(TypePromotionTest, SubInt32Float32)
{
    // 测试 int32 - float32 → float32
    auto a = Tensor({5, 6, 7, 8}, Shape{2, 2}, dtype(DataType::kInt32).device(deviceType()));
    float scalar = 2.0f;
    
    auto result = a - scalar;
    
    EXPECT_EQ(result.dtype(), DataType::kFloat32);
}

// ==================== 乘法运算类型提升测试 ====================

TEST_P(TypePromotionTest, MulFloat32Double)
{
    // 测试 float32 * double → float64
    auto a = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    double scalar = 2.0;
    
    auto result = a * scalar;
    
    EXPECT_EQ(result.dtype(), DataType::kFloat64);
}

TEST_P(TypePromotionTest, MulInt32Float32)
{
    // 测试 int32 * float32 → float32
    auto a = Tensor({2, 3, 4, 5}, Shape{2, 2}, dtype(DataType::kInt32).device(deviceType()));
    float scalar = 2.0f;
    
    auto result = a * scalar;
    
    EXPECT_EQ(result.dtype(), DataType::kFloat32);
    
    // 验证值正确
    auto expected = Tensor({4.0f, 6.0f, 8.0f, 10.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 除法运算类型提升测试 ====================

TEST_P(TypePromotionTest, DivFloat32Double)
{
    // 测试 float32 / double → float64
    auto a = Tensor({4.0f, 6.0f, 8.0f, 10.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    double scalar = 2.0;
    
    auto result = a / scalar;
    
    EXPECT_EQ(result.dtype(), DataType::kFloat64);
}

TEST_P(TypePromotionTest, DivInt32Float32)
{
    // 测试 int32 / float32 → float32
    auto a = Tensor({4, 6, 8, 10}, Shape{2, 2}, dtype(DataType::kInt32).device(deviceType()));
    float scalar = 2.0f;
    
    auto result = a / scalar;
    
    EXPECT_EQ(result.dtype(), DataType::kFloat32);
    
    // 验证值正确
    auto expected = Tensor({2.0f, 3.0f, 4.0f, 5.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 标量在左侧的测试 ====================

TEST_P(TypePromotionTest, ScalarLeftSide)
{
    // 测试标量在左侧的类型提升
    auto a = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    double scalar = 2.0;
    
    // 标量在左侧
    auto result1 = scalar + a;
    EXPECT_EQ(result1.dtype(), DataType::kFloat64);
    
    // 标量在右侧
    auto result2 = a + scalar;
    EXPECT_EQ(result2.dtype(), DataType::kFloat64);
    
    // 结果应该相同
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result1, result2, origin::test::TestTolerance::kDefault);
}

TEST_P(TypePromotionTest, ScalarLeftSideIntFloat)
{
    // 测试标量在左侧的整数+浮点类型提升
    auto a = Tensor({1, 2, 3, 4}, Shape{2, 2}, dtype(DataType::kInt32).device(deviceType()));
    float scalar = 2.5f;
    
    // 标量在左侧
    auto result1 = scalar + a;
    EXPECT_EQ(result1.dtype(), DataType::kFloat32);
    
    // 标量在右侧
    auto result2 = a + scalar;
    EXPECT_EQ(result2.dtype(), DataType::kFloat32);
    
    // 结果应该相同
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result1, result2, origin::test::TestTolerance::kDefault);
}

// ==================== 综合测试 ====================

TEST_P(TypePromotionTest, ComplexExpression)
{
    // 测试复杂表达式的类型提升
    auto a = Tensor({1, 2, 3, 4}, Shape{2, 2}, dtype(DataType::kInt32).device(deviceType()));
    float scalar_f = 2.5f;
    double scalar_d = 1.5;
    
    // int32 + float32 → float32, 然后 float32 + double → float64
    auto result = (a + scalar_f) + scalar_d;
    
    EXPECT_EQ(result.dtype(), DataType::kFloat64);
}

// 实例化测试套件：自动为CPU和可用CUDA生成测试
INSTANTIATE_DEVICE_TEST_SUITE_P(TypePromotionTest);

