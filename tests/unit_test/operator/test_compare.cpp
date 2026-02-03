#include <gtest/gtest.h>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"
#include "origin/operators/math/compare.h"

using namespace origin;

/**
 * @brief 比较运算符测试类（参数化版本）
 * @details 使用参数化测试，自动为CPU和CUDA生成测试用例
 *          测试所有比较运算符：==, !=, <, <=, >, >=
 */
class CompareOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 等于运算符 (==) 测试 ====================

TEST_P(CompareOperatorTest, EqualOperatorBasic)
{
    auto a = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto result = a == Scalar(2.0f);

    EXPECT_EQ(result.shape(), Shape({2, 2}));
    EXPECT_EQ(result.dtype(), DataType::kFloat32);

    auto expected = Tensor({0.0f, 1.0f, 0.0f, 0.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(CompareOperatorTest, EqualOperatorAllTrue)
{
    auto a = Tensor({5.0f, 5.0f, 5.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    auto result = a == Scalar(5.0f);

    auto expected = Tensor({1.0f, 1.0f, 1.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(CompareOperatorTest, EqualOperatorAllFalse)
{
    auto a = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    auto result = a == Scalar(0.0f);

    auto expected = Tensor({0.0f, 0.0f, 0.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 不等于运算符 (!=) 测试 ====================

TEST_P(CompareOperatorTest, NotEqualOperatorBasic)
{
    auto a = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto result = a != Scalar(2.0f);

    auto expected = Tensor({1.0f, 0.0f, 1.0f, 1.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(CompareOperatorTest, NotEqualOperatorAllTrue)
{
    auto a = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    auto result = a != Scalar(0.0f);

    auto expected = Tensor({1.0f, 1.0f, 1.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 小于运算符 (<) 测试 ====================

TEST_P(CompareOperatorTest, LessThanOperatorBasic)
{
    auto a = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto result = a < Scalar(3.0f);

    auto expected = Tensor({1.0f, 1.0f, 0.0f, 0.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(CompareOperatorTest, LessThanOperatorAllFalse)
{
    auto a = Tensor({5.0f, 6.0f, 7.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    auto result = a < Scalar(3.0f);

    auto expected = Tensor({0.0f, 0.0f, 0.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 小于等于运算符 (<=) 测试 ====================

TEST_P(CompareOperatorTest, LessEqualOperatorBasic)
{
    auto a = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto result = a <= Scalar(3.0f);

    auto expected = Tensor({1.0f, 1.0f, 1.0f, 0.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(CompareOperatorTest, LessEqualOperatorEdgeCase)
{
    auto a = Tensor({3.0f, 3.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    auto result = a <= Scalar(3.0f);

    auto expected = Tensor({1.0f, 1.0f, 1.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 大于运算符 (>) 测试 ====================

TEST_P(CompareOperatorTest, GreaterThanOperatorBasic)
{
    auto a = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto result = a > Scalar(2.0f);

    auto expected = Tensor({0.0f, 0.0f, 1.0f, 1.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(CompareOperatorTest, GreaterThanOperatorAllTrue)
{
    auto a = Tensor({5.0f, 6.0f, 7.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    auto result = a > Scalar(3.0f);

    auto expected = Tensor({1.0f, 1.0f, 1.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 大于等于运算符 (>=) 测试 ====================

TEST_P(CompareOperatorTest, GreaterEqualOperatorBasic)
{
    auto a = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto result = a >= Scalar(2.0f);

    auto expected = Tensor({0.0f, 1.0f, 1.0f, 1.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(CompareOperatorTest, GreaterEqualOperatorEdgeCase)
{
    auto a = Tensor({2.0f, 2.0f, 2.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    auto result = a >= Scalar(2.0f);

    auto expected = Tensor({1.0f, 1.0f, 1.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 综合测试 ====================

TEST_P(CompareOperatorTest, CompareOperatorsConsistency)
{
    auto a = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    Scalar threshold(2.0f);

    auto eq_result = a == threshold;
    auto ne_result = a != threshold;
    auto lt_result = a < threshold;
    auto le_result = a <= threshold;
    auto gt_result = a > threshold;
    auto ge_result = a >= threshold;

    // 验证结果形状和类型
    EXPECT_EQ(eq_result.shape(), Shape({3}));
    EXPECT_EQ(ne_result.shape(), Shape({3}));
    EXPECT_EQ(lt_result.shape(), Shape({3}));
    EXPECT_EQ(le_result.shape(), Shape({3}));
    EXPECT_EQ(gt_result.shape(), Shape({3}));
    EXPECT_EQ(ge_result.shape(), Shape({3}));

    // 验证逻辑一致性：eq + ne 应该全为1
    auto eq_ne_sum = eq_result + ne_result;
    auto expected_all_one = Tensor({1.0f, 1.0f, 1.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(eq_ne_sum, expected_all_one, origin::test::TestTolerance::kDefault);

    // 验证逻辑一致性：le + gt 应该全为1
    auto le_gt_sum = le_result + gt_result;
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(le_gt_sum, expected_all_one, origin::test::TestTolerance::kDefault);
}

TEST_P(CompareOperatorTest, CompareOperatorsWithNegativeValues)
{
    auto a = Tensor({-2.0f, -1.0f, 0.0f, 1.0f, 2.0f}, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));
    auto result = a > Scalar(0.0f);

    auto expected = Tensor({0.0f, 0.0f, 0.0f, 1.0f, 1.0f}, Shape{5}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(CompareOperatorTest, CompareOperatorsWithHighDimensionalTensor)
{
    auto a = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));
    auto result = a >= Scalar(4.0f);

    auto expected = Tensor({0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f}, Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// 实例化测试套件
INSTANTIATE_DEVICE_TEST_SUITE_P(CompareOperatorTest);
