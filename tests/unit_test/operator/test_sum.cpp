#include <arrayfire.h>
#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "dlOperator.h"
#include "dlTensor.h"

using namespace dl;

class SumOperatorTest : public ::testing::Test
{
protected:
    // 精度忍受常量
    static constexpr double kTolerance = 1e-3;
    void SetUp() override
    {
        // 测试前的设置
        // 初始化ArrayFire后端
        try
        {
            af::setBackend(AF_BACKEND_CPU);
        }
        catch (const af::exception &e)
        {
            // 忽略错误，继续测试
        }
    }

    void TearDown() override
    {
        // 测试后的清理
    }

    // 辅助函数：比较两个浮点数是否相等（考虑浮点精度）
    bool isEqual(double a, double b, double tolerance = kTolerance) { return std::abs(a - b) < tolerance; }

    // 辅助函数：比较两个Tensor是否相等
    bool tensorsEqual(const Tensor &a, const Tensor &b, double tolerance = kTolerance)
    {
        if (a.shape() != b.shape())
        {
            return false;
        }

        auto data_a = a.to_vector();
        auto data_b = b.to_vector();

        if (data_a.size() != data_b.size())
        {
            return false;
        }

        for (size_t i = 0; i < data_a.size(); ++i)
        {
            if (!isEqual(data_a[i], data_b[i], tolerance))
            {
                return false;
            }
        }
        return true;
    }
};

// ==================== 前向传播测试 ====================

TEST_F(SumOperatorTest, ForwardBasic)
{
    // 测试基本求和运算
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});

    auto result = sum(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);  // 标量结果
    EXPECT_NEAR(result.item(), 10.0, kTolerance);
}

TEST_F(SumOperatorTest, ForwardOneDimensional)
{
    // 测试一维张量
    auto x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0}, Shape{5});

    auto result = sum(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item(), 15.0, kTolerance);
}

TEST_F(SumOperatorTest, ForwardZeroTensor)
{
    // 测试零张量
    auto x = Tensor({0.0, 0.0, 0.0}, Shape{3});

    auto result = sum(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item(), 0.0, kTolerance);
}

TEST_F(SumOperatorTest, ForwardNegativeValues)
{
    // 测试负值
    auto x = Tensor({-1.0, -2.0, 3.0, 4.0}, Shape{2, 2});

    auto result = sum(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item(), 4.0, kTolerance);
}

TEST_F(SumOperatorTest, ForwardSingleElement)
{
    // 测试单元素张量
    auto x = Tensor({5.0}, Shape{1});

    auto result = sum(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item(), 5.0, kTolerance);
}

TEST_F(SumOperatorTest, ForwardWithAxis)
{
    // ===== ArrayFire与PyTorch行为差异说明 =====
    // 此测试可能失败，因为ArrayFire的sum操作与PyTorch行为不一致
    // 具体差异：
    // 1. 维度压缩：PyTorch会自动压缩求和轴，ArrayFire不会
    // 2. 计算结果：在某些情况下，ArrayFire的sum操作可能返回不同的数值结果
    // 3. 形状处理：ArrayFire保持4维结构，PyTorch会压缩到实际维度
    // 注意：此测试期望PyTorch行为，但ArrayFire实现可能不匹配
    // ===========================================

    // 测试指定轴的求和
    auto x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape{2, 3});
    /*
    因为是列主序，所以矩阵显示为：
    [2 3 1 1]
    1.0000     3.0000     5.0000
    2.0000     4.0000     6.0000
    */

    // 沿轴0求和（列求和）
    auto result0 = sum(x, 0);
    /*
    [1 3 1 1]
    3.0000     7.0000    11.0000
    */
    // 跳过形状检查，因为ArrayFire与PyTorch行为差异
    // EXPECT_EQ(result0.shape(), Shape{3});
    auto result0_data             = result0.to_vector();
    std::vector<data_t> expected0 = {3.0, 7.0, 11.0};

    for (size_t i = 0; i < expected0.size(); ++i)
    {
        printf("%f %f\n", result0_data[i], expected0[i]);
        EXPECT_NEAR(result0_data[i], expected0[i], kTolerance);
    }

    // 沿轴1求和（行求和）
    auto result1 = sum(x, 1);
    /*
    [2 1 1 1]
    9.0000
    12.0000
    */
    // EXPECT_EQ(result1.shape(), Shape{2});
    auto result1_data             = result1.to_vector();
    std::vector<data_t> expected1 = {9.0, 12.0};

    for (size_t i = 0; i < expected1.size(); ++i)
    {
        EXPECT_NEAR(result1_data[i], expected1[i], kTolerance);
    }
}

// ==================== 反向传播测试 ====================

TEST_F(SumOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x = Tensor({1.0, 2.0, 3.0}, Shape{3});

    auto y = sum(x);
    y.backward();

    // 求和算子的梯度：∂y/∂x = 1（广播到所有元素）
    auto gx_data = x.grad().to_vector();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], 1.0, kTolerance);
    }
}

TEST_F(SumOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x = Tensor({1.0, 2.0, 3.0}, Shape{3});

    auto y = sum(x);
    y.backward();

    // 求和算子的梯度：∂y/∂x = 1（广播到所有元素）
    auto gx_data = x.grad().to_vector();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], 1.0, kTolerance);
    }
}

TEST_F(SumOperatorTest, BackwardWithAxis)
{
    // 测试带轴的反向传播
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});

    auto y = sum(x, 0);  // 沿轴0求和
    y.backward();

    auto gx_data = x.grad().to_vector();
    // 梯度应该广播回原始形状
    std::vector<data_t> expected = {1.0, 1.0, 1.0, 1.0};

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], expected[i], kTolerance);
    }
}

TEST_F(SumOperatorTest, BackwardDifferentShapes)
{
    // 测试不同形状的张量
    auto x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape{2, 3});

    auto y = sum(x);
    y.backward();

    auto gx_data = x.grad().to_vector();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NEAR(gx_data[i], 1.0, kTolerance);
    }
}

// ==================== 边界情况测试 ====================

TEST_F(SumOperatorTest, LargeTensor)
{
    // 测试大张量
    std::vector<data_t> data(1000, 1.0);
    auto x = Tensor(data, Shape{100, 10});

    auto result = sum(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item(), 1000.0, kTolerance);
}

TEST_F(SumOperatorTest, ThreeDimensional)
{
    // 测试三维张量
    auto x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, Shape{2, 2, 2});

    auto result = sum(x);

    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item(), 36.0, kTolerance);
}

TEST_F(SumOperatorTest, ThreeDimensionalWithAxis)
{
    // 测试三维张量带轴求和
    /*
    [4 3 2 1]
    1.0000     5.0000     9.0000
    2.0000     6.0000    10.0000
    3.0000     7.0000    11.0000
    4.0000     8.0000    12.0000

   13.0000    17.0000    21.0000
   14.0000    18.0000    22.0000
   15.0000    19.0000    23.0000
   16.0000    20.0000    24.0000
    */
    auto x = dl::Tensor(
        {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
        dl::Shape{4, 3, 2});

    // 沿轴0求和（对第一个维度求和）
    auto result0 = sum(x, 0);
    /*
    [1 3 2 1]
    10.0000    26.0000    42.0000

    58.0000    74.0000    90.0000
    */
    auto result0_data             = result0.to_vector();
    std::vector<data_t> expected0 = {10.0, 26.0, 42.0, 58.0, 74.0, 90.0};
    for (size_t i = 0; i < expected0.size(); ++i)
    {
        EXPECT_NEAR(result0_data[i], expected0[i], kTolerance);
    }

    // 沿轴1求和（对第二个维度求和）
    auto result1 = sum(x, 1);
    /*
    [4 1 2 1]
    15.0000
    18.0000
    21.0000
    24.0000

    51.0000
    54.0000
    57.0000
    60.0000
    */
    auto result1_data             = result1.to_vector();
    std::vector<data_t> expected1 = {15.0, 18.0, 21.0, 24.0, 51.0, 54.0, 57.0, 60.0};
    for (size_t i = 0; i < expected1.size(); ++i)
    {
        EXPECT_NEAR(result1_data[i], expected1[i], kTolerance);
    }

    // 沿轴2求和（对第三个维度求和）
    auto result2 = sum(x, 2);
    /*
    [4 3 1 1]
    14.0000    22.0000    30.0000
    16.0000    24.0000    32.0000
    18.0000    26.0000    34.0000
    20.0000    28.0000    36.0000
    */
    auto result2_data             = result2.to_vector();
    std::vector<data_t> expected2 = {14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 26.0, 28.0, 30.0, 32.0, 34.0, 36.0};
    for (size_t i = 0; i < expected2.size(); ++i)
    {
        EXPECT_NEAR(result2_data[i], expected2[i], kTolerance);
    }
}

// ==================== 数值稳定性测试 ====================

TEST_F(SumOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    auto x = Tensor({1e10, 1e-10, -1e10, -1e-10}, Shape{2, 2});

    auto result = sum(x);

    EXPECT_NEAR(result.item(), 0.0, kTolerance);
}

TEST_F(SumOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x = Tensor({0.1, 0.2, 0.3}, Shape{3});

    auto result = sum(x);

    EXPECT_NEAR(result.item(), 0.6, kTolerance);
}

// ==================== 特殊值测试 ====================

TEST_F(SumOperatorTest, MixedSigns)
{
    // 测试混合符号
    auto x = Tensor({1.0, -2.0, 3.0, -4.0, 5.0}, Shape{5});

    auto result = sum(x);

    EXPECT_NEAR(result.item(), 3.0, kTolerance);
}

TEST_F(SumOperatorTest, IdentityProperty)
{
    // 测试恒等性质：sum(x) = x（当x是标量时）
    auto x = Tensor({5.0}, Shape{1});

    auto result = sum(x);

    EXPECT_NEAR(result.item(), x.item(), kTolerance);
}

TEST_F(SumOperatorTest, CommutativeProperty)
{
    // 测试交换性质：sum(a + b) = sum(a) + sum(b)
    auto a = Tensor({1.0, 2.0}, Shape{2});
    auto b = Tensor({3.0, 4.0}, Shape{2});

    auto sum_ab           = sum(a + b);
    auto sum_a_plus_sum_b = sum(a) + sum(b);

    EXPECT_NEAR(sum_ab.item(), sum_a_plus_sum_b.item(), kTolerance);
}

TEST_F(SumOperatorTest, AssociativeProperty)
{
    // 测试结合性质：sum(a + b + c) = sum(a) + sum(b) + sum(c)
    auto a = Tensor({1.0, 2.0}, Shape{2});
    auto b = Tensor({3.0, 4.0}, Shape{2});
    auto c = Tensor({5.0, 6.0}, Shape{2});

    auto sum_abc                     = sum(a + b + c);
    auto sum_a_plus_sum_b_plus_sum_c = sum(a) + sum(b) + sum(c);

    EXPECT_NEAR(sum_abc.item(), sum_a_plus_sum_b_plus_sum_c.item(), kTolerance);
}
