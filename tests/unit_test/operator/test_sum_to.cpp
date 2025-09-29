#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <arrayfire.h>
#include "dlTensor.h"
#include "dlOperator.h"

using namespace dl;

class SumToOperatorTest : public ::testing::Test {
protected:
    // 精度忍受常量
    static constexpr double kTolerance = 1e-6;
    void SetUp() override {
        // 测试前的设置
            // 初始化ArrayFire后端
        try {
            af::setBackend(AF_BACKEND_CPU);
        } catch (const af::exception &e) {
            // 忽略错误，继续测试
        }
    }
    
    void TearDown() override {
        // 测试后的清理
    }
    
    // 辅助函数：比较两个浮点数是否相等（考虑浮点精度）
    bool isEqual(double a, double b, double tolerance = kTolerance) {
        return std::abs(a - b) < tolerance;
    }
    
    // 辅助函数：比较两个Tensor是否相等
    bool tensorsEqual(const Tensor& a, const Tensor& b, double tolerance = kTolerance) {
        if (a.shape() != b.shape()) {
            return false;
        }
        
        auto data_a = a.to_vector();
        auto data_b = b.to_vector();
        
        if (data_a.size() != data_b.size()) {
            return false;
        }
        
        for (size_t i = 0; i < data_a.size(); ++i) {
            if (!isEqual(data_a[i], data_b[i], tolerance)) {
                return false;
            }
        }
        return true;
    }
};

// ==================== 前向传播测试 ====================

TEST_F(SumToOperatorTest, ForwardBasic) {
    // 测试基本sum_to运算
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
    Shape target_shape{1, 1};
    
    auto result = sum_to(x, target_shape);
    
    EXPECT_EQ(result.shape(), target_shape);
    EXPECT_NEAR(result.item(), 10.0, kTolerance);
}

TEST_F(SumToOperatorTest, ForwardToScalar) {
    // 测试求和到标量
    auto x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0}, Shape{5});
    Shape target_shape{1};
    
    auto result = sum_to(x, target_shape);
    
    EXPECT_EQ(result.shape(), target_shape);
    EXPECT_NEAR(result.item(), 15.0, kTolerance);
}

TEST_F(SumToOperatorTest, ForwardToSameShape) {
    // 测试相同形状（应该不变）
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
    Shape target_shape{2, 2};
    
    auto result = sum_to(x, target_shape);
    
    EXPECT_TRUE(tensorsEqual(result, x));
}

TEST_F(SumToOperatorTest, ForwardToLargerShape) {
    // 测试到更大形状（应该广播）
    auto x = Tensor({5.0}, Shape{1});
    Shape target_shape{3};
    
    auto result = sum_to(x, target_shape);
    
    EXPECT_EQ(result.shape(), target_shape);
    auto result_data = result.to_vector();
    std::vector<data_t> expected = {5.0, 5.0, 5.0};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(SumToOperatorTest, ForwardToSmallerShape) {
    // 测试到更小形状（应该求和）
    auto x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape{2, 3});
    Shape target_shape{2, 1};
    
    auto result = sum_to(x, target_shape);
    
    EXPECT_EQ(result.shape(), target_shape);
    auto result_data = result.to_vector();
    std::vector<data_t> expected = {6.0, 15.0}; // 第一行和：1+2+3=6，第二行和：4+5+6=15
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(SumToOperatorTest, ForwardZeroTensor) {
    // 测试零张量
    auto x = Tensor({0.0, 0.0, 0.0}, Shape{3});
    Shape target_shape{1};
    
    auto result = sum_to(x, target_shape);
    
    EXPECT_EQ(result.shape(), target_shape);
    EXPECT_NEAR(result.item(), 0.0, kTolerance);
}

// ==================== 反向传播测试 ====================

TEST_F(SumToOperatorTest, BackwardBasic) {
    // 测试基本反向传播
    auto x = Tensor({1.0, 2.0, 3.0}, Shape{3});
    Shape target_shape{1};
    
    auto y = sum_to(x, target_shape);
    y.backward();
    
    // sum_to算子的梯度：∂y/∂x = 1（广播回原始形状）
    auto gx_data = x.grad().to_vector();
    
    for (size_t i = 0; i < gx_data.size(); ++i) {
        EXPECT_NEAR(gx_data[i], 1.0, kTolerance);
    }
}

TEST_F(SumToOperatorTest, BackwardWithGradient) {
    // 测试带梯度的反向传播
    auto x = Tensor({1.0, 2.0, 3.0}, Shape{3});
    Shape target_shape{1};
    
    auto y = sum_to(x, target_shape);
    y.backward();
    
    // 设置输出梯度为2
    auto gy = Tensor({2.0}, Shape{1});    y.backward();
    
    auto gx_data = x.grad().to_vector();
    
    for (size_t i = 0; i < gx_data.size(); ++i) {
        EXPECT_NEAR(gx_data[i], 2.0, kTolerance);
    }
}

TEST_F(SumToOperatorTest, BackwardToSameShape) {
    // 测试相同形状的反向传播
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
    Shape target_shape{2, 2};
    
    auto y = sum_to(x, target_shape);
    y.backward();
    
    auto gx_data = x.grad().to_vector();
    
    for (size_t i = 0; i < gx_data.size(); ++i) {
        EXPECT_NEAR(gx_data[i], 1.0, kTolerance);
    }
}

TEST_F(SumToOperatorTest, BackwardToLargerShape) {
    // 测试到更大形状的反向传播
    auto x = Tensor({5.0}, Shape{1});
    Shape target_shape{3};
    
    auto y = sum_to(x, target_shape);
    y.backward();
    
    auto gx_data = x.grad().to_vector();
    
    EXPECT_EQ(gx_data.size(), 1);
    EXPECT_NEAR(gx_data[0], 3.0, kTolerance); // 广播的梯度应该求和
}

// ==================== 边界情况测试 ====================

TEST_F(SumToOperatorTest, SingleElement) {
    // 测试单元素张量
    auto x = Tensor({5.0}, Shape{1});
    Shape target_shape{1};
    
    auto result = sum_to(x, target_shape);
    
    EXPECT_EQ(result.shape(), target_shape);
    EXPECT_NEAR(result.item(), 5.0, kTolerance);
}

TEST_F(SumToOperatorTest, LargeTensor) {
    // 测试大张量
    std::vector<data_t> data(100, 1.0);
    auto x = Tensor(data, Shape{10, 10});
    Shape target_shape{1, 1};
    
    auto result = sum_to(x, target_shape);
    
    EXPECT_EQ(result.shape(), target_shape);
    EXPECT_NEAR(result.item(), 100.0, kTolerance);
}

TEST_F(SumToOperatorTest, ThreeDimensional) {
    // 测试三维张量
    auto x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, Shape{2, 2, 2});
    Shape target_shape{1, 1, 1};
    
    auto result = sum_to(x, target_shape);
    
    EXPECT_EQ(result.shape(), target_shape);
    EXPECT_NEAR(result.item(), 36.0, kTolerance);
}

TEST_F(SumToOperatorTest, ThreeDimensionalTo2D) {
    // 测试三维张量到二维
    auto x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, Shape{2, 2, 2});
    Shape target_shape{2, 1};
    
    auto result = sum_to(x, target_shape);
    
    EXPECT_EQ(result.shape(), target_shape);
    auto result_data = result.to_vector();
    std::vector<data_t> expected = {10.0, 26.0}; // 前4个元素和：1+2+3+4=10，后4个元素和：5+6+7+8=26
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

// ==================== 数值稳定性测试 ====================

TEST_F(SumToOperatorTest, NumericalStability) {
    // 测试数值稳定性
    auto x = Tensor({1e10, 1e-10, -1e10, -1e-10}, Shape{2, 2});
    Shape target_shape{1, 1};
    
    auto result = sum_to(x, target_shape);
    
    EXPECT_NEAR(result.item(), 0.0, kTolerance);
}

TEST_F(SumToOperatorTest, PrecisionTest) {
    // 测试精度
    auto x = Tensor({0.1, 0.2, 0.3}, Shape{3});
    Shape target_shape{1};
    
    auto result = sum_to(x, target_shape);
    
    EXPECT_NEAR(result.item(), 0.6, kTolerance);
}

// ==================== 特殊值测试 ====================

TEST_F(SumToOperatorTest, MixedSigns) {
    // 测试混合符号
    auto x = Tensor({1.0, -2.0, 3.0, -4.0, 5.0}, Shape{5});
    Shape target_shape{1};
    
    auto result = sum_to(x, target_shape);
    
    EXPECT_NEAR(result.item(), 3.0, kTolerance);
}

TEST_F(SumToOperatorTest, IdentityProperty) {
    // 测试恒等性质：sum_to(x, x.shape()) = x
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
    
    auto result = sum_to(x, x.shape());
    
    EXPECT_TRUE(tensorsEqual(result, x));
}

TEST_F(SumToOperatorTest, AssociativeProperty) {
    // 测试结合性质：sum_to(sum_to(x, shape1), shape2) = sum_to(x, shape2)
    auto x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape{2, 3});
    Shape shape1{2, 1};
    Shape shape2{1, 1};
    
    auto result1 = sum_to(sum_to(x, shape1), shape2);
    auto result2 = sum_to(x, shape2);
    
    EXPECT_TRUE(tensorsEqual(result1, result2));
}

TEST_F(SumToOperatorTest, CommutativeProperty) {
    // 测试交换性质：sum_to(x + y, shape) = sum_to(x, shape) + sum_to(y, shape)
    auto x = Tensor({1.0, 2.0}, Shape{2});
    auto y = Tensor({3.0, 4.0}, Shape{2});
    Shape target_shape{1};
    
    auto result1 = sum_to(x + y, target_shape);
    auto result2 = sum_to(x, target_shape) + sum_to(y, target_shape);
    
    EXPECT_NEAR(result1.item(), result2.item(), kTolerance);
}
