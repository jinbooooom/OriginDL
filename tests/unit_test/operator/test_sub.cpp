#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <arrayfire.h>
#include "dlTensor.h"
#include "dlOperator.h"

using namespace dl;

class SubOperatorTest : public ::testing::Test {
protected:
    // 精度忍受常量
    static constexpr double kTolerance = 1e-6;
    void SetUp() override {
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

TEST_F(SubOperatorTest, ForwardBasic) {
    // 测试基本减法运算
    auto x0 = Tensor({5.0, 6.0, 7.0, 8.0}, Shape{2, 2});
    auto x1 = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
    
    auto result = sub(x0, x1);
    
    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector();
    std::vector<data_t> expected = {4.0, 4.0, 4.0, 4.0};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(SubOperatorTest, ForwardOperatorOverload) {
    // 测试运算符重载
    auto x0 = Tensor({5.0, 6.0}, Shape{2});
    auto x1 = Tensor({1.0, 2.0}, Shape{2});
    
    auto result = x0 - x1;
    
    Shape expected_shape{2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector();
    std::vector<data_t> expected = {4.0, 4.0};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(SubOperatorTest, ForwardScalarTensor) {
    // 测试标量与张量的减法
    auto x = Tensor({5.0, 6.0, 7.0}, Shape{3});
    data_t scalar = 2.0;
    
    auto result1 = x - scalar;
    auto result2 = scalar - x;
    
    EXPECT_EQ(result1.shape(), Shape{3});
    EXPECT_EQ(result2.shape(), Shape{3});
    
    auto data1 = result1.to_vector();
    auto data2 = result2.to_vector();
    std::vector<data_t> expected1 = {3.0, 4.0, 5.0};
    std::vector<data_t> expected2 = {-3.0, -4.0, -5.0};
    
    for (size_t i = 0; i < expected1.size(); ++i) {
        EXPECT_NEAR(data1[i], expected1[i], kTolerance);
        EXPECT_NEAR(data2[i], expected2[i], kTolerance);
    }
}

TEST_F(SubOperatorTest, ForwardZeroTensor) {
    // 测试零张量减法
    auto x0 = Tensor({1.0, 2.0}, Shape{2});
    auto x1 = Tensor({0.0, 0.0}, Shape{2});
    
    auto result = sub(x0, x1);
    
    EXPECT_TRUE(tensorsEqual(result, x0));
}

TEST_F(SubOperatorTest, ForwardNegativeValues) {
    // 测试负值减法
    auto x0 = Tensor({-1.0, -2.0}, Shape{2});
    auto x1 = Tensor({3.0, 4.0}, Shape{2});
    
    auto result = sub(x0, x1);
    
    auto result_data = result.to_vector();
    std::vector<data_t> expected = {-4.0, -6.0};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

// ==================== 反向传播测试 ====================

TEST_F(SubOperatorTest, BackwardBasic) {
    // 测试基本反向传播
    auto x0 = Tensor({1.0, 2.0}, Shape{2});
    auto x1 = Tensor({3.0, 4.0}, Shape{2});
    
    auto y = sub(x0, x1);
    y.backward();
    
    // 减法算子的梯度：∂y/∂x0 = 1, ∂y/∂x1 = -1
    auto gx0_data = x0.grad().to_vector();
    auto gx1_data = x1.grad().to_vector();
    
    for (size_t i = 0; i < gx0_data.size(); ++i) {
        EXPECT_NEAR(gx0_data[i], 1.0, kTolerance);
        EXPECT_NEAR(gx1_data[i], -1.0, kTolerance);
    }
}

TEST_F(SubOperatorTest, BackwardWithGradient) {
    // 测试带梯度的反向传播
    auto x0 = Tensor({2.0, 3.0}, Shape{2});
    auto x1 = Tensor({1.0, 1.0}, Shape{2});
    
    auto y = sub(x0, x1);
    
    // 减法算子的梯度：∂y/∂x0 = 1, ∂y/∂x1 = -1
    y.backward();
    
    auto gx0_data = x0.grad().to_vector();
    auto gx1_data = x1.grad().to_vector();
    
    for (size_t i = 0; i < gx0_data.size(); ++i) {
        EXPECT_NEAR(gx0_data[i], 1.0, kTolerance);
        EXPECT_NEAR(gx1_data[i], -1.0, kTolerance);
    }
}

TEST_F(SubOperatorTest, BackwardDifferentShapes) {
    // 测试不同形状的张量减法反向传播
    auto x0 = Tensor({1.0, 2.0}, Shape{2});
    auto x1 = Tensor({3.0}, Shape{1});
    
    auto y = sub(x0, x1);
    y.backward();
    
    // 梯度应该正确广播
    auto gx0_data = x0.grad().to_vector();
    auto gx1_data = x1.grad().to_vector();
    
    EXPECT_EQ(gx0_data.size(), 2);
    EXPECT_EQ(gx1_data.size(), 1);
    
    for (size_t i = 0; i < gx0_data.size(); ++i) {
        EXPECT_NEAR(gx0_data[i], 1.0, kTolerance);
    }
    EXPECT_NEAR(gx1_data[0], -2.0, kTolerance); // 广播后的梯度
}

// ==================== 边界情况测试 ====================

TEST_F(SubOperatorTest, SingleElement) {
    // 测试单元素张量
    auto x0 = Tensor({5.0}, Shape{1});
    auto x1 = Tensor({3.0}, Shape{1});
    
    auto result = sub(x0, x1);
    
    Shape expected_shape{1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item(), 2.0, kTolerance);
}

TEST_F(SubOperatorTest, LargeTensor) {
    // 测试大张量
    std::vector<data_t> data1(100, 5.0);
    std::vector<data_t> data2(100, 2.0);
    auto x0 = Tensor(data1, Shape{10, 10});
    auto x1 = Tensor(data2, Shape{10, 10});
    
    auto result = sub(x0, x1);
    
    Shape expected_shape{10, 10};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector();
    
    for (size_t i = 0; i < result_data.size(); ++i) {
        EXPECT_NEAR(result_data[i], 3.0, kTolerance);
    }
}

TEST_F(SubOperatorTest, ThreeDimensional) {
    // 测试三维张量
    auto x0 = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, Shape{2, 2, 2});
    auto x1 = Tensor({0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}, Shape{2, 2, 2});
    
    auto result = sub(x0, x1);
    
    Shape expected_shape{2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector();
    
    // 期望值：x0[i] - x1[i]
    std::vector<double> expected = {0.9, 1.8, 2.7, 3.6, 4.5, 5.4, 6.3, 7.2};
    
    for (size_t i = 0; i < result_data.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

// ==================== 数值稳定性测试 ====================

TEST_F(SubOperatorTest, NumericalStability) {
    // 测试数值稳定性
    auto x0 = Tensor({1e10, 1e-10}, Shape{2});
    auto x1 = Tensor({1e-10, 1e10}, Shape{2});
    
    auto result = sub(x0, x1);
    
    auto result_data = result.to_vector();
    std::vector<data_t> expected = {1e10, -1e10};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(SubOperatorTest, PrecisionTest) {
    // 测试精度
    auto x0 = Tensor({0.3, 0.4}, Shape{2});
    auto x1 = Tensor({0.1, 0.2}, Shape{2});
    
    auto result = sub(x0, x1);
    
    auto result_data = result.to_vector();
    std::vector<data_t> expected = {0.2, 0.2};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}
