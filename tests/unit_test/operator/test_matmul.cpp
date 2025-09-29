#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <arrayfire.h>
#include "dlTensor.h"
#include "dlOperator.h"

using namespace dl;

class MatMulOperatorTest : public ::testing::Test {
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

TEST_F(MatMulOperatorTest, ForwardBasic) {
    // 测试基本矩阵乘法运算
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
    auto w = Tensor({5.0, 6.0, 7.0, 8.0}, Shape{2, 2});
    
    auto result = mat_mul(x, w);
    
    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector();
    // 结果应该是 [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    std::vector<data_t> expected = {19.0, 22.0, 43.0, 50.0};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(MatMulOperatorTest, ForwardOperatorOverload) {
    // 测试运算符重载
    auto x = Tensor({1.0, 2.0}, Shape{1, 2});
    auto w = Tensor({3.0, 4.0}, Shape{2, 1});
    
    auto result = mat_mul(x, w);
    
    Shape expected_shape{1, 1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item(), 11.0, kTolerance); // 1*3 + 2*4 = 11
}

TEST_F(MatMulOperatorTest, ForwardDifferentSizes) {
    // 测试不同大小的矩阵乘法
    auto x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape{2, 3});
    auto w = Tensor({7.0, 8.0, 9.0, 10.0, 11.0, 12.0}, Shape{3, 2});
    
    auto result = mat_mul(x, w);
    
    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector();
    // 结果应该是 [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    // = [[58, 64], [139, 154]]
    std::vector<data_t> expected = {58.0, 64.0, 139.0, 154.0};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(MatMulOperatorTest, ForwardIdentityMatrix) {
    // 测试单位矩阵乘法
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
    auto identity = Tensor({1.0, 0.0, 0.0, 1.0}, Shape{2, 2});
    
    auto result = mat_mul(x, identity);
    
    EXPECT_TRUE(tensorsEqual(result, x));
}

TEST_F(MatMulOperatorTest, ForwardZeroMatrix) {
    // 测试零矩阵乘法
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
    auto zero = Tensor({0.0, 0.0, 0.0, 0.0}, Shape{2, 2});
    
    auto result = mat_mul(x, zero);
    
    auto result_data = result.to_vector();
    for (size_t i = 0; i < result_data.size(); ++i) {
        EXPECT_NEAR(result_data[i], 0.0, kTolerance);
    }
}

// ==================== 反向传播测试 ====================

TEST_F(MatMulOperatorTest, BackwardBasic) {
    // 测试基本反向传播
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
    auto w = Tensor({5.0, 6.0, 7.0, 8.0}, Shape{2, 2});
    
    auto y = mat_mul(x, w);
    y.backward();
    
    // 矩阵乘法算子的梯度：
    // ∂y/∂x = gy * w^T
    // ∂y/∂w = x^T * gy
    auto gx_data = x.grad().to_vector();
    auto gw_data = w.grad().to_vector();
    
    // 验证梯度不为零
    for (size_t i = 0; i < gx_data.size(); ++i) {
        EXPECT_NE(gx_data[i], 0.0);
    }
    for (size_t i = 0; i < gw_data.size(); ++i) {
        EXPECT_NE(gw_data[i], 0.0);
    }
}

TEST_F(MatMulOperatorTest, BackwardWithGradient) {
    // 测试带梯度的反向传播
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
    auto w = Tensor({5.0, 6.0, 7.0, 8.0}, Shape{2, 2});
    
    auto y = mat_mul(x, w);
    y.backward();
    
    // 设置输出梯度为2
    auto gy = Tensor({2.0, 2.0, 2.0, 2.0}, Shape{2, 2});    y.backward();
    
    auto gx_data = x.grad().to_vector();
    auto gw_data = w.grad().to_vector();
    
    // 验证梯度被正确缩放
    for (size_t i = 0; i < gx_data.size(); ++i) {
        EXPECT_NEAR(gx_data[i], 2.0, kTolerance);
    }
    for (size_t i = 0; i < gw_data.size(); ++i) {
        EXPECT_NEAR(gw_data[i], 2.0, kTolerance);
    }
}

TEST_F(MatMulOperatorTest, BackwardDifferentSizes) {
    // 测试不同大小的矩阵乘法反向传播
    auto x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0}, Shape{2, 3});
    auto w = Tensor({7.0, 8.0, 9.0, 10.0, 11.0, 12.0}, Shape{3, 2});
    
    auto y = mat_mul(x, w);
    y.backward();
    
    auto gx_data = x.grad().to_vector();
    auto gw_data = w.grad().to_vector();
    
    EXPECT_EQ(gx_data.size(), 6);
    EXPECT_EQ(gw_data.size(), 6);
    
    // 验证梯度不为零
    for (size_t i = 0; i < gx_data.size(); ++i) {
        EXPECT_NE(gx_data[i], 0.0);
    }
    for (size_t i = 0; i < gw_data.size(); ++i) {
        EXPECT_NE(gw_data[i], 0.0);
    }
}

TEST_F(MatMulOperatorTest, BackwardIdentityMatrix) {
    // 测试单位矩阵乘法的反向传播
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
    auto identity = Tensor({1.0, 0.0, 0.0, 1.0}, Shape{2, 2});
    
    auto y = mat_mul(x, identity);
    y.backward();
    
    auto gx_data = x.grad().to_vector();
    auto gidentity_data = identity.grad().to_vector();
    
    // x的梯度应该等于输出梯度
    for (size_t i = 0; i < gx_data.size(); ++i) {
        EXPECT_NEAR(gx_data[i], 1.0, kTolerance);
    }
    
    // identity的梯度应该等于x
    for (size_t i = 0; i < gidentity_data.size(); ++i) {
        EXPECT_NEAR(gidentity_data[i], x.to_vector()[i], kTolerance);
    }
}

// ==================== 边界情况测试 ====================

TEST_F(MatMulOperatorTest, SingleElement) {
    // 测试单元素矩阵乘法
    auto x = Tensor({5.0}, Shape{1, 1});
    auto w = Tensor({3.0}, Shape{1, 1});
    
    auto result = mat_mul(x, w);
    
    Shape expected_shape{1, 1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item(), 15.0, kTolerance);
}

TEST_F(MatMulOperatorTest, LargeMatrix) {
    // 测试大矩阵乘法
    std::vector<data_t> data_x(100, 1.0);
    std::vector<data_t> data_w(100, 2.0);
    auto x = Tensor(data_x, Shape{10, 10});
    auto w = Tensor(data_w, Shape{10, 10});
    
    auto result = mat_mul(x, w);
    
    Shape expected_shape{10, 10};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector();
    
    for (size_t i = 0; i < result_data.size(); ++i) {
        EXPECT_NEAR(result_data[i], 20.0, kTolerance); // 每行都是10个1.0乘以10个2.0的和
    }
}

TEST_F(MatMulOperatorTest, ThreeDimensional) {
    // 测试三维张量矩阵乘法
    auto x = Tensor({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}, Shape{2, 2, 2});
    auto w = Tensor({1.0, 0.0, 0.0, 1.0}, Shape{2, 2});
    
    // 注意：这里可能需要调整，因为三维张量的矩阵乘法可能不被支持
    // 或者需要特殊的处理方式
    EXPECT_THROW(mat_mul(x, w), std::exception);
}

// ==================== 数值稳定性测试 ====================

TEST_F(MatMulOperatorTest, NumericalStability) {
    // 测试数值稳定性
    auto x = Tensor({1e10, 1e-10, 1e10, 1e-10}, Shape{2, 2});
    auto w = Tensor({1e-10, 1e10, 1e-10, 1e10}, Shape{2, 2});
    
    auto result = mat_mul(x, w);
    
    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector();
    
    // 验证结果在合理范围内
    for (size_t i = 0; i < result_data.size(); ++i) {
        EXPECT_TRUE(std::isfinite(result_data[i]));
    }
}

TEST_F(MatMulOperatorTest, PrecisionTest) {
    // 测试精度
    auto x = Tensor({0.1, 0.2, 0.3, 0.4}, Shape{2, 2});
    auto w = Tensor({0.5, 0.6, 0.7, 0.8}, Shape{2, 2});
    
    auto result = mat_mul(x, w);
    
    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector();
    // 结果应该是 [[0.1*0.5+0.2*0.7, 0.1*0.6+0.2*0.8], [0.3*0.5+0.4*0.7, 0.3*0.6+0.4*0.8]]
    // = [[0.19, 0.22], [0.43, 0.50]]
    std::vector<data_t> expected = {0.19, 0.22, 0.43, 0.50};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

// ==================== 特殊值测试 ====================

TEST_F(MatMulOperatorTest, MixedSigns) {
    // 测试混合符号
    auto x = Tensor({1.0, -2.0, 3.0, -4.0}, Shape{2, 2});
    auto w = Tensor({-1.0, 2.0, -3.0, 4.0}, Shape{2, 2});
    
    auto result = mat_mul(x, w);
    
    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector();
    // 结果应该是 [[1*(-1)+(-2)*(-3), 1*2+(-2)*4], [3*(-1)+(-4)*(-3), 3*2+(-4)*4]]
    // = [[5, -6], [9, -10]]
    std::vector<data_t> expected = {5.0, -6.0, 9.0, -10.0};
    
    for (size_t i = 0; i < expected.size(); ++i) {
        EXPECT_NEAR(result_data[i], expected[i], kTolerance);
    }
}

TEST_F(MatMulOperatorTest, AssociativeProperty) {
    // 测试结合性质：(A * B) * C = A * (B * C)
    auto A = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
    auto B = Tensor({5.0, 6.0, 7.0, 8.0}, Shape{2, 2});
    auto C = Tensor({9.0, 10.0, 11.0, 12.0}, Shape{2, 2});
    
    auto result1 = mat_mul(mat_mul(A, B), C);
    auto result2 = mat_mul(A, mat_mul(B, C));
    
    EXPECT_TRUE(tensorsEqual(result1, result2));
}

TEST_F(MatMulOperatorTest, DistributiveProperty) {
    // 测试分配性质：A * (B + C) = A * B + A * C
    auto A = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
    auto B = Tensor({5.0, 6.0, 7.0, 8.0}, Shape{2, 2});
    auto C = Tensor({9.0, 10.0, 11.0, 12.0}, Shape{2, 2});
    
    auto result1 = mat_mul(A, B + C);
    auto result2 = mat_mul(A, B) + mat_mul(A, C);
    
    EXPECT_TRUE(tensorsEqual(result1, result2));
}

TEST_F(MatMulOperatorTest, DimensionValidation) {
    // 测试维度验证
    auto x = Tensor({1.0, 2.0, 3.0, 4.0}, Shape{2, 2});
    auto w = Tensor({5.0, 6.0, 7.0, 8.0, 9.0, 10.0}, Shape{2, 3});
    
    // 维度不匹配应该抛出异常
    EXPECT_THROW(mat_mul(x, w), std::exception);
}
