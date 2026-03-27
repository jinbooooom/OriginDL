#include <gtest/gtest.h>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"

using namespace origin;
namespace F = origin::functional;

/**
 * @brief MatMul 3D 算子测试类
 */
class MatMul3DTest : public origin::test::OperatorTestBase
{
};

// ==================== 3D x 3D 矩阵乘法测试 ====================

TEST_P(MatMul3DTest, MatMul3D_3D)
{
    // 创建 3D 张量：{2, 2, 3} x {2, 3, 2} -> {2, 2, 2}
    // A: batch 0: [[1, 2, 3], [4, 5, 6]]
    //    batch 1: [[7, 8, 9], [10, 11, 12]]
    // B: batch 0: [[1, 2], [3, 4], [5, 6]]
    //    batch 1: [[7, 8], [9, 10], [11, 12]]

    std::vector<float> a_data = {
        1, 2, 3, 4, 5, 6,    // batch 0
        7, 8, 9, 10, 11, 12  // batch 1
    };
    std::vector<float> b_data = {
        1, 2, 3, 4, 5, 6,    // batch 0
        7, 8, 9, 10, 11, 12  // batch 1
    };

    auto a = Tensor(a_data, Shape{2, 2, 3}, dtype(DataType::kFloat32).device(deviceType()));
    auto b = Tensor(b_data, Shape{2, 3, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto c = F::mat_mul(a, b);

    // 验证输出形状
    auto result_shape = c.shape();
    EXPECT_EQ(result_shape.size(), 3);
    EXPECT_EQ(result_shape[0], 2);
    EXPECT_EQ(result_shape[1], 2);
    EXPECT_EQ(result_shape[2], 2);

    // 验证输出值
    auto c_vec = c.to_vector<float>();

    // batch 0:
    // [1, 2, 3]   [1, 2]     [1*1+2*3+3*5, 1*2+2*4+3*6]     [22, 28]
    // [4, 5, 6] x [3, 4]  =  [4*1+5*3+6*5, 4*2+5*4+6*6]  =  [49, 64]
    //            [5, 6]

    // batch 1:
    // [7,  8,  9]   [7,  8]     [7*7+8*9+9*11,  7*8+8*10+9*12]     [220, 244]
    // [10, 11, 12] x [9, 10]  =  [10*7+11*9+12*11, 10*8+11*10+12*12] = [301, 334]
    //               [11, 12]

    std::vector<float> expected = {
        22, 28, 49, 64,    // batch 0
        220, 244, 301, 334  // batch 1
    };

    for (size_t i = 0; i < c_vec.size(); ++i)
    {
        EXPECT_NEAR(c_vec[i], expected[i], 1e-5);
    }
}

// ==================== 3D x 2D 矩阵乘法测试 ====================

TEST_P(MatMul3DTest, MatMul3D_2D)
{
    // 创建 3D 张量：{2, 2, 3} x {3, 2} -> {2, 2, 2}
    std::vector<float> a_data = {
        1, 2, 3, 4, 5, 6,    // batch 0
        7, 8, 9, 10, 11, 12  // batch 1
    };
    std::vector<float> b_data = {
        1, 2, 3, 4, 5, 6     // {3, 2}
    };

    auto a = Tensor(a_data, Shape{2, 2, 3}, dtype(DataType::kFloat32).device(deviceType()));
    auto b = Tensor(b_data, Shape{3, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto c = F::mat_mul(a, b);

    // 验证输出形状
    auto result_shape = c.shape();
    EXPECT_EQ(result_shape.size(), 3);
    EXPECT_EQ(result_shape[0], 2);
    EXPECT_EQ(result_shape[1], 2);
    EXPECT_EQ(result_shape[2], 2);

    // 验证输出值
    auto c_vec = c.to_vector<float>();

    // batch 0:
    // [1, 2, 3]   [1, 2]     [22, 28]
    // [4, 5, 6] x [3, 4]  =  [49, 64]
    //            [5, 6]

    // batch 1:
    // [7,  8, 9]   [1, 2]     [7*1+8*3+9*5,  7*2+8*4+9*6]     [76, 100]
    // [10, 11, 12] x [3, 4]  =  [10*1+11*3+12*5, 10*2+11*4+12*6] = [103, 136]
    //              [5, 6]

    std::vector<float> expected = {
        22, 28, 49, 64,    // batch 0
        76, 100, 103, 136  // batch 1
    };

    for (size_t i = 0; i < c_vec.size(); ++i)
    {
        EXPECT_NEAR(c_vec[i], expected[i], 1e-5);
    }
}

// Instantiate test suite: automatically generate tests for CPU and available CUDA
INSTANTIATE_DEVICE_TEST_SUITE_P(MatMul3DTest);

