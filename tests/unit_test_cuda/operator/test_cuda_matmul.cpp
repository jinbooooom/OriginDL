#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "origin.h"

using namespace origin;

/**
 * @brief CUDA矩阵乘法算子测试类
 * @details 测试CUDA张量的矩阵乘法运算
 */
class CudaMatmulTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // CUDA matmul算子已实现，可以进行测试
    }

    void TearDown() override
    {
        // 清理CUDA资源
        cudaDeviceSynchronize();
    }

    // 精度容忍常量
    static constexpr double kFloatTolerance = 1e-5;
};

// ============================================================================
// 基础矩阵乘法测试
// ============================================================================

TEST_F(CudaMatmulTest, BasicMatrixMultiplication)
{
    // 测试基本矩阵乘法运算
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{2.0f, 0.0f, 1.0f, 2.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto result = origin::mat_mul(a, b);

    EXPECT_EQ(result.shape(), Shape({2, 2}));
    EXPECT_EQ(result.dtype(), DataType::kFloat32);
    EXPECT_EQ(result.device().type(), DeviceType::kCUDA);

    auto result_data            = result.to_vector<float>();
    std::vector<float> expected = {4.0f, 4.0f, 10.0f, 8.0f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaMatmulTest, MatrixMultiplicationWithIdentity)
{
    // 测试与单位矩阵的乘法
    auto a        = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto identity = Tensor(std::vector<float>{1.0f, 0.0f, 0.0f, 1.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto result = origin::mat_mul(a, identity);

    auto result_data   = result.to_vector<float>();
    auto original_data = a.to_vector<float>();

    for (size_t i = 0; i < original_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], original_data[i], kFloatTolerance);
    }
}

TEST_F(CudaMatmulTest, MatrixMultiplicationWithZero)
{
    // 测试与零矩阵的乘法
    auto a    = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto zero = Tensor(std::vector<float>{0.0f, 0.0f, 0.0f, 0.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto result = origin::mat_mul(a, zero);

    auto result_data = result.to_vector<float>();

    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], 0.0f, kFloatTolerance);
    }
}

// ============================================================================
// 不同形状矩阵乘法测试
// ============================================================================

TEST_F(CudaMatmulTest, RectangularMatrixMultiplication)
{
    // 测试矩形矩阵乘法 (2x3) * (3x2) = (2x2)
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{3, 2}, dtype(Float32).device(kCUDA));

    auto result = origin::mat_mul(a, b);

    EXPECT_EQ(result.shape(), Shape({2, 2}));

    auto result_data            = result.to_vector<float>();
    std::vector<float> expected = {22.0f, 28.0f, 49.0f, 64.0f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaMatmulTest, VectorMatrixMultiplication)
{
    // 测试向量与矩阵的乘法 (1x2) * (2x3) = (1x3)
    auto a = Tensor(std::vector<float>{1.0f, 2.0f}, Shape{1, 2}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}, dtype(Float32).device(kCUDA));

    auto result = origin::mat_mul(a, b);

    EXPECT_EQ(result.shape(), Shape({1, 3}));

    auto result_data            = result.to_vector<float>();
    std::vector<float> expected = {9.0f, 12.0f, 15.0f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaMatmulTest, MatrixVectorMultiplication)
{
    // 测试矩阵与向量的乘法 (2x3) * (3x1) = (2x1)
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f}, Shape{3, 1}, dtype(Float32).device(kCUDA));

    auto result = origin::mat_mul(a, b);

    EXPECT_EQ(result.shape(), Shape({2, 1}));

    auto result_data            = result.to_vector<float>();
    std::vector<float> expected = {14.0f, 32.0f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

// ============================================================================
// 边界情况测试
// ============================================================================

TEST_F(CudaMatmulTest, SingleElementMatrixMultiplication)
{
    // 测试单元素矩阵乘法 (1x1) * (1x1) = (1x1)
    auto a = Tensor(std::vector<float>{5.0f}, Shape{1, 1}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{3.0f}, Shape{1, 1}, dtype(Float32).device(kCUDA));

    auto result = origin::mat_mul(a, b);

    EXPECT_EQ(result.shape(), Shape({1, 1}));
    auto result_data = result.to_vector<float>();
    EXPECT_NEAR(result_data[0], 15.0f, kFloatTolerance);
}

TEST_F(CudaMatmulTest, LargeMatrixMultiplication)
{
    // 测试大矩阵乘法
    const size_t m = 64, n = 64, k = 64;
    std::vector<float> data_a(m * k);
    std::vector<float> data_b(k * n);

    // 初始化矩阵A为递增序列
    for (size_t i = 0; i < m * k; ++i)
    {
        data_a[i] = static_cast<float>(i + 1);
    }

    // 初始化矩阵B为递减序列
    for (size_t i = 0; i < k * n; ++i)
    {
        data_b[i] = static_cast<float>(k * n - i);
    }

    auto a = Tensor(data_a, Shape{m, k}, dtype(Float32).device(kCUDA));
    auto b = Tensor(data_b, Shape{k, n}, dtype(Float32).device(kCUDA));

    auto result = origin::mat_mul(a, b);

    EXPECT_EQ(result.shape(), Shape({m, n}));

    // 验证结果形状和基本性质
    auto result_data = result.to_vector<float>();
    EXPECT_EQ(result_data.size(), m * n);

    // 检查结果不为零（对于非零输入矩阵）
    bool has_non_zero = false;
    for (size_t i = 0; i < std::min(result_data.size(), size_t(100)); ++i)
    {
        if (std::abs(result_data[i]) > kFloatTolerance)
        {
            has_non_zero = true;
            break;
        }
    }
    EXPECT_TRUE(has_non_zero);
}

// ============================================================================
// 数值稳定性测试
// ============================================================================

TEST_F(CudaMatmulTest, SmallNumbersMatrixMultiplication)
{
    // 测试小数值矩阵乘法
    auto a = Tensor(std::vector<float>{0.1f, 0.2f, 0.3f, 0.4f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{0.01f, 0.02f, 0.03f, 0.04f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto result = origin::mat_mul(a, b);

    auto result_data            = result.to_vector<float>();
    std::vector<float> expected = {0.007f, 0.01f, 0.015f, 0.022f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

TEST_F(CudaMatmulTest, LargeNumbersMatrixMultiplication)
{
    // 测试大数值矩阵乘法
    auto a = Tensor(std::vector<float>{1000.0f, 2000.0f, 3000.0f, 4000.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{0.001f, 0.002f, 0.003f, 0.004f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto result = origin::mat_mul(a, b);

    auto result_data            = result.to_vector<float>();
    std::vector<float> expected = {7.0f, 10.0f, 15.0f, 22.0f};

    for (size_t i = 0; i < expected.size(); ++i)
    {
        EXPECT_NEAR(result_data[i], expected[i], kFloatTolerance);
    }
}

// ============================================================================
// 数学性质测试
// ============================================================================

TEST_F(CudaMatmulTest, MatrixMultiplicationAssociativity)
{
    // 测试矩阵乘法的结合律：(AB)C = A(BC)
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{2.0f, 0.0f, 1.0f, 2.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto c = Tensor(std::vector<float>{1.0f, 1.0f, 0.0f, 1.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto ab  = origin::mat_mul(a, b);
    auto abc = origin::mat_mul(ab, c);

    auto bc   = origin::mat_mul(b, c);
    auto abc2 = origin::mat_mul(a, bc);

    auto data1 = abc.to_vector<float>();
    auto data2 = abc2.to_vector<float>();

    for (size_t i = 0; i < data1.size(); ++i)
    {
        EXPECT_NEAR(data1[i], data2[i], kFloatTolerance);
    }
}

TEST_F(CudaMatmulTest, MatrixMultiplicationDistributivity)
{
    // 测试矩阵乘法的分配律：A(B + C) = AB + AC
    auto a = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto b = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto c = Tensor(std::vector<float>{2.0f, 1.0f, 1.0f, 2.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));

    auto b_plus_c         = b + c;
    auto a_times_b_plus_c = origin::mat_mul(a, b_plus_c);

    auto ab         = origin::mat_mul(a, b);
    auto ac         = origin::mat_mul(a, c);
    auto ab_plus_ac = ab + ac;

    auto data1 = a_times_b_plus_c.to_vector<float>();
    auto data2 = ab_plus_ac.to_vector<float>();

    for (size_t i = 0; i < data1.size(); ++i)
    {
        EXPECT_NEAR(data1[i], data2[i], kFloatTolerance);
    }
}

TEST_F(CudaMatmulTest, MatrixMultiplicationWithScalar)
{
    // 测试矩阵乘法与标量乘法的关系：k(AB) = (kA)B = A(kB)
    auto a  = Tensor(std::vector<float>{1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    auto b  = Tensor(std::vector<float>{2.0f, 0.0f, 1.0f, 2.0f}, Shape{2, 2}, dtype(Float32).device(kCUDA));
    float k = 3.0f;

    auto ab   = origin::mat_mul(a, b);
    auto k_ab = ab * k;

    auto ka   = a * k;
    auto ka_b = origin::mat_mul(ka, b);

    auto kb   = b * k;
    auto a_kb = origin::mat_mul(a, kb);

    auto data1 = k_ab.to_vector<float>();
    auto data2 = ka_b.to_vector<float>();
    auto data3 = a_kb.to_vector<float>();

    for (size_t i = 0; i < data1.size(); ++i)
    {
        EXPECT_NEAR(data1[i], data2[i], kFloatTolerance);
        EXPECT_NEAR(data1[i], data3[i], kFloatTolerance);
    }
}
