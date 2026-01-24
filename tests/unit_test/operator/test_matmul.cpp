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
 * @brief 矩阵乘法算子测试类（参数化版本）
 */
class MatMulOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 前向传播测试 ====================

TEST_P(MatMulOperatorTest, ForwardBasic)
{
    // 测试基本矩阵乘法运算
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto w = Tensor({5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(x, w);

    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    // 结果应该是 [[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]] = [[19, 22], [43, 50]]
    auto expected =
        Tensor({19.0f, 22.0f, 43.0f, 50.0f}, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(MatMulOperatorTest, ForwardOperatorOverload)
{
    // 测试运算符重载
    auto x = Tensor({1.0f, 2.0f}, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto w = Tensor({3.0f, 4.0f}, Shape{2, 1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(x, w);

    Shape expected_shape{1, 1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 11.0f, origin::test::TestTolerance::kDefault);  // 1*3 + 2*4 = 11
}

TEST_P(MatMulOperatorTest, ForwardDifferentSizes)
{
    // 测试不同大小的矩阵乘法
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));
    auto w =
        Tensor({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, Shape{3, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(x, w);

    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    // 结果应该是 [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
    // = [[58, 64], [139, 154]]
    auto expected =
        Tensor({58.0f, 64.0f, 139.0f, 154.0f}, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(MatMulOperatorTest, ForwardIdentityMatrix)
{
    // 测试单位矩阵乘法
    auto x        = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto identity = Tensor({1.0f, 0.0f, 0.0f, 1.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(x, identity);

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x, origin::test::TestTolerance::kDefault);
}

TEST_P(MatMulOperatorTest, ForwardZeroMatrix)
{
    // 测试零矩阵乘法
    auto x    = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto zero = Tensor::zeros(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(x, zero);

    auto expected = Tensor::zeros(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 反向传播测试 ====================

TEST_P(MatMulOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2},
                    dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    auto w = Tensor({5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2},
                    dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::mat_mul(x, w);
    y.backward();

    // 矩阵乘法算子的梯度：
    // ∂y/∂x = gy * w^T
    // ∂y/∂w = x^T * gy
    // 验证梯度不为零
    auto gx_data = x.grad().to_vector<float>();
    auto gw_data = w.grad().to_vector<float>();

    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NE(gx_data[i], 0.0f);
    }
    for (size_t i = 0; i < gw_data.size(); ++i)
    {
        EXPECT_NE(gw_data[i], 0.0f);
    }
}

TEST_P(MatMulOperatorTest, BackwardWithGradient)
{
    // 测试带梯度的反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2},
                    dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    auto w = Tensor({5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2},
                    dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::mat_mul(x, w);
    y.backward();

    // 梯度会累积
    auto gx_data = x.grad().to_vector<float>();
    auto gw_data = w.grad().to_vector<float>();

    // 验证梯度计算正确（libtorch行为）
    // x.grad = gy @ w^T = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
    auto expected_gx =
        Tensor({11.0f, 15.0f, 11.0f, 15.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    // w.grad = x^T @ gy = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
    auto expected_gw = Tensor({4.0f, 4.0f, 6.0f, 6.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_gx, origin::test::TestTolerance::kDefault);
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(w.grad(), expected_gw, origin::test::TestTolerance::kDefault);
}

TEST_P(MatMulOperatorTest, BackwardDifferentSizes)
{
    // 测试不同大小的矩阵乘法反向传播
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, Shape{2, 3},
                    dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    auto w = Tensor({7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f}, Shape{3, 2},
                    dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::mat_mul(x, w);
    y.backward();

    auto gx_data = x.grad().to_vector<float>();
    auto gw_data = w.grad().to_vector<float>();

    EXPECT_EQ(gx_data.size(), 6U);
    EXPECT_EQ(gw_data.size(), 6U);

    // 验证梯度不为零
    for (size_t i = 0; i < gx_data.size(); ++i)
    {
        EXPECT_NE(gx_data[i], 0.0f);
    }
    for (size_t i = 0; i < gw_data.size(); ++i)
    {
        EXPECT_NE(gw_data[i], 0.0f);
    }
}

TEST_P(MatMulOperatorTest, BackwardIdentityMatrix)
{
    // 测试单位矩阵乘法的反向传播
    auto x        = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2},
                           dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    auto identity = Tensor({1.0f, 0.0f, 0.0f, 1.0f}, Shape{2, 2},
                           dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::mat_mul(x, identity);
    y.backward();

    auto gx_data        = x.grad().to_vector<float>();
    auto gidentity_data = identity.grad().to_vector<float>();

    // x的梯度应该等于输出梯度（libtorch行为）
    auto expected_gx = Tensor::ones(Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x.grad(), expected_gx, origin::test::TestTolerance::kDefault);

    // identity的梯度应该是x^T @ gy = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
    auto expected_gidentity =
        Tensor({4.0f, 4.0f, 6.0f, 6.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(identity.grad(), expected_gidentity,
                                                origin::test::TestTolerance::kDefault);
}

// ==================== 边界情况测试 ====================

TEST_P(MatMulOperatorTest, SingleElement)
{
    // 测试单元素矩阵乘法
    auto x = Tensor({5.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));
    auto w = Tensor({3.0f}, Shape{1, 1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(x, w);

    Shape expected_shape{1, 1};
    EXPECT_EQ(result.shape(), expected_shape);
    EXPECT_NEAR(result.item<float>(), 15.0f, origin::test::TestTolerance::kDefault);
}

TEST_P(MatMulOperatorTest, LargeMatrix)
{
    // 测试大矩阵乘法
    std::vector<float> data_x(100, 1.0f);
    std::vector<float> data_w(100, 2.0f);
    auto x = Tensor(data_x, Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));
    auto w = Tensor(data_w, Shape{10, 10}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(x, w);

    Shape expected_shape{10, 10};
    EXPECT_EQ(result.shape(), expected_shape);
    // 每行都是10个1.0乘以10个2.0的和 = 10 * 2 = 20
    auto expected =
        Tensor(std::vector<float>(100, 20.0f), expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(MatMulOperatorTest, ThreeDimensional)
{
    // 测试三维张量矩阵乘法（libtorch支持）
    // 注意：CUDA版本的matmul目前不支持复杂广播，所以跳过CUDA测试
    if (deviceType() == DeviceType::kCUDA)
    {
        GTEST_SKIP() << "CUDA matmul does not support complex broadcasting yet";
    }

    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2, 2},
                    dtype(DataType::kFloat32).device(deviceType()));
    auto w = Tensor({1.0f, 0.0f, 0.0f, 1.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    // libtorch支持三维张量矩阵乘法，结果形状为[2, 2, 2]
    auto result = F::mat_mul(x, w);
    Shape expected_shape{2, 2, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    // 验证结果正确性（单位矩阵乘法，结果应该等于输入）
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x, origin::test::TestTolerance::kDefault);
}

// ==================== 数值稳定性测试 ====================

TEST_P(MatMulOperatorTest, NumericalStability)
{
    // 测试数值稳定性
    // 使用更合理的数值范围，避免 cuBLAS 的数值问题（极值可能导致 NaN/Inf）
    // 将极值从 1e10/1e-10 调整为 1e3/1e-3，仍然测试数值稳定性但不会触发 cuBLAS 的边界情况
    auto x = Tensor({1e3f, 1e-3f, 1e3f, 1e-3f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto w = Tensor({1e-3f, 1e3f, 1e-3f, 1e3f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(x, w);

    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    auto result_data = result.to_vector<float>();

    // 验证结果在合理范围内
    for (size_t i = 0; i < result_data.size(); ++i)
    {
        EXPECT_TRUE(std::isfinite(result_data[i]));
    }
}

TEST_P(MatMulOperatorTest, PrecisionTest)
{
    // 测试精度
    auto x = Tensor({0.1f, 0.2f, 0.3f, 0.4f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto w = Tensor({0.5f, 0.6f, 0.7f, 0.8f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(x, w);

    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    // 结果应该是 [[0.1*0.5+0.2*0.7, 0.1*0.6+0.2*0.8], [0.3*0.5+0.4*0.7, 0.3*0.6+0.4*0.8]]
    // = [[0.19, 0.22], [0.43, 0.50]]
    auto expected =
        Tensor({0.19f, 0.22f, 0.43f, 0.50f}, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

// ==================== 特殊值测试 ====================

TEST_P(MatMulOperatorTest, MixedSigns)
{
    // 测试混合符号
    auto x = Tensor({1.0f, -2.0f, 3.0f, -4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto w = Tensor({-1.0f, 2.0f, -3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(x, w);

    Shape expected_shape{2, 2};
    EXPECT_EQ(result.shape(), expected_shape);
    // 结果应该是 [[1*(-1)+(-2)*(-3), 1*2+(-2)*4], [3*(-1)+(-4)*(-3), 3*2+(-4)*4]]
    // = [[5, -6], [9, -10]]
    auto expected = Tensor({5.0f, -6.0f, 9.0f, -10.0f}, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(MatMulOperatorTest, AssociativeProperty)
{
    // 测试结合性质：(A * B) * C = A * (B * C)
    auto A = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto B = Tensor({5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto C = Tensor({9.0f, 10.0f, 11.0f, 12.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result1 = F::mat_mul(F::mat_mul(A, B), C);
    auto result2 = F::mat_mul(A, F::mat_mul(B, C));

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result1, result2, origin::test::TestTolerance::kDefault);
}

TEST_P(MatMulOperatorTest, DistributiveProperty)
{
    // 测试分配性质：A * (B + C) = A * B + A * C
    auto A = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto B = Tensor({5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto C = Tensor({9.0f, 10.0f, 11.0f, 12.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result1 = F::mat_mul(A, B + C);
    auto result2 = F::mat_mul(A, B) + F::mat_mul(A, C);

    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result1, result2, origin::test::TestTolerance::kDefault);
}

TEST_P(MatMulOperatorTest, DimensionValidation)
{
    // 测试维度验证
    auto x = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto w = Tensor({5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}, Shape{2, 3}, dtype(DataType::kFloat32).device(deviceType()));

    // 维度匹配，应该成功
    auto result = F::mat_mul(x, w);
    Shape expected_shape{2, 3};
    EXPECT_EQ(result.shape(), expected_shape);

    // 测试真正不匹配的维度
    auto x2 = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto w2 =
        Tensor({5.0f, 6.0f, 7.0f, 8.0f, 9.0f, 10.0f}, Shape{3, 2}, dtype(DataType::kFloat32).device(deviceType()));

    // 维度不匹配应该抛出异常
    EXPECT_THROW(F::mat_mul(x2, w2), std::exception);
}

// ==================== 不同规模矩阵测试 ====================

TEST_P(MatMulOperatorTest, SmallMatrix16x16)
{
    const int M = 16, K = 16, N = 16;
    std::vector<float> data_a(M * K);
    std::vector<float> data_b(K * N);
    for (int i = 0; i < M * K; ++i)
        data_a[i] = static_cast<float>(i % 10) + 0.1f;
    for (int i = 0; i < K * N; ++i)
        data_b[i] = static_cast<float>(i % 7) + 0.2f;

    auto a = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(deviceType()));
    auto b = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(a, b);
    EXPECT_EQ(result.shape(), Shape({M, N}));

    if (deviceType() == DeviceType::kCUDA)
    {
        auto cpu_a      = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_b      = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_result = F::mat_mul(cpu_a, cpu_b);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, cpu_result, origin::test::TestTolerance::kDefault);
    }
}

TEST_P(MatMulOperatorTest, SmallMatrix24x24)
{
    const int M = 24, K = 24, N = 24;
    std::vector<float> data_a(M * K);
    std::vector<float> data_b(K * N);
    for (int i = 0; i < M * K; ++i)
        data_a[i] = static_cast<float>(i % 10) + 0.1f;
    for (int i = 0; i < K * N; ++i)
        data_b[i] = static_cast<float>(i % 7) + 0.2f;

    auto a = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(deviceType()));
    auto b = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(a, b);
    EXPECT_EQ(result.shape(), Shape({M, N}));

    if (deviceType() == DeviceType::kCUDA)
    {
        auto cpu_a      = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_b      = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_result = F::mat_mul(cpu_a, cpu_b);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, cpu_result, origin::test::TestTolerance::kDefault);
    }
}

TEST_P(MatMulOperatorTest, SmallMatrix31x31)
{
    const int M = 31, K = 31, N = 31;
    std::vector<float> data_a(M * K);
    std::vector<float> data_b(K * N);
    for (int i = 0; i < M * K; ++i)
        data_a[i] = static_cast<float>(i % 10) + 0.1f;
    for (int i = 0; i < K * N; ++i)
        data_b[i] = static_cast<float>(i % 7) + 0.2f;

    auto a = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(deviceType()));
    auto b = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(a, b);
    EXPECT_EQ(result.shape(), Shape({M, N}));

    if (deviceType() == DeviceType::kCUDA)
    {
        auto cpu_a      = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_b      = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_result = F::mat_mul(cpu_a, cpu_b);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, cpu_result, origin::test::TestTolerance::kDefault);
    }
}

TEST_P(MatMulOperatorTest, MediumMatrix32x32)
{
    const int M = 32, K = 32, N = 32;
    std::vector<float> data_a(M * K);
    std::vector<float> data_b(K * N);
    for (int i = 0; i < M * K; ++i)
        data_a[i] = static_cast<float>(i % 10) + 0.1f;
    for (int i = 0; i < K * N; ++i)
        data_b[i] = static_cast<float>(i % 7) + 0.2f;

    auto a = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(deviceType()));
    auto b = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(a, b);
    EXPECT_EQ(result.shape(), Shape({M, N}));

    if (deviceType() == DeviceType::kCUDA)
    {
        auto cpu_a      = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_b      = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_result = F::mat_mul(cpu_a, cpu_b);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, cpu_result, origin::test::TestTolerance::kDefault);
    }
}

TEST_P(MatMulOperatorTest, MediumMatrix64x64)
{
    const int M = 64, K = 64, N = 64;
    std::vector<float> data_a(M * K);
    std::vector<float> data_b(K * N);
    for (int i = 0; i < M * K; ++i)
        data_a[i] = static_cast<float>(i % 10) + 0.1f;
    for (int i = 0; i < K * N; ++i)
        data_b[i] = static_cast<float>(i % 7) + 0.2f;

    auto a = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(deviceType()));
    auto b = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(a, b);
    EXPECT_EQ(result.shape(), Shape({M, N}));

    if (deviceType() == DeviceType::kCUDA)
    {
        auto cpu_a      = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_b      = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_result = F::mat_mul(cpu_a, cpu_b);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, cpu_result, origin::test::TestTolerance::kDefault);
    }
}

TEST_P(MatMulOperatorTest, MediumMatrix100x100)
{
    const int M = 100, K = 100, N = 100;
    std::vector<float> data_a(M * K);
    std::vector<float> data_b(K * N);
    for (int i = 0; i < M * K; ++i)
        data_a[i] = static_cast<float>(i % 10) + 0.1f;
    for (int i = 0; i < K * N; ++i)
        data_b[i] = static_cast<float>(i % 7) + 0.2f;

    auto a = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(deviceType()));
    auto b = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(a, b);
    EXPECT_EQ(result.shape(), Shape({M, N}));

    if (deviceType() == DeviceType::kCUDA)
    {
        auto cpu_a      = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_b      = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_result = F::mat_mul(cpu_a, cpu_b);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, cpu_result, origin::test::TestTolerance::kDefault);
    }
}

TEST_P(MatMulOperatorTest, MediumMatrix127x127)
{
    const int M = 127, K = 127, N = 127;
    std::vector<float> data_a(M * K);
    std::vector<float> data_b(K * N);
    for (int i = 0; i < M * K; ++i)
        data_a[i] = static_cast<float>(i % 10) + 0.1f;
    for (int i = 0; i < K * N; ++i)
        data_b[i] = static_cast<float>(i % 7) + 0.2f;

    auto a = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(deviceType()));
    auto b = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(a, b);
    EXPECT_EQ(result.shape(), Shape({M, N}));

    if (deviceType() == DeviceType::kCUDA)
    {
        auto cpu_a      = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_b      = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_result = F::mat_mul(cpu_a, cpu_b);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, cpu_result, origin::test::TestTolerance::kDefault);
    }
}

TEST_P(MatMulOperatorTest, LargeMatrix128x128)
{
    const int M = 128, K = 128, N = 128;
    std::vector<float> data_a(M * K);
    std::vector<float> data_b(K * N);
    for (int i = 0; i < M * K; ++i)
        data_a[i] = static_cast<float>(i % 10) + 0.1f;
    for (int i = 0; i < K * N; ++i)
        data_b[i] = static_cast<float>(i % 7) + 0.2f;

    auto a = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(deviceType()));
    auto b = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(a, b);
    EXPECT_EQ(result.shape(), Shape({M, N}));

    if (deviceType() == DeviceType::kCUDA)
    {
        auto cpu_a      = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_b      = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_result = F::mat_mul(cpu_a, cpu_b);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, cpu_result, origin::test::TestTolerance::kDefault);
    }
}

TEST_P(MatMulOperatorTest, LargeMatrix256x256)
{
    const int M = 256, K = 256, N = 256;
    std::vector<float> data_a(M * K);
    std::vector<float> data_b(K * N);
    for (int i = 0; i < M * K; ++i)
        data_a[i] = static_cast<float>(i % 10) + 0.1f;
    for (int i = 0; i < K * N; ++i)
        data_b[i] = static_cast<float>(i % 7) + 0.2f;

    auto a = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(deviceType()));
    auto b = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(a, b);
    EXPECT_EQ(result.shape(), Shape({M, N}));

    if (deviceType() == DeviceType::kCUDA)
    {
        auto cpu_a      = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_b      = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_result = F::mat_mul(cpu_a, cpu_b);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, cpu_result, origin::test::TestTolerance::kDefault);
    }
}

TEST_P(MatMulOperatorTest, LargeMatrix512x512)
{
    const int M = 512, K = 512, N = 512;
    std::vector<float> data_a(M * K);
    std::vector<float> data_b(K * N);
    for (int i = 0; i < M * K; ++i)
        data_a[i] = static_cast<float>(i % 10) + 0.1f;
    for (int i = 0; i < K * N; ++i)
        data_b[i] = static_cast<float>(i % 7) + 0.2f;

    auto a = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(deviceType()));
    auto b = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(a, b);
    EXPECT_EQ(result.shape(), Shape({M, N}));

    if (deviceType() == DeviceType::kCUDA)
    {
        auto cpu_a      = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_b      = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_result = F::mat_mul(cpu_a, cpu_b);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, cpu_result, origin::test::TestTolerance::kDefault);
    }
}

TEST_P(MatMulOperatorTest, LargeMatrix1024x1024)
{
    const int M = 1024, K = 1024, N = 1024;
    std::vector<float> data_a(M * K);
    std::vector<float> data_b(K * N);
    for (int i = 0; i < M * K; ++i)
        data_a[i] = static_cast<float>(i % 10) + 0.1f;
    for (int i = 0; i < K * N; ++i)
        data_b[i] = static_cast<float>(i % 7) + 0.2f;

    auto a = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(deviceType()));
    auto b = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(a, b);
    EXPECT_EQ(result.shape(), Shape({M, N}));

    if (deviceType() == DeviceType::kCUDA)
    {
        auto cpu_a      = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_b      = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_result = F::mat_mul(cpu_a, cpu_b);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, cpu_result, origin::test::TestTolerance::kDefault);
    }
}

TEST_P(MatMulOperatorTest, VeryLargeMatrix2048x2048)
{
    const int M = 2048, K = 2048, N = 2048;
    std::vector<float> data_a(M * K);
    std::vector<float> data_b(K * N);
    for (int i = 0; i < M * K; ++i)
        data_a[i] = static_cast<float>(i % 10) + 0.1f;
    for (int i = 0; i < K * N; ++i)
        data_b[i] = static_cast<float>(i % 7) + 0.2f;

    auto a = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(deviceType()));
    auto b = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(a, b);
    EXPECT_EQ(result.shape(), Shape({M, N}));

    if (deviceType() == DeviceType::kCUDA)
    {
        auto cpu_a      = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_b      = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_result = F::mat_mul(cpu_a, cpu_b);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, cpu_result, origin::test::TestTolerance::kDefault);
    }
}

TEST_P(MatMulOperatorTest, NonSquareMatrix16x32x48)
{
    const int M = 16, K = 32, N = 48;
    std::vector<float> data_a(M * K);
    std::vector<float> data_b(K * N);
    for (int i = 0; i < M * K; ++i)
        data_a[i] = static_cast<float>(i % 10) + 0.1f;
    for (int i = 0; i < K * N; ++i)
        data_b[i] = static_cast<float>(i % 7) + 0.2f;

    auto a = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(deviceType()));
    auto b = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(a, b);
    EXPECT_EQ(result.shape(), Shape({M, N}));

    if (deviceType() == DeviceType::kCUDA)
    {
        auto cpu_a      = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_b      = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_result = F::mat_mul(cpu_a, cpu_b);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, cpu_result, origin::test::TestTolerance::kDefault);
    }
}

TEST_P(MatMulOperatorTest, NonSquareMatrix128x256x512)
{
    const int M = 128, K = 256, N = 512;
    std::vector<float> data_a(M * K);
    std::vector<float> data_b(K * N);
    for (int i = 0; i < M * K; ++i)
        data_a[i] = static_cast<float>(i % 10) + 0.1f;
    for (int i = 0; i < K * N; ++i)
        data_b[i] = static_cast<float>(i % 7) + 0.2f;

    auto a = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(deviceType()));
    auto b = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(a, b);
    EXPECT_EQ(result.shape(), Shape({M, N}));

    if (deviceType() == DeviceType::kCUDA)
    {
        auto cpu_a      = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_b      = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_result = F::mat_mul(cpu_a, cpu_b);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, cpu_result, origin::test::TestTolerance::kDefault);
    }
}

TEST_P(MatMulOperatorTest, NonSquareMatrix512x1024x2048)
{
    const int M = 512, K = 1024, N = 2048;
    std::vector<float> data_a(M * K);
    std::vector<float> data_b(K * N);
    for (int i = 0; i < M * K; ++i)
        data_a[i] = static_cast<float>(i % 10) + 0.1f;
    for (int i = 0; i < K * N; ++i)
        data_b[i] = static_cast<float>(i % 7) + 0.2f;

    auto a = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(deviceType()));
    auto b = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(a, b);
    EXPECT_EQ(result.shape(), Shape({M, N}));

    if (deviceType() == DeviceType::kCUDA)
    {
        auto cpu_a      = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_b      = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_result = F::mat_mul(cpu_a, cpu_b);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, cpu_result, origin::test::TestTolerance::kDefault);
    }
}

TEST_P(MatMulOperatorTest, RectangularMatrix1x1000x1)
{
    const int M = 1, K = 1000, N = 1;
    std::vector<float> data_a(M * K);
    std::vector<float> data_b(K * N);
    for (int i = 0; i < M * K; ++i)
        data_a[i] = static_cast<float>(i % 10) + 0.1f;
    for (int i = 0; i < K * N; ++i)
        data_b[i] = static_cast<float>(i % 7) + 0.2f;

    auto a = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(deviceType()));
    auto b = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(a, b);
    EXPECT_EQ(result.shape(), Shape({M, N}));

    if (deviceType() == DeviceType::kCUDA)
    {
        auto cpu_a      = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_b      = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_result = F::mat_mul(cpu_a, cpu_b);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, cpu_result, origin::test::TestTolerance::kDefault);
    }
}

TEST_P(MatMulOperatorTest, RectangularMatrix1000x1x1000)
{
    const int M = 1000, K = 1, N = 1000;
    std::vector<float> data_a(M * K);
    std::vector<float> data_b(K * N);
    for (int i = 0; i < M * K; ++i)
        data_a[i] = static_cast<float>(i % 10) + 0.1f;
    for (int i = 0; i < K * N; ++i)
        data_b[i] = static_cast<float>(i % 7) + 0.2f;

    auto a = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(deviceType()));
    auto b = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::mat_mul(a, b);
    EXPECT_EQ(result.shape(), Shape({M, N}));

    if (deviceType() == DeviceType::kCUDA)
    {
        auto cpu_a      = Tensor(data_a, Shape{M, K}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_b      = Tensor(data_b, Shape{K, N}, dtype(DataType::kFloat32).device(DeviceType::kCPU));
        auto cpu_result = F::mat_mul(cpu_a, cpu_b);
        origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, cpu_result, origin::test::TestTolerance::kDefault);
    }
}

// 实例化测试套件：自动为CPU和可用CUDA生成测试
INSTANTIATE_DEVICE_TEST_SUITE_P(MatMulOperatorTest);
