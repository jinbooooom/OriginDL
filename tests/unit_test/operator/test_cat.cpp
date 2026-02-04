#include <gtest/gtest.h>
#include <vector>
#include "../common/device_test_base.h"
#include "../common/gtest_utils.h"
#include "../common/test_utils.h"
#include "origin.h"
#include "origin/operators/shape/cat.h"

using namespace origin;
namespace F = origin::functional;

/**
 * @brief Cat 算子测试类（参数化版本）
 */
class CatOperatorTest : public origin::test::OperatorTestBase
{};

// ==================== 前向传播测试 ====================

TEST_P(CatOperatorTest, ForwardBasic)
{
    // 测试基本拼接：在 dim=0 上拼接
    auto x0 = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({3.0f, 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::cat({x0, x1}, 0);

    Shape expected_shape{4};
    EXPECT_EQ(result.shape(), expected_shape);

    std::vector<float> expected_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(CatOperatorTest, ForwardDim1)
{
    // 测试在 dim=1 上拼接
    auto x0 = Tensor({1.0f, 2.0f}, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({3.0f, 4.0f}, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::cat({x0, x1}, 1);

    Shape expected_shape{1, 4};
    EXPECT_EQ(result.shape(), expected_shape);

    std::vector<float> expected_data = {1.0f, 2.0f, 3.0f, 4.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(CatOperatorTest, ForwardMultipleTensors)
{
    // 测试拼接多个张量
    auto x0 = Tensor({1.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({2.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    auto x2 = Tensor({3.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::cat({x0, x1, x2}, 0);

    Shape expected_shape{3};
    EXPECT_EQ(result.shape(), expected_shape);

    std::vector<float> expected_data = {1.0f, 2.0f, 3.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(CatOperatorTest, ForwardTwoDimensional)
{
    // 测试 2D 张量拼接
    auto x0 = Tensor({1.0f, 2.0f, 3.0f, 4.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({5.0f, 6.0f, 7.0f, 8.0f}, Shape{2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    // 在 dim=0 上拼接
    auto result0 = F::cat({x0, x1}, 0);
    Shape expected_shape0{4, 2};
    EXPECT_EQ(result0.shape(), expected_shape0);

    // 在 dim=1 上拼接
    auto result1 = F::cat({x0, x1}, 1);
    Shape expected_shape1{2, 4};
    EXPECT_EQ(result1.shape(), expected_shape1);
}

TEST_P(CatOperatorTest, ForwardSingleTensor)
{
    // 测试单个张量（应该直接返回）
    auto x = Tensor({1.0f, 2.0f, 3.0f}, Shape{3}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::cat({x}, 0);

    // 应该和输入相同
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, x, origin::test::TestTolerance::kDefault);
}

// ==================== 反向传播测试 ====================

TEST_P(CatOperatorTest, BackwardBasic)
{
    // 测试基本反向传播
    auto x0 = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    auto x1 = Tensor({3.0f, 4.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::cat({x0, x1}, 0);
    y.backward();

    // 验证梯度形状
    EXPECT_EQ(x0.grad().shape(), x0.shape());
    EXPECT_EQ(x1.grad().shape(), x1.shape());

    // 梯度应该被分割回各个输入
    // 由于输出梯度是全1（默认），每个输入的梯度应该是全1
    auto expected_grad = Tensor::ones(Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x0.grad(), expected_grad, origin::test::TestTolerance::kDefault);
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(x1.grad(), expected_grad, origin::test::TestTolerance::kDefault);
}

TEST_P(CatOperatorTest, BackwardDim1)
{
    // 测试在 dim=1 上的反向传播
    auto x0 = Tensor({1.0f, 2.0f}, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));
    auto x1 = Tensor({3.0f, 4.0f}, Shape{1, 2}, dtype(DataType::kFloat32).device(deviceType()).requires_grad(true));

    auto y = F::cat({x0, x1}, 1);
    y.backward();

    // 验证梯度形状
    EXPECT_EQ(x0.grad().shape(), x0.shape());
    EXPECT_EQ(x1.grad().shape(), x1.shape());
}

// ==================== 边界情况测试 ====================

TEST_P(CatOperatorTest, SingleElement)
{
    // 测试单元素张量拼接
    auto x0 = Tensor({5.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({6.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::cat({x0, x1}, 0);

    Shape expected_shape{2};
    EXPECT_EQ(result.shape(), expected_shape);

    std::vector<float> expected_data = {5.0f, 6.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(CatOperatorTest, DifferentSizes)
{
    // 测试不同大小的张量拼接
    auto x0 = Tensor({1.0f, 2.0f}, Shape{2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor({3.0f}, Shape{1}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::cat({x0, x1}, 0);

    Shape expected_shape{3};
    EXPECT_EQ(result.shape(), expected_shape);

    std::vector<float> expected_data = {1.0f, 2.0f, 3.0f};
    auto expected = Tensor(expected_data, expected_shape, dtype(DataType::kFloat32).device(deviceType()));
    origin::test::GTestUtils::EXPECT_TENSORS_EQ(result, expected, origin::test::TestTolerance::kDefault);
}

TEST_P(CatOperatorTest, HighDimensional)
{
    // 测试高维张量拼接（6维，每个维度都大于1）
    // 形状: [2, 3, 4, 2, 3, 2] 在 dim=2 上拼接
    // 输入1: [2, 3, 2, 2, 3, 2] (C=2)
    // 输入2: [2, 3, 2, 2, 3, 2] (C=2)
    // 输出: [2, 3, 4, 2, 3, 2] (C=4)
    
    // 创建输入数据：每个输入有 2*3*2*2*3*2 = 144 个元素
    std::vector<float> data1(144);
    std::vector<float> data2(144);
    for (size_t i = 0; i < 144; ++i)
    {
        data1[i] = static_cast<float>(i);
        data2[i] = static_cast<float>(i + 144);
    }
    
    auto x0 = Tensor(data1, Shape{2, 3, 2, 2, 3, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor(data2, Shape{2, 3, 2, 2, 3, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::cat({x0, x1}, 2);

    // 验证输出形状
    Shape expected_shape{2, 3, 4, 2, 3, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    // 验证输出数据：应该先包含 x0 的所有数据，然后是 x1 的所有数据
    // 由于在 dim=2 上拼接，每个 chunk (对应 M=2*3=6 个 chunk) 应该先包含 x0 的 C=2 个通道，然后是 x1 的 C=2 个通道
    auto result_data = result.to_vector<float>();
    EXPECT_EQ(result_data.size(), 288);  // 2*3*4*2*3*2 = 288
    
    // 完整验证：验证所有元素
    // M=6, C=2, N=12 (2*3*2)
    // 每个 chunk 有 output_C*N = 4*12 = 48 个元素
    // 每个输入 chunk 有 C*N = 2*12 = 24 个元素
    const size_t M = 6;
    const size_t C = 2;
    const size_t N = 12;
    const size_t output_C = 4;
    
    for (size_t m_idx = 0; m_idx < M; ++m_idx)
    {
        size_t chunk_start = m_idx * output_C * N;  // 每个 chunk 有 48 个元素
        size_t x0_base = m_idx * C * N;             // x0 在这个 chunk 中的起始索引
        size_t x1_base = m_idx * C * N + 144;       // x1 在这个 chunk 中的起始索引
        
        // 验证 x0 的 24 个元素
        for (size_t i = 0; i < C * N; ++i)
        {
            EXPECT_FLOAT_EQ(result_data[chunk_start + i], static_cast<float>(x0_base + i))
                << "m_idx=" << m_idx << ", chunk_start=" << chunk_start << ", i=" << i;
        }
        
        // 验证 x1 的 24 个元素
        for (size_t i = 0; i < C * N; ++i)
        {
            EXPECT_FLOAT_EQ(result_data[chunk_start + C * N + i], static_cast<float>(x1_base + i))
                << "m_idx=" << m_idx << ", chunk_start=" << chunk_start << ", i=" << i;
        }
    }
}

TEST_P(CatOperatorTest, HighDimensionalDim0)
{
    // 测试高维张量在最左边维度（dim=0）上拼接（边界测试）
    // 形状: [4, 2, 3, 2, 3, 2] 在 dim=0 上拼接
    // 输入1: [2, 2, 3, 2, 3, 2] (A=2)
    // 输入2: [2, 2, 3, 2, 3, 2] (A=2)
    // 输出: [4, 2, 3, 2, 3, 2] (A=4)
    // 转换为3维: [M=1, C=2, N=2*3*2*3*2=72] -> [M=1, C=4, N=72]
    
    // 创建输入数据：每个输入有 2*2*3*2*3*2 = 144 个元素
    std::vector<float> data1(144);
    std::vector<float> data2(144);
    for (size_t i = 0; i < 144; ++i)
    {
        data1[i] = static_cast<float>(i);
        data2[i] = static_cast<float>(i + 144);
    }
    
    auto x0 = Tensor(data1, Shape{2, 2, 3, 2, 3, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor(data2, Shape{2, 2, 3, 2, 3, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::cat({x0, x1}, 0);

    // 验证输出形状
    Shape expected_shape{4, 2, 3, 2, 3, 2};
    EXPECT_EQ(result.shape(), expected_shape);

    // 验证输出数据
    // 在 dim=0 上拼接，M=1，所以只有一个 chunk
    // 输出应该先包含 x0 的所有 144 个元素，然后是 x1 的所有 144 个元素
    auto result_data = result.to_vector<float>();
    EXPECT_EQ(result_data.size(), 288);  // 4*2*3*2*3*2 = 288
    
    // 验证前 144 个元素来自 x0
    for (size_t i = 0; i < 144; ++i)
    {
        EXPECT_FLOAT_EQ(result_data[i], static_cast<float>(i));
    }
    
    // 验证接下来的 144 个元素来自 x1
    for (size_t i = 0; i < 144; ++i)
    {
        EXPECT_FLOAT_EQ(result_data[144 + i], static_cast<float>(i + 144));
    }
}

TEST_P(CatOperatorTest, HighDimensionalDim5)
{
    // 测试高维张量在最右边维度（dim=5）上拼接（边界测试）
    // 形状: [2, 3, 2, 3, 2, 4] 在 dim=5 上拼接
    // 输入1: [2, 3, 2, 3, 2, 2] (F=2)
    // 输入2: [2, 3, 2, 3, 2, 2] (F=2)
    // 输出: [2, 3, 2, 3, 2, 4] (F=4)
    // 转换为3维: [M=2*3*2*3*2=72, C=2, N=1] -> [M=72, C=4, N=1]
    
    // 创建输入数据：每个输入有 2*3*2*3*2*2 = 144 个元素
    std::vector<float> data1(144);
    std::vector<float> data2(144);
    for (size_t i = 0; i < 144; ++i)
    {
        data1[i] = static_cast<float>(i);
        data2[i] = static_cast<float>(i + 144);
    }
    
    auto x0 = Tensor(data1, Shape{2, 3, 2, 3, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));
    auto x1 = Tensor(data2, Shape{2, 3, 2, 3, 2, 2}, dtype(DataType::kFloat32).device(deviceType()));

    auto result = F::cat({x0, x1}, 5);

    // 验证输出形状
    Shape expected_shape{2, 3, 2, 3, 2, 4};
    EXPECT_EQ(result.shape(), expected_shape);

    // 验证输出数据
    // 在 dim=5 上拼接，N=1，所以每个 chunk 只有 C 个元素
    // M=72，所以有 72 个 chunk
    // 每个 chunk 先包含 x0 的 2 个元素，然后是 x1 的 2 个元素
    auto result_data = result.to_vector<float>();
    EXPECT_EQ(result_data.size(), 288);  // 2*3*2*3*2*4 = 288
    
    // 完整验证：验证所有元素
    // 对于每个 chunk (m_idx = 0 到 71)，每个 chunk 有 4 个元素
    // chunk 内布局: [x0[m_idx*2], x0[m_idx*2+1], x1[m_idx*2+144], x1[m_idx*2+145]]
    for (size_t m_idx = 0; m_idx < 72; ++m_idx)
    {
        size_t chunk_start = m_idx * 4;  // 每个 chunk 有 4 个元素
        size_t x0_base = m_idx * 2;       // x0 在这个 chunk 中的起始索引
        size_t x1_base = m_idx * 2 + 144; // x1 在这个 chunk 中的起始索引
        
        // 验证 x0 的两个元素
        EXPECT_FLOAT_EQ(result_data[chunk_start], static_cast<float>(x0_base))
            << "m_idx=" << m_idx << ", chunk_start=" << chunk_start;
        EXPECT_FLOAT_EQ(result_data[chunk_start + 1], static_cast<float>(x0_base + 1))
            << "m_idx=" << m_idx << ", chunk_start=" << chunk_start;
        
        // 验证 x1 的两个元素
        EXPECT_FLOAT_EQ(result_data[chunk_start + 2], static_cast<float>(x1_base))
            << "m_idx=" << m_idx << ", chunk_start=" << chunk_start;
        EXPECT_FLOAT_EQ(result_data[chunk_start + 3], static_cast<float>(x1_base + 1))
            << "m_idx=" << m_idx << ", chunk_start=" << chunk_start;
    }
}

// Instantiate test suite: automatically generate tests for CPU and available CUDA
INSTANTIATE_DEVICE_TEST_SUITE_P(CatOperatorTest);
