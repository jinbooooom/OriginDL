#include <gtest/gtest.h>
#include "origin/core/tensor.h"
#include "origin/nn/layers/embedding.h"
#include "test_utils.h"

using namespace origin;
namespace nn = origin::nn;

/**
 * @brief Embedding 层 CPU 版本测试
 * @note 仅测试前向传播，不涉及反向传播
 */

class EmbeddingTest : public ::testing::Test
{
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// ==================== 基础功能测试 ====================

TEST_F(EmbeddingTest, BasicForwardCPU_SingleToken)
{
    // 测试单个 token 的查表功能
    int vocab_size    = 100;
    int embedding_dim = 16;

    nn::Embedding embed(vocab_size, embedding_dim);
    embed.to(Device(DeviceType::kCPU));

    // 创建单个 token 的输入
    Tensor input = Tensor({42}, Shape{1}, DataType::kInt32);

    // 前向传播
    Tensor output = embed.forward(input);

    // 验证输出形状: (1, embedding_dim)
    EXPECT_EQ(output.shape().size(), 2U);
    EXPECT_EQ(output.shape()[0], 1U);
    EXPECT_EQ(output.shape()[1], static_cast<size_t>(embedding_dim));

    // 验证输出值：手动查表验证
    auto weight_data = embed.weight()->data_ptr<float>();
    auto output_data = output.data_ptr<float>();

    for (int j = 0; j < embedding_dim; ++j)
    {
        float expected = weight_data[42 * embedding_dim + j];
        EXPECT_NEAR(output_data[j], expected, 1e-5) << "Mismatch at dim=" << j;
    }
}

TEST_F(EmbeddingTest, BasicForwardCPU_MultipleTokens)
{
    // 测试多个 token 的查表功能
    int vocab_size    = 100;
    int embedding_dim = 16;

    nn::Embedding embed(vocab_size, embedding_dim);
    embed.to(Device(DeviceType::kCPU));

    // 创建输入: 2 个 batch，每个 batch 有 3 个 token
    // [[1, 5, 10],
    //  [2, 8, 15]]
    Tensor input = Tensor({1, 5, 10, 2, 8, 15}, Shape{2, 3}, DataType::kInt32);

    // 前向传播
    Tensor output = embed.forward(input);

    // 验证输出形状: (2, 3, embedding_dim)
    EXPECT_EQ(output.shape().size(), 3U);
    EXPECT_EQ(output.shape()[0], 2U);
    EXPECT_EQ(output.shape()[1], 3U);
    EXPECT_EQ(output.shape()[2], static_cast<size_t>(embedding_dim));

    // 验证输出值
    auto weight_data = embed.weight()->data_ptr<float>();
    auto output_data = output.data_ptr<float>();

    // 期望的 token IDs
    int expected_tokens[2][3] = {{1, 5, 10}, {2, 8, 15}};

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            int token_id = expected_tokens[i][j];
            for (int k = 0; k < embedding_dim; ++k)
            {
                size_t output_idx = i * 3 * embedding_dim + j * embedding_dim + k;
                float expected    = weight_data[token_id * embedding_dim + k];
                EXPECT_NEAR(output_data[output_idx], expected, 1e-5)
                    << "Mismatch at position (" << i << ", " << j << ", " << k << ")";
            }
        }
    }
}

TEST_F(EmbeddingTest, BasicForwardCPU_1DInput)
{
    // 测试 1D 输入 (单个 batch，序列长度为 N)
    int vocab_size    = 100;
    int embedding_dim = 16;

    nn::Embedding embed(vocab_size, embedding_dim);
    embed.to(Device(DeviceType::kCPU));

    // 创建 1D 输入: [5, 10, 15, 20, 25]
    Tensor input = Tensor({5, 10, 15, 20, 25}, Shape{5}, DataType::kInt32);

    // 前向传播
    Tensor output = embed.forward(input);

    // 验证输出形状: (5, embedding_dim)
    EXPECT_EQ(output.shape().size(), 2U);
    EXPECT_EQ(output.shape()[0], 5U);
    EXPECT_EQ(output.shape()[1], static_cast<size_t>(embedding_dim));

    // 验证输出值
    auto weight_data = embed.weight()->data_ptr<float>();
    auto output_data = output.data_ptr<float>();

    int expected_tokens[5] = {5, 10, 15, 20, 25};

    for (int i = 0; i < 5; ++i)
    {
        int token_id = expected_tokens[i];
        for (int j = 0; j < embedding_dim; ++j)
        {
            float expected = weight_data[token_id * embedding_dim + j];
            EXPECT_NEAR(output_data[i * embedding_dim + j], expected, 1e-5)
                << "Mismatch at position (" << i << ", " << j << ")";
        }
    }
}

TEST_F(EmbeddingTest, BasicForwardCPU_3DInput)
{
    // 测试 3D 输入 (batch_size, seq_len, features)
    // Embedding 应该保持原始形状并在最后添加 embedding_dim
    int vocab_size    = 100;
    int embedding_dim = 16;

    nn::Embedding embed(vocab_size, embedding_dim);
    embed.to(Device(DeviceType::kCPU));

    // 创建 3D 输入: (2, 2, 3) -> 输出应该是 (2, 2, 3, embedding_dim)
    Tensor input = Tensor({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, Shape{2, 2, 3}, DataType::kInt32);

    // 前向传播
    Tensor output = embed.forward(input);

    // 验证输出形状: (2, 2, 3, embedding_dim)
    EXPECT_EQ(output.shape().size(), 4U);
    EXPECT_EQ(output.shape()[0], 2U);
    EXPECT_EQ(output.shape()[1], 2U);
    EXPECT_EQ(output.shape()[2], 3U);
    EXPECT_EQ(output.shape()[3], static_cast<size_t>(embedding_dim));
}

// ==================== 边界情况测试 ====================

TEST_F(EmbeddingTest, EdgeCaseCPU_ZeroToken)
{
    // 测试 token ID 为 0 的情况
    int vocab_size    = 100;
    int embedding_dim = 16;

    nn::Embedding embed(vocab_size, embedding_dim);
    embed.to(Device(DeviceType::kCPU));

    Tensor input = Tensor({0}, Shape{1}, DataType::kInt32);

    Tensor output = embed.forward(input);

    // 验证输出
    auto weight_data = embed.weight()->data_ptr<float>();
    auto output_data = output.data_ptr<float>();

    for (int j = 0; j < embedding_dim; ++j)
    {
        EXPECT_NEAR(output_data[j], weight_data[j], 1e-5);
    }
}

TEST_F(EmbeddingTest, EdgeCaseCPU_LastToken)
{
    // 测试 token ID 为 vocab_size - 1 的情况
    int vocab_size    = 100;
    int embedding_dim = 16;

    nn::Embedding embed(vocab_size, embedding_dim);
    embed.to(Device(DeviceType::kCPU));

    Tensor input = Tensor({99}, Shape{1}, DataType::kInt32);

    Tensor output = embed.forward(input);

    // 验证输出
    auto weight_data = embed.weight()->data_ptr<float>();
    auto output_data = output.data_ptr<float>();

    for (int j = 0; j < embedding_dim; ++j)
    {
        EXPECT_NEAR(output_data[j], weight_data[99 * embedding_dim + j], 1e-5);
    }
}

TEST_F(EmbeddingTest, EdgeCaseCPU_EmptyInput)
{
    // 测试空输入（单个元素为 0 的张量，不是真正的空）
    int vocab_size    = 100;
    int embedding_dim = 16;

    nn::Embedding embed(vocab_size, embedding_dim);
    embed.to(Device(DeviceType::kCPU));

    // 创建一个 1x1 的输入
    Tensor input = Tensor({5}, Shape{1, 1}, DataType::kInt32);

    Tensor output = embed.forward(input);

    // 验证输出形状: (1, 1, embedding_dim)
    EXPECT_EQ(output.shape().size(), 3U);
    EXPECT_EQ(output.shape()[0], 1U);
    EXPECT_EQ(output.shape()[1], 1U);
    EXPECT_EQ(output.shape()[2], static_cast<size_t>(embedding_dim));
}

TEST_F(EmbeddingTest, EdgeCaseCPU_LargeEmbeddingDim)
{
    // 测试较大的 embedding 维度
    int vocab_size    = 100;
    int embedding_dim = 512;

    nn::Embedding embed(vocab_size, embedding_dim);
    embed.to(Device(DeviceType::kCPU));

    Tensor input = Tensor({10}, Shape{1}, DataType::kInt32);

    Tensor output = embed.forward(input);

    // 验证输出形状
    EXPECT_EQ(output.shape()[1], static_cast<size_t>(embedding_dim));

    // 验证输出值
    auto weight_data = embed.weight()->data_ptr<float>();
    auto output_data = output.data_ptr<float>();

    for (int j = 0; j < embedding_dim; ++j)
    {
        EXPECT_NEAR(output_data[j], weight_data[10 * embedding_dim + j], 1e-5);
    }
}

// ==================== 参数测试 ====================

TEST_F(EmbeddingTest, ParameterAccessCPU)
{
    // 测试参数访问
    int vocab_size    = 100;
    int embedding_dim = 16;

    nn::Embedding embed(vocab_size, embedding_dim);

    // 测试 vocab_size() 和 embedding_dim()
    EXPECT_EQ(embed.vocab_size(), vocab_size);
    EXPECT_EQ(embed.embedding_dim(), embedding_dim);

    // 测试 weight() 访问
    Parameter *weight = embed.weight();
    ASSERT_NE(weight, nullptr);

    // 验证权重形状
    Tensor weight_tensor = static_cast<Tensor>(*weight);
    EXPECT_EQ(weight_tensor.shape().size(), 2U);
    EXPECT_EQ(weight_tensor.shape()[0], static_cast<size_t>(vocab_size));
    EXPECT_EQ(weight_tensor.shape()[1], static_cast<size_t>(embedding_dim));
}

TEST_F(EmbeddingTest, ParameterResetCPU)
{
    // 测试参数重置
    int vocab_size    = 100;
    int embedding_dim = 16;

    nn::Embedding embed(vocab_size, embedding_dim);
    embed.to(Device(DeviceType::kCPU));

    // 保存原始权重
    Tensor old_weight = static_cast<Tensor>(*embed.weight());

    // 重置参数
    embed.reset_parameters();

    // 获取新权重
    Tensor new_weight = static_cast<Tensor>(*embed.weight());

    // 验证新权重与旧权重不同（因为使用了随机初始化）
    auto old_data = old_weight.data_ptr<float>();
    auto new_data = new_weight.data_ptr<float>();

    bool is_different = false;
    for (size_t i = 0; i < old_weight.elements(); ++i)
    {
        if (std::abs(old_data[i] - new_data[i]) > 1e-5)
        {
            is_different = true;
            break;
        }
    }

    EXPECT_TRUE(is_different) << "Weights should be different after reset";
}

TEST_F(EmbeddingTest, DataTypeValidationCPU)
{
    // 测试输入数据类型验证
    int vocab_size    = 100;
    int embedding_dim = 16;

    nn::Embedding embed(vocab_size, embedding_dim);
    embed.to(Device(DeviceType::kCPU));

    // 创建错误类型的输入（float 而非 int32）
    std::vector<float> float_data = {1.0f, 2.0f, 3.0f};
    Tensor input                  = Tensor(float_data, Shape{3}, DataType::kFloat32);

    // 期望抛出异常
    EXPECT_THROW(
        {
            try
            {
                Tensor output = embed.forward(input);
            }
            catch (const std::exception &e)
            {
                // 验证异常信息包含 "int32"
                std::string msg(e.what());
                EXPECT_TRUE(msg.find("int32") != std::string::npos);
                throw;
            }
        },
        std::exception);
}

// ==================== 重复 Token 测试 ====================

TEST_F(EmbeddingTest, RepeatedTokensCPU)
{
    // 测试输入中有重复 token ID 的情况
    int vocab_size    = 100;
    int embedding_dim = 16;

    nn::Embedding embed(vocab_size, embedding_dim);
    embed.to(Device(DeviceType::kCPU));

    // 创建输入，token ID 5 出现 3 次
    Tensor input = Tensor({5, 5, 5}, Shape{1, 3}, DataType::kInt32);

    Tensor output = embed.forward(input);

    // 验证所有位置的输出都应该相同（都对应 token ID 5）
    auto weight_data = embed.weight()->data_ptr<float>();
    auto output_data = output.data_ptr<float>();

    for (int pos = 0; pos < 3; ++pos)
    {
        for (int j = 0; j < embedding_dim; ++j)
        {
            float expected = weight_data[5 * embedding_dim + j];
            EXPECT_NEAR(output_data[pos * embedding_dim + j], expected, 1e-5)
                << "Mismatch at position " << pos << ", dim " << j;
        }
    }
}

// ==================== 反向传播测试 ====================

TEST_F(EmbeddingTest, Backward_SingleToken)
{
    // 测试单个 token 的反向传播
    int vocab_size    = 100;
    int embedding_dim = 16;

    nn::Embedding embed(vocab_size, embedding_dim);
    embed.to(Device(DeviceType::kCPU));

    // 创建输入（使用 TensorOptions 设置 requires_grad）
    Tensor input = Tensor({42}, Shape{1}, dtype(DataType::kInt32).requires_grad(true));

    // 前向传播
    Tensor output = embed.forward(input);

    // 反向传播（自动计算梯度）
    output.backward();

    // 验证权重梯度存在
    Tensor grad_weight    = embed.weight()->grad();
    auto grad_weight_data = grad_weight.data_ptr<float>();

    // 只有 token ID 42 的行应该有梯度
    for (size_t i = 0; i < static_cast<size_t>(vocab_size); ++i)
    {
        for (int j = 0; j < embedding_dim; ++j)
        {
            float expected = (i == 42) ? 1.0f : 0.0f;
            EXPECT_NEAR(grad_weight_data[i * embedding_dim + j], expected, 1e-5)
                << "Mismatch at vocab_index=" << i << ", dim=" << j;
        }
    }
}

TEST_F(EmbeddingTest, Backward_GradientAccumulation)
{
    // 测试梯度累加：同一个 token ID 出现多次时，梯度应该累加
    int vocab_size    = 100;
    int embedding_dim = 16;

    nn::Embedding embed(vocab_size, embedding_dim);
    embed.to(Device(DeviceType::kCPU));

    // token ID 5 出现 3 次
    Tensor input = Tensor({5, 5, 5}, Shape{1, 3}, dtype(DataType::kInt32).requires_grad(true));

    Tensor output = embed.forward(input);

    // 反向传播
    output.backward();

    // 验证权重梯度
    Tensor grad_weight    = embed.weight()->grad();
    auto grad_weight_data = grad_weight.data_ptr<float>();

    // token ID 5 的梯度应该是 3.0（因为出现了 3 次，每次梯度都是 1）
    for (int j = 0; j < embedding_dim; ++j)
    {
        EXPECT_NEAR(grad_weight_data[5 * embedding_dim + j], 3.0f, 1e-5) << "Gradient accumulation failed at dim=" << j;
    }

    // 其他 token 的梯度应该为 0
    for (int i = 0; i < vocab_size; ++i)
    {
        if (i != 5)
        {
            for (int j = 0; j < embedding_dim; ++j)
            {
                EXPECT_NEAR(grad_weight_data[i * embedding_dim + j], 0.0f, 1e-5)
                    << "Non-zero gradient for unused token at vocab_index=" << i;
            }
        }
    }
}

TEST_F(EmbeddingTest, Backward_MultipleTokens)
{
    // 测试多个不同 token 的梯度计算
    int vocab_size    = 100;
    int embedding_dim = 16;

    nn::Embedding embed(vocab_size, embedding_dim);
    embed.to(Device(DeviceType::kCPU));

    // 输入包含多个不同的 token，且有重复
    Tensor input = Tensor({10, 20, 10, 30}, Shape{1, 4}, dtype(DataType::kInt32).requires_grad(true));

    Tensor output = embed.forward(input);

    // 反向传播
    output.backward();

    // 验证权重梯度
    Tensor grad_weight    = embed.weight()->grad();
    auto grad_weight_data = grad_weight.data_ptr<float>();

    // token 10 出现两次，梯度应该是 2.0
    for (int j = 0; j < embedding_dim; ++j)
    {
        EXPECT_NEAR(grad_weight_data[10 * embedding_dim + j], 2.0f, 1e-5) << "Token 10 gradient mismatch at dim=" << j;
    }

    // token 20 出现一次，梯度应该是 1.0
    for (int j = 0; j < embedding_dim; ++j)
    {
        EXPECT_NEAR(grad_weight_data[20 * embedding_dim + j], 1.0f, 1e-5) << "Token 20 gradient mismatch at dim=" << j;
    }

    // token 30 出现一次，梯度应该是 1.0
    for (int j = 0; j < embedding_dim; ++j)
    {
        EXPECT_NEAR(grad_weight_data[30 * embedding_dim + j], 1.0f, 1e-5) << "Token 30 gradient mismatch at dim=" << j;
    }

    // 其他 token 的梯度应该为 0
    for (int i = 0; i < vocab_size; ++i)
    {
        if (i != 10 && i != 20 && i != 30)
        {
            for (int j = 0; j < embedding_dim; ++j)
            {
                EXPECT_NEAR(grad_weight_data[i * embedding_dim + j], 0.0f, 1e-5)
                    << "Non-zero gradient for unused token at vocab_index=" << i;
            }
        }
    }
}

TEST_F(EmbeddingTest, Backward_2DInput)
{
    // 测试 2D 输入的反向传播
    int vocab_size    = 100;
    int embedding_dim = 16;

    nn::Embedding embed(vocab_size, embedding_dim);
    embed.to(Device(DeviceType::kCPU));

    // 2D 输入: (2, 3)
    Tensor input = Tensor({1, 5, 10, 2, 8, 15}, Shape{2, 3}, dtype(DataType::kInt32).requires_grad(true));

    Tensor output = embed.forward(input);

    // 反向传播
    output.backward();

    // 验证权重梯度
    Tensor grad_weight    = embed.weight()->grad();
    auto grad_weight_data = grad_weight.data_ptr<float>();

    // 验证被访问的 token 的梯度（每个 token 出现一次，梯度为 1）
    int expected_tokens[] = {1, 2, 5, 8, 10, 15};

    for (int k = 0; k < 6; ++k)
    {
        int token_id = expected_tokens[k];

        for (int j = 0; j < embedding_dim; ++j)
        {
            EXPECT_NEAR(grad_weight_data[token_id * embedding_dim + j], 1.0f, 1e-5)
                << "Token " << token_id << " gradient mismatch at dim=" << j;
        }
    }
}

TEST_F(EmbeddingTest, Backward_ZeroGradient)
{
    // 测试：如果没有调用 backward，梯度应该为空或零
    int vocab_size    = 100;
    int embedding_dim = 16;

    nn::Embedding embed(vocab_size, embedding_dim);
    embed.to(Device(DeviceType::kCPU));

    Tensor input = Tensor({5, 10}, Shape{1, 2}, dtype(DataType::kInt32).requires_grad(true));

    Tensor output = embed.forward(input);

    // 不调用 backward，检查梯度是否存在
    Tensor grad_weight = embed.weight()->grad();
    // 梯度应该存在（已分配内存）
    EXPECT_GT(grad_weight.elements(), 0);
}

TEST_F(EmbeddingTest, Backward_LargeEmbeddingDim)
{
    // 测试较大 embedding 维度的反向传播
    int vocab_size    = 100;
    int embedding_dim = 512;

    nn::Embedding embed(vocab_size, embedding_dim);
    embed.to(Device(DeviceType::kCPU));

    Tensor input = Tensor({42}, Shape{1}, dtype(DataType::kInt32).requires_grad(true));

    Tensor output = embed.forward(input);

    // 反向传播
    output.backward();

    // 验证权重梯度
    Tensor grad_weight    = embed.weight()->grad();
    auto grad_weight_data = grad_weight.data_ptr<float>();

    // 验证 token ID 42 的梯度
    for (int j = 0; j < embedding_dim; ++j)
    {
        EXPECT_NEAR(grad_weight_data[42 * embedding_dim + j], 1.0f, 1e-5) << "Mismatch at dim=" << j;
    }

    // 验证其他 token 的梯度为 0
    for (int i = 0; i < vocab_size; ++i)
    {
        if (i != 42)
        {
            for (int j = 0; j < embedding_dim; ++j)
            {
                EXPECT_NEAR(grad_weight_data[i * embedding_dim + j], 0.0f, 1e-5)
                    << "Non-zero gradient for token " << i << " at dim=" << j;
            }
        }
    }
}

// ==================== 多数据类型测试 ====================

TEST_F(EmbeddingTest, Float32_Default)
{
    // 测试默认使用 float32 的 Embedding 层
    int vocab_size    = 100;
    int embedding_dim = 16;

    auto embed = nn::Embedding(vocab_size, embedding_dim);
    embed.to(Device(DeviceType::kCPU));

    // 验证默认数据类型是 float32
    EXPECT_EQ(embed.dtype(), DataType::kFloat32);
    EXPECT_EQ(embed.weight()->dtype(), DataType::kFloat32);

    Tensor input = Tensor({5, 10, 15}, Shape{1, 3}, DataType::kInt32);
    Tensor output = embed.forward(input);

    EXPECT_EQ(output.dtype(), DataType::kFloat32);
}

TEST_F(EmbeddingTest, Float64_ForwardOnly)
{
    // 测试使用 float64 权重的 Embedding 层
    int vocab_size    = 100;
    int embedding_dim = 16;

    auto embed = nn::Embedding(vocab_size, embedding_dim, DataType::kFloat64);
    embed.to(Device(DeviceType::kCPU));

    // 验证数据类型是 float64
    EXPECT_EQ(embed.dtype(), DataType::kFloat64);
    EXPECT_EQ(embed.weight()->dtype(), DataType::kFloat64);

    Tensor input = Tensor({5, 10, 5}, Shape{1, 3}, DataType::kInt32);

    // 前向传播
    Tensor output = embed.forward(input);

    // 验证输出形状
    EXPECT_EQ(output.shape().size(), 3U);
    EXPECT_EQ(output.shape()[0], 1U);
    EXPECT_EQ(output.shape()[1], 3U);
    EXPECT_EQ(output.shape()[2], static_cast<size_t>(embedding_dim));

    // 验证输出数据类型是 float64
    EXPECT_EQ(output.dtype(), DataType::kFloat64);

    // 验证输出值正确
    const double *weight_data = embed.weight()->data_ptr<double>();
    const double *output_data = output.data_ptr<double>();

    // 验证第一个位置的输出（token 5）
    for (int j = 0; j < embedding_dim; ++j)
    {
        double expected = weight_data[5 * embedding_dim + j];
        EXPECT_NEAR(output_data[j], expected, 1e-9) << "Mismatch at position 0, dim=" << j;
    }

    // 验证第三个位置的输出（token 5）
    for (int j = 0; j < embedding_dim; ++j)
    {
        double expected = weight_data[5 * embedding_dim + j];
        EXPECT_NEAR(output_data[2 * embedding_dim + j], expected, 1e-9) << "Mismatch at position 2, dim=" << j;
    }
}

TEST_F(EmbeddingTest, Float64_ForwardAndBackward)
{
    // 测试 float64 类型的前向和反向传播
    int vocab_size    = 100;
    int embedding_dim = 16;

    auto embed = nn::Embedding(vocab_size, embedding_dim, DataType::kFloat64);
    embed.to(Device(DeviceType::kCPU));

    // 验证数据类型
    EXPECT_EQ(embed.dtype(), DataType::kFloat64);
    EXPECT_EQ(embed.weight()->dtype(), DataType::kFloat64);

    // 创建输入 - 注意：需要使用 requires_grad(true) 才能正确初始化 indices_
    Tensor input = Tensor({5}, Shape{1}, dtype(DataType::kInt32).requires_grad(true));

    // 前向传播
    Tensor output = embed.forward(input);

    // 验证输出数据类型
    EXPECT_EQ(output.dtype(), DataType::kFloat64);

    // 反向传播
    output.backward();

    // 验证梯度存在
    Tensor grad_weight = embed.weight()->grad();
    EXPECT_GT(grad_weight.elements(), 0U);
    EXPECT_EQ(grad_weight.dtype(), DataType::kFloat64);

    // 验证梯度值（token 5 的梯度应该为 1.0）
    const double *grad_data = grad_weight.data_ptr<double>();
    for (int j = 0; j < embedding_dim; ++j)
    {
        EXPECT_NEAR(grad_data[5 * embedding_dim + j], 1.0, 1e-9) << "Mismatch at dim=" << j;
    }
}

TEST_F(EmbeddingTest, DataType_Int32InputOnly)
{
    // 验证输入必须是 int32 类型
    int vocab_size    = 50;
    int embedding_dim = 8;

    nn::Embedding embed(vocab_size, embedding_dim);
    embed.to(Device(DeviceType::kCPU));

    // 尝试使用 float32 作为输入（应该抛出异常）
    Tensor invalid_input = Tensor({1.0f, 2.0f, 3.0f}, Shape{1, 3}, DataType::kFloat32);

    EXPECT_THROW(
        {
            try
            {
                Tensor output = embed.forward(invalid_input);
            }
            catch (const std::exception &e)
            {
                std::string msg(e.what());
                EXPECT_TRUE(msg.find("int32") != std::string::npos);
                throw;
            }
        },
        std::exception);
}

TEST_F(EmbeddingTest, InvalidDtype_Int32)
{
    // 验证不能使用 int32 作为 Embedding 层的 dtype
    int vocab_size    = 50;
    int embedding_dim = 8;

    EXPECT_THROW(
        {
            try
            {
                nn::Embedding embed(vocab_size, embedding_dim, DataType::kInt32);
            }
            catch (const std::exception &e)
            {
                std::string msg(e.what());
                EXPECT_TRUE(msg.find("floating-point") != std::string::npos ||
                           msg.find("float32") != std::string::npos ||
                           msg.find("float64") != std::string::npos);
                throw;
            }
        },
        std::exception);
}

TEST_F(EmbeddingTest, InvalidDtype_Int8)
{
    // 验证不能使用 int8 作为 Embedding 层的 dtype
    int vocab_size    = 50;
    int embedding_dim = 8;

    EXPECT_THROW(
        {
            try
            {
                nn::Embedding embed(vocab_size, embedding_dim, DataType::kInt8);
            }
            catch (const std::exception &e)
            {
                std::string msg(e.what());
                EXPECT_TRUE(msg.find("floating-point") != std::string::npos);
                throw;
            }
        },
        std::exception);
}

TEST_F(EmbeddingTest, InvalidDtype_Int64)
{
    // 验证不能使用 int64 作为 Embedding 层的 dtype
    int vocab_size    = 50;
    int embedding_dim = 8;

    EXPECT_THROW(
        {
            try
            {
                nn::Embedding embed(vocab_size, embedding_dim, DataType::kInt64);
            }
            catch (const std::exception &e)
            {
                std::string msg(e.what());
                EXPECT_TRUE(msg.find("floating-point") != std::string::npos);
                throw;
            }
        },
        std::exception);
}

TEST_F(EmbeddingTest, InvalidDtype_UInt8)
{
    // 验证不能使用 uint8 作为 Embedding 层的 dtype
    int vocab_size    = 50;
    int embedding_dim = 8;

    EXPECT_THROW(
        {
            try
            {
                nn::Embedding embed(vocab_size, embedding_dim, DataType::kUInt8);
            }
            catch (const std::exception &e)
            {
                std::string msg(e.what());
                EXPECT_TRUE(msg.find("floating-point") != std::string::npos);
                throw;
            }
        },
        std::exception);
}

// ==================== CUDA 测试 ====================

TEST_F(EmbeddingTest, BasicForwardCUDA_SingleToken)
{
    // 测试 CUDA 版本的单个 token 查表
    int vocab_size    = 100;
    int embedding_dim = 16;

    nn::Embedding embed(vocab_size, embedding_dim);

    // 检查是否有 CUDA 可用
    if (!origin::test::TestUtils::isCudaAvailable())
    {
        GTEST_SKIP() << "CUDA not available, skipping CUDA tests";
    }

    embed.to(Device(DeviceType::kCUDA, 0));

    // 创建单个 token 的输入（先在 CPU 创建，然后移到 CUDA）
    Tensor input = Tensor({42}, Shape{1}, DataType::kInt32);
    input = input.to(Device(DeviceType::kCUDA, 0));

    // 前向传播
    Tensor output = embed.forward(input);

    // 验证输出形状
    EXPECT_EQ(output.shape().size(), 2U);
    EXPECT_EQ(output.shape()[0], 1U);
    EXPECT_EQ(output.shape()[1], static_cast<size_t>(embedding_dim));

    // 将输出移到 CPU 验证值
    Tensor output_cpu = output.to(Device(DeviceType::kCPU));
    Tensor weight_cpu = static_cast<Tensor>(*embed.weight()).to(Device(DeviceType::kCPU));

    auto weight_data = weight_cpu.data_ptr<float>();
    auto output_data = output_cpu.data_ptr<float>();

    for (int j = 0; j < embedding_dim; ++j)
    {
        float expected = weight_data[42 * embedding_dim + j];
        EXPECT_NEAR(output_data[j], expected, 1e-5) << "Mismatch at dim=" << j;
    }
}

TEST_F(EmbeddingTest, BasicForwardCUDA_MultipleTokens)
{
    // 测试 CUDA 版本的多个 token 查表
    int vocab_size    = 100;
    int embedding_dim = 16;

    nn::Embedding embed(vocab_size, embedding_dim);

    if (!origin::test::TestUtils::isCudaAvailable())
    {
        GTEST_SKIP() << "CUDA not available, skipping CUDA tests";
    }

    embed.to(Device(DeviceType::kCUDA, 0));

    // 创建输入
    Tensor input = Tensor({1, 5, 10, 2, 8, 15}, Shape{2, 3}, DataType::kInt32);
    input = input.to(Device(DeviceType::kCUDA, 0));

    // 前向传播
    Tensor output = embed.forward(input);

    // 验证输出形状
    EXPECT_EQ(output.shape().size(), 3U);
    EXPECT_EQ(output.shape()[0], 2U);
    EXPECT_EQ(output.shape()[1], 3U);
    EXPECT_EQ(output.shape()[2], static_cast<size_t>(embedding_dim));

    // 将输出移到 CPU 验证值
    Tensor output_cpu = output.to(Device(DeviceType::kCPU));
    Tensor weight_cpu = static_cast<Tensor>(*embed.weight()).to(Device(DeviceType::kCPU));

    auto weight_data = weight_cpu.data_ptr<float>();
    auto output_data = output_cpu.data_ptr<float>();

    int expected_tokens[2][3] = {{1, 5, 10}, {2, 8, 15}};

    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            int token_id = expected_tokens[i][j];
            for (int k = 0; k < embedding_dim; ++k)
            {
                size_t output_idx = i * 3 * embedding_dim + j * embedding_dim + k;
                float expected    = weight_data[token_id * embedding_dim + k];
                EXPECT_NEAR(output_data[output_idx], expected, 1e-5)
                    << "Mismatch at position (" << i << ", " << j << ", " << k << ")";
            }
        }
    }
}

TEST_F(EmbeddingTest, BackwardCUDA_SingleToken)
{
    // 测试 CUDA 版本的反向传播
    int vocab_size    = 100;
    int embedding_dim = 16;

    nn::Embedding embed(vocab_size, embedding_dim);

    if (!origin::test::TestUtils::isCudaAvailable())
    {
        GTEST_SKIP() << "CUDA not available, skipping CUDA tests";
    }

    embed.to(Device(DeviceType::kCUDA, 0));

    // 创建输入
    Tensor input = Tensor({42}, Shape{1}, dtype(DataType::kInt32).requires_grad(true));
    input = input.to(Device(DeviceType::kCUDA, 0));

    // 前向传播
    Tensor output = embed.forward(input);

    // 反向传播
    output.backward();

    // 验证权重梯度
    Tensor grad_weight = static_cast<Tensor>(*embed.weight()).grad();
    Tensor grad_weight_cpu = grad_weight.to(Device(DeviceType::kCPU));

    auto grad_data = grad_weight_cpu.data_ptr<float>();

    // 只有 token ID 42 的行应该有梯度
    for (size_t i = 0; i < static_cast<size_t>(vocab_size); ++i)
    {
        for (int j = 0; j < embedding_dim; ++j)
        {
            float expected = (i == 42) ? 1.0f : 0.0f;
            EXPECT_NEAR(grad_data[i * embedding_dim + j], expected, 1e-4)
                << "Mismatch at vocab_index=" << i << ", dim=" << j;
        }
    }
}

TEST_F(EmbeddingTest, Float64_CUDA_ForwardAndBackward)
{
    // 测试 CUDA 版本的 Float64 前向和反向传播
    int vocab_size    = 100;
    int embedding_dim = 16;

    auto embed = nn::Embedding(vocab_size, embedding_dim, DataType::kFloat64);

    if (!origin::test::TestUtils::isCudaAvailable())
    {
        GTEST_SKIP() << "CUDA not available, skipping CUDA tests";
    }

    embed.to(Device(DeviceType::kCUDA, 0));

    // 创建输入
    Tensor input = Tensor({5, 10}, Shape{2}, dtype(DataType::kInt32).requires_grad(true));
    input = input.to(Device(DeviceType::kCUDA, 0));

    // 前向传播
    Tensor output = embed.forward(input);

    EXPECT_EQ(output.dtype(), DataType::kFloat64);

    // 反向传播
    output.backward();

    // 验证梯度
    Tensor grad_weight = static_cast<Tensor>(*embed.weight()).grad();
    Tensor grad_weight_cpu = grad_weight.to(Device(DeviceType::kCPU));

    auto grad_data = grad_weight_cpu.data_ptr<double>();

    // token 5 和 10 的梯度应该为 1.0
    for (int j = 0; j < embedding_dim; ++j)
    {
        EXPECT_NEAR(grad_data[5 * embedding_dim + j], 1.0, 1e-9) << "Token 5 gradient mismatch at dim=" << j;
        EXPECT_NEAR(grad_data[10 * embedding_dim + j], 1.0, 1e-9) << "Token 10 gradient mismatch at dim=" << j;
    }
}

// ==================== 主函数 ====================

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
