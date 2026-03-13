#ifndef __ORIGIN_DL_EMBEDDING_H__
#define __ORIGIN_DL_EMBEDDING_H__
#include "../../core/parameter.h"
#include "../../core/tensor.h"
#include "../layer.h"

namespace origin
{
namespace nn
{
/**
 * @brief Embedding 层
 * @details 离散token ID转换为向量表示 (B,S)->(B,S,embedding_dim)
 */
class Embedding : public Layer
{
private:
    Parameter weight_;  // 权重矩阵(vocab_size, embedding_dim)
    int vocab_size_;
    int embedding_dim_;
    DataType dtype_;     // 权重数据类型

public:
    /**
     * @brief 构造函数（默认使用 float32）
     * @param vocab_size 词汇表大小（token ID 的范围 [0, vocab_size-1]）
     * @param embedding_dim 嵌入向量的维度
     */
    Embedding(int vocab_size, int embedding_dim);

    /**
     * @brief 构造函数（指定数据类型）
     * @param vocab_size 词汇表大小（token ID 的范围 [0, vocab_size-1]）
     * @param embedding_dim 嵌入向量的维度
     * @param dtype 权重数据类型（默认 float32，支持 float64 等）
     */
    Embedding(int vocab_size, int embedding_dim, DataType dtype);

    /**
     * @brief 前向传播
     * @param input 输入 token IDs，形状 (batch_size, seq_len)
     * @return 嵌入向量，形状 (batch_size, seq_len, embedding_dim)
     */
    Tensor forward(const Tensor &input) override;

    /**
     * @brief 参数访问
     * @return 权重参数
     */
    Parameter *weight() { return &weight_; }

    /**
     * @brief 获取词汇表大小
     */
    int vocab_size() const { return vocab_size_; }

    /**
     * @brief 获取嵌入维度
     */
    int embedding_dim() const { return embedding_dim_; }

    /**
     * @brief 获取数据类型
     */
    DataType dtype() const { return dtype_; }
    /**
     * @brief 重置参数
     */
    void reset_parameters();

private:
    /**
     * @brief 初始化权重参数
     */
    Parameter init_weight();

    /**
     * @brief 初始化参数（用于reset_parameters）
     */
    void init_parameters();
};
}  // namespace nn
}  // namespace origin
#endif