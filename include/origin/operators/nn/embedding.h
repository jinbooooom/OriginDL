#ifndef __ORIGIN_DL_EMBEDDING_OPERATOR_H__
#define __ORIGIN_DL_EMBEDDING_OPERATOR_H__
#include "../../core/operator.h"
namespace origin
{
namespace functional
{
/**
 * @brief Embedding 算子（Embedding层）
 * @details 实现 tokenID -> 向量
 * 输入：x (N,),
 * 输出：y (N, embedding_dim)
 */
class EmbeddingOp : public Operator
{

public:
    int vocab_size_;
    int embedding_dim_;
    EmbeddingOp(int vocab_size, int embedding_dim) : vocab_size_(vocab_size), embedding_dim_(embedding_dim) {};
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;
    std::vector<Tensor> backward(const std::vector<Tensor> &gxs) override;

private:
    Tensor indices_;  // 为了反向传播
};

/**
 * @brief 函数式接口 embedding 算子
 * @param x 输入张量（索引，必须是 int32 类型）
 * @param weight 查表权重 (vocab_size, embedding_dim)
 * @param vocab_size_ 词汇表大小
 * @param embedding_dim_ 嵌入维度
 * @return 向量化的token
 */
Tensor embedding(const Tensor &x, const Tensor &weight, int vocab_size_, int embedding_dim_);
}  // namespace functional

}  // namespace origin

#endif