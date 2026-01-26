#ifndef __ORIGIN_DL_SPLIT_H__
#define __ORIGIN_DL_SPLIT_H__

#include "../../core/operator.h"
#include <vector>

namespace origin
{
namespace functional
{

/**
 * @brief Split 分割算子
 * @details 在指定维度上将张量分割成多个张量
 */
class Split : public Operator
{
public:
    int dim_;                          // 分割的维度
    std::vector<size_t> split_sizes_;  // 每个分割的大小列表

    /**
     * @brief 构造函数：按大小列表分割
     * @param split_sizes 每个分割的大小列表
     * @param dim 分割的维度，默认 0
     */
    explicit Split(const std::vector<size_t> &split_sizes, int dim = 0) : dim_(dim), split_sizes_(split_sizes) {}

    /**
     * @brief 构造函数：按固定大小分割
     * @param split_size 每个分割的大小
     * @param dim 分割的维度，默认 0
     */
    explicit Split(size_t split_size, int dim = 0) : dim_(dim)
    {
        // split_size 会在 forward 中根据实际大小计算
        split_sizes_.push_back(split_size);
    }

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;
    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief Split 函数：按大小列表分割
 * @param x 输入张量
 * @param split_sizes 每个分割的大小列表
 * @param dim 分割的维度，默认 0
 * @return 分割后的张量列表
 */
std::vector<Tensor> split(const Tensor &x, const std::vector<size_t> &split_sizes, int dim = 0);

/**
 * @brief Split 函数：按固定大小分割
 * @param x 输入张量
 * @param split_size 每个分割的大小
 * @param dim 分割的维度，默认 0
 * @return 分割后的张量列表
 */
std::vector<Tensor> split(const Tensor &x, size_t split_size, int dim = 0);

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_SPLIT_H__
