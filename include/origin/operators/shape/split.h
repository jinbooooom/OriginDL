#ifndef __ORIGIN_DL_SPLIT_H__
#define __ORIGIN_DL_SPLIT_H__

#include <initializer_list>
#include <type_traits>
#include <vector>
#include "../../core/operator.h"
#include "../../utils/size_array_ref.h"

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
     * @brief 构造函数：从 SizeArrayRef（统一接口，支持 C 数组、vector、initializer_list）
     * @param split_sizes 每个分割的大小列表（通过 SizeArrayRef 传递）
     * @param dim 分割的维度，默认 0
     */
    explicit Split(SizeArrayRef split_sizes, int dim = 0) : dim_(dim), split_sizes_(split_sizes.to_vector()) {}

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
 * @brief Split 函数：从 SizeArrayRef（统一接口，支持 C 数组、vector、initializer_list）
 * @param x 输入张量
 * @param split_sizes 每个分割的大小列表（支持 C 数组、std::vector<size_t>、std::initializer_list<size_t>）
 * @param dim 分割的维度，默认 0
 * @return 分割后的张量列表
 */
std::vector<Tensor> split(const Tensor &x, SizeArrayRef split_sizes, int dim = 0);

/**
 * @brief Split 函数模板重载：SizeArrayRef 仅支持 initializer_list<size_t> 类型，对于其他类型，如 initializer_list<int>
 等，需要进行转换 为了支持 split(x, {1, 2, 3}, 0) 这样的调用方式，需要进行转换
 * @details 将 integral 类型转换为 size_t，然后调用主函数
 */
template <typename T>
requires(std::is_integral_v<T> && !std::is_same_v<T, bool> &&
         !std::is_same_v<T, size_t>) inline std::vector<Tensor> split(const Tensor &x,
                                                                      std::initializer_list<T> split_sizes,
                                                                      int dim = 0)
{
    // 转换为 size_t vector，然后传递给 SizeArrayRef
    std::vector<size_t> vec;
    vec.reserve(split_sizes.size());
    for (const auto &val : split_sizes)
    {
        vec.push_back(static_cast<size_t>(val));
    }
    // 直接传递 vector，SizeArrayRef 零拷贝接收
    return split(x, vec, dim);
}

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
