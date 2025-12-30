#ifndef __ORIGIN_DL_DATASET_H__
#define __ORIGIN_DL_DATASET_H__

#include <utility>
#include "../core/tensor.h"

namespace origin
{

/**
 * @brief 数据集抽象基类
 * 
 * 定义数据集的接口规范，所有数据集类都应该继承此类
 */
class Dataset
{
public:
    /**
     * @brief 虚析构函数
     */
    virtual ~Dataset() = default;

    /**
     * @brief 获取单个数据项
     * @param index 数据索引
     * @return 数据对 (input, target)，input 是输入张量，target 是目标张量（标签）
     */
    virtual std::pair<Tensor, Tensor> get_item(size_t index) = 0;

    /**
     * @brief 获取数据集大小
     * @return 数据集中的样本数量
     */
    virtual size_t size() const = 0;

    /**
     * @brief 检查索引是否有效
     * @param index 数据索引
     * @return 索引是否在有效范围内
     */
    bool valid_index(size_t index) const
    {
        return index < size();
    }
};

}  // namespace origin

#endif  // __ORIGIN_DL_DATASET_H__

