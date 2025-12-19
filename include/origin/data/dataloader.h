#ifndef __ORIGIN_DL_DATALOADER_H__
#define __ORIGIN_DL_DATALOADER_H__

#include "dataset.h"
#include <vector>
#include <random>
#include <algorithm>
#include <memory>

namespace origin
{

/**
 * @brief 数据加载器
 * 
 * 支持批处理、随机打乱和迭代器接口
 */
class DataLoader
{
private:
    Dataset *dataset_;           // 数据集指针（不拥有所有权）
    size_t batch_size_;          // 批大小
    bool shuffle_;               // 是否随机打乱
    std::vector<size_t> indices_; // 索引列表
    size_t current_index_;       // 当前索引位置

    /**
     * @brief 重置索引列表（如果需要打乱，则随机打乱）
     */
    void reset_indices();

public:
    /**
     * @brief 构造函数
     * @param dataset 数据集引用（不拥有所有权）
     * @param batch_size 批大小，默认为 1
     * @param shuffle 是否随机打乱，默认为 false
     */
    DataLoader(Dataset &dataset, size_t batch_size = 1, bool shuffle = false);

    /**
     * @brief 获取下一个批次
     * @return 批次数据对 (inputs, targets)
     *         - inputs: 形状为 (batch_size, input_size) 的张量
     *         - targets: 形状为 (batch_size,) 的张量
     * @note 如果到达数据集末尾，返回空批次（inputs 和 targets 的元素数为 0）
     */
    std::pair<Tensor, Tensor> next();

    /**
     * @brief 检查是否还有更多数据
     * @return 是否还有更多批次
     */
    bool has_next() const;

    /**
     * @brief 重置数据加载器（重新开始迭代）
     */
    void reset();

    /**
     * @brief 获取数据集大小
     * @return 数据集中的样本数量
     */
    size_t dataset_size() const { return dataset_->size(); }

    /**
     * @brief 获取批大小
     * @return 批大小
     */
    size_t batch_size() const { return batch_size_; }
};

}  // namespace origin

#endif  // __ORIGIN_DL_DATALOADER_H__

