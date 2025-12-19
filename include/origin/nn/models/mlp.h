#ifndef __ORIGIN_DL_MLP_H__
#define __ORIGIN_DL_MLP_H__

#include <functional>
#include <vector>
#include "../module.h"
#include "../layers/linear.h"
#include "../../core/tensor.h"
#include "../../core/operator.h"

namespace origin
{

/**
 * @brief 多层感知机（MLP）模型
 * 
 * 支持自定义隐藏层大小和激活函数
 * 例如：MLP({784, 100, 100, 10}, relu) 创建一个3层MLP，输入784维，两个隐藏层各100维，输出10维
 */
class MLP : public Module
{
private:
    std::vector<std::unique_ptr<Linear>> layers_;  // 线性层列表
    std::function<Tensor(const Tensor &)> activation_;  // 激活函数

public:
    /**
     * @brief 构造函数
     * @param hidden_sizes 隐藏层大小列表，例如 {784, 100, 100, 10} 表示输入784，隐藏层100和100，输出10
     * @param activation 激活函数，默认为relu
     */
    MLP(const std::vector<int> &hidden_sizes, 
        std::function<Tensor(const Tensor &)> activation = nullptr);

    /**
     * @brief 前向传播
     * @param input 输入张量，形状为 (batch_size, input_size)
     * @return 输出张量，形状为 (batch_size, output_size)
     */
    Tensor forward(const Tensor &input) override;

    /**
     * @brief 重写parameters以收集所有层的参数
     * @return 参数向量
     */
    std::vector<Parameter *> parameters() override;

    /**
     * @brief 重写to以迁移所有层到指定设备
     * @param device 目标设备
     */
    void to(Device device) override;
};

}  // namespace origin

#endif  // __ORIGIN_DL_MLP_H__

