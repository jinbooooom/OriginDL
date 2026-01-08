#pragma once

#include "origin/core/tensor.h"

namespace origin
{

/**
 * @brief 计算分类准确率
 *
 * 计算预测结果与真实标签的准确率
 * 公式：accuracy = mean(argmax(y, axis=1) == target)
 *
 * @param y 预测 logits 或概率，形状为 (N, C)，N 是 batch size，C 是类别数
 * @param target 真实标签，形状为 (N,)，每个元素是类别索引（0 到 C-1）
 * @return 准确率，标量张量（float 类型），范围 [0, 1]
 *
 * @note 这是一个评估函数，不需要反向传播，不参与计算图
 */
Tensor accuracy(const Tensor &y, const Tensor &target);

}  // namespace origin
