#include "dlOperator.h"

namespace dl
{

void Operator::setup_computation_graph(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs)
{
    int max_gen = 0;
    for (const auto &input : inputs)
    {
        if (input.impl_->generation_ > max_gen)
        {
            max_gen = input.impl_->generation_;
        }
    }
    this->generation_ = max_gen;

    this->inputs_ = inputs;
    this->outputs_.clear();
    for (const auto &output : outputs)
    {
        // 关键问题：值语义 Tensor 的生命周期管理
        //
        // 原始设计（使用 weak_ptr）的问题：
        // 1. 用户代码：Tensor y = x0 + x1;  // y 是值对象
        // 2. 算子存储：weak_ptr<Tensor> 指向 y
        // 3. 问题：当 y 超出作用域时，weak_ptr 失效
        // 4. 反向传播：TensorImpl::backward() 无法访问失效的输出
        //
        // 当前解决方案（使用 shared_ptr）：
        // 1. 创建新的 shared_ptr<Tensor> 副本
        // 2. 延长输出张量的生命周期
        // 3. 确保反向传播时输出仍然有效
        // 4. 代价：违背了原始的所有权模型
        //
        // 未来改进方向：
        // - 重新设计 Tensor 的生命周期管理
        // - 或者让用户代码使用 TensorPtr 而不是 Tensor
        // - 或者实现更智能的弱引用管理
        auto tensor_ptr = std::make_shared<Tensor>(output);
        this->outputs_.push_back(tensor_ptr);
    }
}

}  // namespace dl
