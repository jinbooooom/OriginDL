#include "origin/core/operator.h"

namespace origin
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
        // 根本解决方案：使用 weak_ptr 避免循环引用
        // 1. Tensor 是值语义的，但 TensorImpl 是引用语义的（通过 shared_ptr 管理）
        // 2. 我们存储 TensorImpl 的 weak_ptr，而不是 Tensor 的 weak_ptr
        // 3. 在 backward() 时，将 weak_ptr 转换为 shared_ptr（如果有效）
        // 4. 如果 weak_ptr 失效，说明用户代码中的 tensor 已经超出作用域，这是正常的
        // 5. 这样可以避免循环引用（Operator -> outputs_ -> TensorImpl -> creator_ -> Operator），解决内存泄漏
        if (output.impl_)
        {
            this->outputs_.push_back(output.impl_);  // 存储 weak_ptr<TensorImpl>
        }
    }
}

}  // namespace origin
