#include "origin/optim/optimizer.h"
#include "origin/utils/log.h"

namespace origin
{

Optimizer::Optimizer(Module &target) : target_(&target)
{
    collect_parameters();
}

void Optimizer::collect_parameters()
{
    parameters_ = target_->parameters();

    logd("   Optimizer::collect_parameters: collected {} parameters", parameters_.size());
}

void Optimizer::step()
{
    // 过滤有梯度的参数
    std::vector<Parameter *> params_with_grad;
    for (auto *param : parameters_)
    {
        // 检查是否有梯度
        try
        {
            auto grad = param->grad();
            // 注意：即使梯度形状是{}（0维），elements()也会返回1
            // 所以我们需要检查grad_是否存在，而不是检查elements()
            if (grad.shape().elements() > 0)  // 有梯度
            {
                params_with_grad.push_back(param);
            }
        }
        catch (...)
        {
            logw("   Optimizer::step: no gradient for parameter at {}", static_cast<const void *>(param));
        }
    }

    // 执行Hook
    for (auto &hook : hooks_)
    {
        hook(params_with_grad);
    }

    // 更新每个参数
    for (auto *param : params_with_grad)
    {
        step_one(*param);
    }
}

void Optimizer::zero_grad()
{
    // 清除所有参数的梯度
    target_->zero_grad();
}

void Optimizer::register_hook(std::function<void(std::vector<Parameter *> &)> hook)
{
    hooks_.push_back(hook);
}

}  // namespace origin
