#include "origin/optim/sgd.h"
#include <any>
#include <unordered_map>
#include "origin/core/operator.h"
#include "origin/core/parameter.h"
#include "origin/core/tensor.h"
#include "origin/io/model_io.h"
#include "origin/mat/scalar.h"
#include "origin/utils/exception.h"

namespace origin
{

SGD::SGD(Module &target, float lr, float momentum, float weight_decay, bool nesterov)
    : Optimizer(target), lr_(lr), momentum_(momentum), weight_decay_(weight_decay), nesterov_(nesterov)
{}

void SGD::step_one(Parameter &param)
{
    // 获取梯度
    auto grad = param.grad();

    // 这里直接在 SGD 内部实现“经典版” SGD 行为（带 weight_decay / momentum / Nesterov），
    // 而不是完全依赖 Optimizer::hooks_，主要考虑：
    // 1）和 PyTorch 的 torch.optim.SGD 接口对齐：传入 weight_decay/momentum/nesterov 时，
    //    期望一个“打包好的”优化算法，而不是还要额外注册 Hook；
    // 2）动量和 Nesterov 逻辑本身依赖内部状态（momentum_buffers_），天然属于具体优化器实现的一部分；
    // 3）全局 Hook（如 WeightDecay）仍然可以叠加在所有优化器之上，用于实现更通用的正则/裁剪策略，
    //    而 SGD 自己的 weight_decay 字段则对应“本优化器内部集成的 L2 正则”。
    //
    // 换句话说：Hook 提供的是“额外的、可组合”的修改通道，而 SGD 里这段是为了直接支持
    // torch 风格的超参数（lr/momentum/weight_decay/nesterov）而内置的标准实现。

    // 权重衰减
    if (weight_decay_ > 0.0f)
    {
        grad = grad + param * weight_decay_;
    }

    // 动量更新
    if (momentum_ > 0.0f)
    {
        auto it = momentum_buffers_.find(&param);
        if (it == momentum_buffers_.end())
        {
            // 初始化动量缓冲区
            momentum_buffers_[&param] = Tensor::zeros(grad.shape(), TensorOptions(grad.dtype()).device(grad.device()));
        }

        auto &buffer = momentum_buffers_[&param];

        // 更新动量：buffer = momentum * buffer + grad
        buffer = buffer * Scalar(momentum_) + grad;

        // Nesterov动量
        if (nesterov_)
        {
            grad = grad + buffer * Scalar(momentum_);
        }
        else
        {
            grad = buffer;
        }
    }

    // 更新参数：param = param - lr * grad
    // 注意：param是引用，指向Module中存储的Parameter对象
    // 计算新的参数值
    auto updated = param - grad * Scalar(lr_);

    // 使用Parameter的赋值运算符更新参数
    // 这会调用Parameter::operator=(const Tensor&)，它会调用Tensor::operator=
    // Tensor::operator=会将impl_指针（shared_ptr）更新为新的TensorImpl
    param = Parameter(updated);
}

std::unordered_map<std::string, std::any> SGD::state_dict() const
{
    std::unordered_map<std::string, std::any> state;

    // 保存优化器配置
    state["lr"]           = lr_;
    state["momentum"]     = momentum_;
    state["weight_decay"] = weight_decay_;
    state["nesterov"]     = nesterov_;

    // 获取参数名称映射
    auto named_params = target_->named_parameters("");

    // 保存 momentum_buffers（如果使用动量）
    if (momentum_ > 0.0f)
    {
        StateDict momentum_buffers_dict;
        for (const auto &[param_ptr, momentum_buffer] : momentum_buffers_)
        {
            // 查找参数名称
            std::string param_name;
            for (const auto &[name, named_param] : named_params)
            {
                if (named_param == param_ptr)
                {
                    param_name = name;
                    break;
                }
            }
            if (!param_name.empty())
            {
                momentum_buffers_dict["momentum_" + param_name] = momentum_buffer;
            }
        }
        state["momentum_buffers"] = momentum_buffers_dict;
    }

    return state;
}

void SGD::load_state_dict(const std::unordered_map<std::string, std::any> &state_dict)
{
    // 加载优化器配置
    if (state_dict.find("lr") != state_dict.end())
    {
        lr_ = std::any_cast<float>(state_dict.at("lr"));
    }
    if (state_dict.find("momentum") != state_dict.end())
    {
        momentum_ = std::any_cast<float>(state_dict.at("momentum"));
    }
    if (state_dict.find("weight_decay") != state_dict.end())
    {
        weight_decay_ = std::any_cast<float>(state_dict.at("weight_decay"));
    }
    if (state_dict.find("nesterov") != state_dict.end())
    {
        nesterov_ = std::any_cast<bool>(state_dict.at("nesterov"));
    }

    // 加载 momentum_buffers
    if (state_dict.find("momentum_buffers") != state_dict.end() && momentum_ > 0.0f)
    {
        auto momentum_buffers_dict = std::any_cast<StateDict>(state_dict.at("momentum_buffers"));
        auto named_params          = target_->named_parameters("");
        momentum_buffers_.clear();
        for (const auto &[key, tensor] : momentum_buffers_dict)
        {
            // key 格式: "momentum_param_name"，提取 param_name
            if (key.size() > 10 && key.substr(0, 10) == "momentum_")
            {
                std::string param_name = key.substr(10);
                if (named_params.find(param_name) != named_params.end())
                {
                    Parameter *param_ptr         = named_params[param_name];
                    momentum_buffers_[param_ptr] = tensor;
                }
            }
        }
    }
}

}  // namespace origin
