#include "origin/optim/sgd.h"
#include <unordered_map>
#include "origin/core/operator.h"
#include "origin/core/parameter.h"
#include "origin/core/tensor.h"
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

    // 调试：检查参数更新前的值（已修复，移除调试输出）
    // float old_val = 0.0f;
    // try {
    //     if (param.shape().elements() == 1) {
    //         old_val = param.item<float>();
    //     }
    // } catch (...) {}

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

    // 调试：检查参数更新后的值（已修复，移除调试输出）
    // float new_val = 0.0f;
    // try {
    //     if (param.shape().elements() == 1) {
    //         new_val = param.item<float>();
    //         std::cout << "   SGD Debug: param old_val=" << old_val << ", expected=" << expected_new_val << ",
    //         actual=" << new_val << std::endl;
    //     }
    // } catch (...) {}
}

}  // namespace origin
