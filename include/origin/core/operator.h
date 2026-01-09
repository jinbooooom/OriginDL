#ifndef __ORIGIN_DL_OPERATOR_H__
#define __ORIGIN_DL_OPERATOR_H__

#include "../utils/exception.h"
#include "origin/mat/scalar.h"
#include "tensor.h"

namespace origin
{

class Operator : public std::enable_shared_from_this<Operator>
{
public:
    virtual ~Operator() {}

    // 单输入版本 - 返回 Tensor
    Tensor operator()(const Tensor &input) { return (*this)(std::vector<Tensor>{input})[0]; }

    // 多输入版本 - 返回 Tensor 向量
    std::vector<Tensor> operator()(const std::vector<Tensor> &inputs)
    {
        // 直接调用 forward
        auto outputs = this->forward(inputs);

        // 设置 creator
        for (auto &output : outputs)
        {
            output.set_creator(shared_from_this());
        }

        // 设置计算图信息
        this->setup_computation_graph(inputs, outputs);

        return outputs;
    }

    // 纯虚函数 - 使用 Tensor
    virtual std::vector<Tensor> forward(const std::vector<Tensor> &inputs)        = 0;
    virtual std::vector<Tensor> backward(const std::vector<Tensor> &grad_outputs) = 0;

public:
    // 静态成员变量：用于标记空 Tensor，区分一元和二元原地操作
    static const Tensor kNullTensor_;

    std::vector<Tensor> inputs_;  // 前向传播的入参，考虑多输入

    // 根本解决方案：使用 weak_ptr 避免循环引用
    // 1. 原始设计理念：算子不拥有输出张量，只观察它们（使用 weak_ptr 避免循环引用）
    // 2. 生命周期问题：当用户代码中的 Tensor 对象超出作用域时，weak_ptr 会失效
    // 3. 解决方案：在 backward() 时，将 weak_ptr 转换为 shared_ptr（如果有效）
    // 4. 如果 weak_ptr 失效，说明用户代码中的 tensor 已经超出作用域，这是正常的
    // 5. 这样可以避免循环引用（Operator -> outputs_ -> Tensor -> creator_ -> Operator），解决内存泄漏
    //
    // 注意：outputs_ 存储的是 TensorImpl 的 weak_ptr，而不是 Tensor 的 weak_ptr
    // 因为 Tensor 是值语义的，而 TensorImpl 是引用语义的（通过 shared_ptr 管理）
    std::vector<std::weak_ptr<TensorImpl>> outputs_;  // 前向传播的输出，考虑多输出（使用weak_ptr避免循环引用）

    int generation_;  // 对于复杂的计算图，用来区分哪个先计算

protected:
    // 获取Mat引用（仅限Operator子类使用）
    const Mat &mat(const Tensor &tensor) const
    {
        if (!tensor.impl_)
        {
            THROW_RUNTIME_ERROR("mat() called on Tensor with null impl_");
        }
        if (!tensor.impl_->data_)
        {
            THROW_RUNTIME_ERROR("mat() called on Tensor with null data_, shape was: {}",
                                tensor.impl_->shape().to_string());
        }
        return *tensor.impl_->data_;
    }
    Mat &mat(Tensor &tensor)
    {
        if (!tensor.impl_)
        {
            THROW_RUNTIME_ERROR("mat() called on Tensor with null impl_");
        }
        if (!tensor.impl_->data_)
        {
            THROW_RUNTIME_ERROR("mat() called on Tensor with null data_, shape was: {}",
                                tensor.impl_->shape().to_string());
        }
        return *tensor.impl_->data_;
    }

    // 从Mat创建Tensor（仅限Operator子类使用）
    Tensor convert_mat_to_tensor(std::unique_ptr<Mat> mat) { return Tensor(std::move(mat)); }

    // 原地操作虚函数，会修改 input0.（默认实现抛出异常）
    // input1 默认为 kNullTensor_，当 input1 == kNullTensor_ 时表示一元操作，否则为二元操作
    virtual void forward_inplace(Tensor &input0, const Tensor &input1 = kNullTensor_)
    {
        THROW_RUNTIME_ERROR("forward_inplace not implemented for this operator");
    }

private:
    void setup_computation_graph(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs);
};

}  // namespace origin

// 包含所有算子头文件
// Activation operators
#include "origin/operators/activation/relu.h"
#include "origin/operators/activation/sigmoid.h"
#include "origin/operators/activation/silu.h"
#include "origin/operators/activation/softmax.h"

// Convolution operators
#include "origin/operators/conv/conv2d.h"

// Pooling operators
#include "origin/operators/pooling/max_pool2d.h"
#include "origin/operators/pooling/avg_pool2d.h"
#include "origin/operators/pooling/adaptive_avg_pool2d.h"

// Normalization operators
#include "origin/operators/normalization/batch_norm.h"

// Loss operators
#include "origin/operators/loss/softmax_cross_entropy.h"

// Math operators
#include "origin/operators/math/add.h"
#include "origin/operators/math/sub.h"
#include "origin/operators/math/mul.h"
#include "origin/operators/math/div.h"
#include "origin/operators/math/exp.h"
#include "origin/operators/math/log.h"
#include "origin/operators/math/pow.h"
#include "origin/operators/math/neg.h"
#include "origin/operators/math/square.h"
#include "origin/operators/math/mat_mul.h"
#include "origin/operators/math/sum.h"
#include "origin/operators/math/broadcast_to.h"
#include "origin/operators/math/sum_to.h"

// Shape operators
#include "origin/operators/shape/reshape.h"
#include "origin/operators/shape/transpose.h"
#include "origin/operators/shape/flatten.h"
#include "origin/operators/shape/cat.h"

// Neural network operators
#include "origin/operators/nn/dropout.h"
#include "origin/operators/nn/identity.h"
#include "origin/operators/nn/upsample.h"

// Custom operators
#include "origin/operators/custom/linear.h"
#include "origin/operators/custom/yolo_detect.h"

#endif