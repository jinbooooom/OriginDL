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

private:
    void setup_computation_graph(const std::vector<Tensor> &inputs, const std::vector<Tensor> &outputs);
};

class Neg : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

Tensor neg(const std::vector<Tensor> &xs);
Tensor neg(const Tensor &x);
Tensor operator-(const Tensor &x);

class Add : public Operator
{
public:
    Shape shape0_;
    Shape shape1_;

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

extern Tensor add(const std::vector<Tensor> &xs);
extern Tensor add(const Tensor &lhs, const Tensor &rhs);
Tensor operator+(const Tensor &lhs, const Tensor &rhs);
Tensor operator+(const Tensor &lhs, const Scalar &rhs);
Tensor operator+(const Scalar &lhs, const Tensor &rhs);

class Sub : public Operator
{
public:
    Shape shape0_;
    Shape shape1_;

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

extern Tensor sub(const std::vector<Tensor> &xs);
extern Tensor sub(const Tensor &lhs, const Tensor &rhs);
Tensor operator-(const Tensor &lhs, const Tensor &rhs);
Tensor operator-(const Tensor &lhs, const Scalar &rhs);
Tensor operator-(const Scalar &lhs, const Tensor &rhs);

class Mul : public Operator
{
public:
    Shape shape0_;
    Shape shape1_;

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

extern Tensor mul(const std::vector<Tensor> &xs);
extern Tensor mul(const Tensor &lhs, const Tensor &rhs);
Tensor operator*(const Tensor &lhs, const Tensor &rhs);
Tensor operator*(const Tensor &lhs, const Scalar &rhs);
Tensor operator*(const Scalar &lhs, const Tensor &rhs);

class Div : public Operator
{
public:
    Shape shape0_;
    Shape shape1_;

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

extern Tensor div(const std::vector<Tensor> &xs);
extern Tensor div(const Tensor &lhs, const Tensor &rhs);
Tensor operator/(const Tensor &lhs, const Tensor &rhs);
Tensor operator/(const Tensor &lhs, const Scalar &rhs);
Tensor operator/(const Scalar &lhs, const Tensor &rhs);

class Square : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

extern Tensor square(const Tensor &x);

class Pow : public Operator
{
public:
    // 支持多种类型的指数构造函数
    Pow(Scalar n) : exponent_(n){};

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;

    Scalar exponent_;  // 幂函数的指数，支持多种数值类型
};

extern Tensor pow(const Tensor &base, const Scalar &exponent);        // 支持标量指数
extern Tensor operator^(const Tensor &base, const Scalar &exponent);  // 支持标量指数

class Exp : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

extern Tensor exp(const Tensor &x);

/**
 * @brief 自然对数算子（以 e 为底的对数）
 *
 * 计算输入张量的自然对数，即 log_e(x) = ln(x)
 * 与 PyTorch 的 torch.log() 行为一致
 */
class Log : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief 计算张量的自然对数（以 e 为底）
 *
 * @param x 输入张量，必须为正数
 * @return 自然对数结果，log_e(x) = ln(x)
 *
 * @note 与 PyTorch 的 torch.log() 行为一致
 */
extern Tensor log(const Tensor &x);

/**
 * @brief Softmax 算子
 *
 * 计算 softmax 归一化，用于多分类任务
 * 公式：softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
 * 注意数值稳定性：先减去最大值再计算
 */
class Softmax : public Operator
{
public:
    int axis_;  // 计算 softmax 的轴，默认为 -1（最后一个维度）

    Softmax() : axis_(-1) {}
    Softmax(int axis) : axis_(axis) {}

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief 计算张量的 softmax 归一化
 *
 * @param x 输入张量
 * @param axis 计算 softmax 的轴，默认为 -1（最后一个维度）
 * @return softmax 归一化结果
 */
extern Tensor softmax(const Tensor &x, int axis = -1);

/**
 * @brief ReLU 激活函数算子
 *
 * 计算 ReLU 激活函数，用于神经网络
 * 公式：ReLU(x) = max(0, x)
 */
class ReLU : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief 计算张量的 ReLU 激活函数
 *
 * @param x 输入张量
 * @return ReLU 激活结果，ReLU(x) = max(0, x)
 */
extern Tensor relu(const Tensor &x);

/**
 * @brief Sigmoid 激活函数算子
 *
 * 计算 Sigmoid 激活函数，用于神经网络
 * 公式：sigmoid(x) = 1 / (1 + exp(-x))
 */
class Sigmoid : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief 计算张量的 Sigmoid 激活函数
 *
 * @param x 输入张量
 * @return Sigmoid 激活结果，sigmoid(x) = 1 / (1 + exp(-x))
 */
extern Tensor sigmoid(const Tensor &x);

/**
 * @brief SoftmaxCrossEntropy 损失函数算子
 *
 * 计算 softmax 交叉熵损失，用于多分类任务
 * 公式：loss = -mean(log(softmax(x)[target]))
 *
 * 输入：
 * - x: (N, C) 形状，N 是 batch size，C 是类别数
 * - target: (N,) 形状，每个元素是类别索引（0 到 C-1）
 *
 * 输出：
 * - loss: 标量
 */
class SoftmaxCrossEntropy : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief 计算 softmax 交叉熵损失
 *
 * @param x 输入 logits，形状为 (N, C)，N 是 batch size，C 是类别数
 * @param target 目标类别索引，形状为 (N,)，每个元素是类别索引（0 到 C-1）
 * @return 交叉熵损失，标量
 */
extern Tensor softmax_cross_entropy(const Tensor &x, const Tensor &target);

class Reshape : public Operator
{
public:
    Shape shape_;  // 输出的形状

    Shape x_shape_;  // 输入的形状

    Reshape(const Shape &shape) : shape_(shape) {}

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};
extern Tensor reshape(const Tensor &x, const Shape &shape);

class Transpose : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};
extern Tensor transpose(const Tensor &x);

class Sum : public Operator
{
public:
    int axis_;  // 对那个轴求和

    Shape x_shape_;  // 输入的形状
    Sum() : axis_(-1){};
    Sum(const int axis) : axis_(axis){};

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};
extern Tensor sum(const Tensor &x, int axis = -1);  // -1 意味着所有元素相加

class BroadcastTo : public Operator
{
public:
    Shape shape_;  // 输出的形状

    Shape x_shape_;  // 输入的形状

    BroadcastTo(const Shape &shape) : shape_(shape){};

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};
extern Tensor broadcast_to(const Tensor &x, const Shape &shape);

class SumTo : public Operator
{
public:
    Shape shape_;  // 输出的形状

    Shape x_shape_;  // 输入的形状

    SumTo(const Shape &shape) : shape_(shape){};

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};
extern Tensor sum_to(const Tensor &x, const Shape &shape);

class MatMul : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};
extern Tensor mat_mul(const Tensor &x, const Tensor &w);

class BatchNorm : public Operator
{
public:
    bool training_;
    float eps_;
    float momentum_;
    int num_dims_;

    BatchNorm(bool training, float eps, float momentum, int num_dims);

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;

private:
    Tensor saved_mean_;
    Tensor saved_var_;
    Tensor saved_x_norm_;
};

class Dropout : public Operator
{
public:
    float p_;  // dropout 概率
    bool training_;

    Dropout(float p, bool training);

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;

private:
    Tensor mask_;
};

}  // namespace origin

#endif