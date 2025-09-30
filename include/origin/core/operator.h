#ifndef __ORIGIN_DL_OPERATOR_H__
#define __ORIGIN_DL_OPERATOR_H__

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

    // 使用 shared_ptr 而不是 weak_ptr 的原因：
    // 1. 原始设计理念：算子不拥有输出张量，只观察它们（使用 weak_ptr 避免循环引用）
    // 2. 实际问题：当前使用值语义的 Tensor 对象，用户代码中 Tensor 的生命周期与算子存储的引用不一致
    // 3. 生命周期问题：当用户代码中的 Tensor 对象超出作用域时，weak_ptr 会失效
    // 4. 反向传播失败：TensorImpl::backward() 无法访问已失效的输出张量
    // 5. 解决方案：使用 shared_ptr 确保输出张量在反向传播期间仍然有效
    // 6. 权衡：虽然违背了原始的所有权模型，但确保了实际运行时的正确性
    // 7. 未来改进：理想情况下应该重新设计 Tensor 的生命周期管理，恢复 weak_ptr 的使用
    std::vector<std::shared_ptr<Tensor>> outputs_;  // 前向传播的输出，考虑多输出

    int generation_;  // 对于复杂的计算图，用来区分哪个先计算

protected:
    // 获取Mat引用（仅限Operator子类使用）
    const Mat &mat(const Tensor &tensor) const { return *tensor.impl_->data_; }
    Mat &mat(Tensor &tensor) { return *tensor.impl_->data_; }

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
Tensor operator+(const Tensor &lhs, data_t rhs);
Tensor operator+(data_t lhs, const Tensor &rhs);

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
Tensor operator-(const Tensor &lhs, data_t rhs);
Tensor operator-(data_t lhs, const Tensor &rhs);

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
Tensor operator*(const Tensor &lhs, data_t rhs);
Tensor operator*(data_t lhs, const Tensor &rhs);

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
Tensor operator/(const Tensor &lhs, data_t rhs);
Tensor operator/(data_t lhs, const Tensor &rhs);

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
    Pow(int n) : exponent_(n) {};

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;

    int exponent_;  // 幂函数的指数
};
Tensor pow(const Tensor &base, int exponent);
Tensor operator^(const Tensor &base, int exponent);

class Exp : public Operator
{
public:
    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

extern Tensor exp(const Tensor &x);

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
    Sum() : axis_(-1) {};
    Sum(const int axis) : axis_(axis) {};

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};
extern Tensor sum(const Tensor &x, int axis = -1);  // -1 意味着所有元素相加

class BroadcastTo : public Operator
{
public:
    Shape shape_;  // 输出的形状

    Shape x_shape_;  // 输入的形状

    BroadcastTo(const Shape &shape) : shape_(shape) {};

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};
extern Tensor broadcast_to(const Tensor &x, const Shape &shape);

class SumTo : public Operator
{
public:
    Shape shape_;  // 输出的形状

    Shape x_shape_;  // 输入的形状

    SumTo(const Shape &shape) : shape_(shape) {};

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

extern NdArray numerical_diff(std::function<Tensor(Tensor)> f, const Tensor &x, data_t eps = 1e-4);

}  // namespace origin

#endif