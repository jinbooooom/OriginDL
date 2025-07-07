#ifndef __ORIGIN_DL_OPERATOR_H__
#define __ORIGIN_DL_OPERATOR_H__

#include "dlTensor.h"

namespace dl
{

class Operator : public std::enable_shared_from_this<Operator>
{
  public:
    virtual ~Operator() {}

    VariablePtrList operator()(const VariablePtr &input);

    VariablePtrList operator()(const VariablePtrList &inputs);

    virtual NdArrayPtrList Forward(const NdArrayPtrList &xs) = 0;

    virtual NdArrayPtrList Backward(const NdArrayPtrList &gys) = 0;

  public:
    VariablePtrList inputs;  // 前向传播的入参，考虑多输入

    // 算子拥有 inputs，但是不拥有 outputs，outputs 是下一个计算节点的输入，被下一个计算节点所拥有，
    // 因此 output 为 weak 指针，表示当前算子仅仅是使用 output，不拥有所有权。
    VariableWPtrList outputs;  // 前向传播的输出，考虑多输出

    int generation;  // 对于复杂的计算图，用来区分哪个先计算
};

class Neg : public Operator
{
  public:
    NdArrayPtrList Forward(const NdArrayPtrList &xs) override;

    NdArrayPtrList Backward(const NdArrayPtrList &gys) override;
};

VariablePtr neg(const VariablePtrList &xs);
VariablePtr neg(const VariablePtr &x);
VariablePtr operator-(const VariablePtr &x);

class Add : public Operator
{
  public:
    NdArrayPtrList Forward(const NdArrayPtrList &xs) override;

    NdArrayPtrList Backward(const NdArrayPtrList &gys) override;
};

extern VariablePtr add(const VariablePtrList &xs);
extern VariablePtr add(const VariablePtr &lhs, const VariablePtr &rhs);
VariablePtr operator+(const VariablePtr &lhs, const VariablePtr &rhs);
VariablePtr operator+(const VariablePtr &lhs, data_t rhs);
VariablePtr operator+(data_t lhs, const VariablePtr &rhs);

class Sub : public Operator
{
  public:
    NdArrayPtrList Forward(const NdArrayPtrList &xs) override;

    NdArrayPtrList Backward(const NdArrayPtrList &gys) override;
};

extern VariablePtr sub(const VariablePtrList &xs);
extern VariablePtr sub(const VariablePtr &lhs, const VariablePtr &rhs);
VariablePtr operator-(const VariablePtr &lhs, const VariablePtr &rhs);
VariablePtr operator-(const VariablePtr &lhs, data_t rhs);
VariablePtr operator-(data_t lhs, const VariablePtr &rhs);

class Mul : public Operator
{
  public:
    NdArrayPtrList Forward(const NdArrayPtrList &xs) override;

    NdArrayPtrList Backward(const NdArrayPtrList &gys) override;
};

extern VariablePtr mul(const VariablePtrList &xs);
extern VariablePtr mul(const VariablePtr &lhs, const VariablePtr &rhs);
VariablePtr operator*(const VariablePtr &lhs, const VariablePtr &rhs);
VariablePtr operator*(const VariablePtr &lhs, data_t rhs);
VariablePtr operator*(data_t lhs, const VariablePtr &rhs);

class Div : public Operator
{
  public:
    NdArrayPtrList Forward(const NdArrayPtrList &xs) override;

    NdArrayPtrList Backward(const NdArrayPtrList &gys) override;
};

extern VariablePtr div(const VariablePtrList &xs);
extern VariablePtr div(const VariablePtr &lhs, const VariablePtr &rhs);
VariablePtr operator/(const VariablePtr &lhs, const VariablePtr &rhs);
VariablePtr operator/(const VariablePtr &lhs, data_t rhs);
VariablePtr operator/(data_t lhs, const VariablePtr &rhs);

class Square : public Operator
{
  public:
    NdArrayPtrList Forward(const NdArrayPtrList &xs) override;

    NdArrayPtrList Backward(const NdArrayPtrList &gys) override;
};

extern VariablePtr square(const VariablePtr &x);

class Pow : public Operator
{
  public:
    Pow(int n) : mExponent(n) {};

    NdArrayPtrList Forward(const NdArrayPtrList &xs) override;

    NdArrayPtrList Backward(const NdArrayPtrList &gys) override;

    int mExponent;  // 幂函数的指数
};
VariablePtr pow(const VariablePtr &base, int exponent);
VariablePtr operator^(const VariablePtr &base, int exponent);

class Exp : public Operator
{
  public:
    NdArrayPtrList Forward(const NdArrayPtrList &xs) override;

    NdArrayPtrList Backward(const NdArrayPtrList &gys) override;
};

extern VariablePtr exp(const VariablePtr &x);

extern NdArray NumericalDiff(std::function<Variable(Variable)> f, const Variable &x, data_t eps = 1e-4);

}  // namespace dl

#endif