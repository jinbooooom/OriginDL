#ifndef __ORIGIN_DL_FUNCTION_H__
#define __ORIGIN_DL_FUNCTION_H__

#include "dlTensor.h"

namespace dl
{

class Function : public std::enable_shared_from_this<Function>
{
  public:
    virtual ~Function() {}

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

class Square : public Function
{
  public:
    NdArrayPtrList Forward(const NdArrayPtrList &xs) override;

    NdArrayPtrList Backward(const NdArrayPtrList &gys) override;
};

extern VariablePtr square(const VariablePtr &x);
VariablePtr operator^(const VariablePtr &lhs, int n);

class Exp : public Function
{
  public:
    NdArrayPtrList Forward(const NdArrayPtrList &xs) override;

    NdArrayPtrList Backward(const NdArrayPtrList &gys) override;
};

extern VariablePtr exp(const VariablePtr &x);

class Add : public Function
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

class Mul : public Function
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

extern NdArray NumericalDiff(std::function<Variable(Variable)> f, const Variable &x, data_t eps = 1e-4);

}  // namespace dl

#endif