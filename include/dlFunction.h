#ifndef __ORIGIN_DL_FUNCTION_H__
#define __ORIGIN_DL_FUNCTION_H__

#include "dlTensor.h"

namespace dl
{

class Function : public std::enable_shared_from_this<Function>
{
  public:
    virtual ~Function() {}

    VariablePtr operator()(const VariablePtr &input);

    VariablePtrList operator()(const VariablePtrList &inputs);

    virtual NdArrayPtrList Forward(const NdArrayPtrList &x) = 0;

    virtual NdArray Backward(const NdArray &gy) = 0;

  public:
    VariablePtrList inputs;  // 前向传播的入参，考虑多输入

    VariablePtrList outputs;  // 前向传播的输出，考虑多输出
};

class Square : public Function
{
  public:
    NdArrayPtrList Forward(const NdArrayPtrList &x) override;

    NdArray Backward(const NdArray &gy) override;
};

class Exp : public Function
{
  public:
    NdArrayPtrList Forward(const NdArrayPtrList &x) override;

    NdArray Backward(const NdArray &gy) override;
};

class Add : public Function
{
  public:
    NdArrayPtrList Forward(const NdArrayPtrList &x) override;

    NdArray Backward(const NdArray &gy) override;
};

extern NdArray NumericalDiff(std::function<Variable(Variable)> f, const Variable &x, data_t eps = 1e-4);

}  // namespace dl

#endif