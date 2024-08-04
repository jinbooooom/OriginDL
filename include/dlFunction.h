#ifndef __DLZERO_FUNCTION_H__
#define __DLZERO_FUNCTION_H__

#include "dlVariable.h"

namespace dl
{

class Function
{
  public:
    virtual ~Function() {}

    Variable operator()(const Variable &input);

    virtual NdArray Forward(const NdArray &x) = 0;

	virtual NdArray Backward(const NdArray& gy) = 0;

	protected:
		std::shared_ptr<Variable> input; // 前向传播的入参

};

class Square : public Function
{
  public:
    NdArray Forward(const NdArray &x) override;

	NdArray Backward(const NdArray& gy) override;
};

class Exp : public Function
{
  public:
    NdArray Forward(const NdArray &x) override;

	NdArray Backward(const NdArray& gy) override;
};

extern NdArray NumericalDiff(std::function<Variable(Variable)> f, const Variable &x, data_t eps = 1e-4);

}  // namespace dl

#endif