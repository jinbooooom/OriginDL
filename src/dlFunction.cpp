#include "dlFunction.h"

namespace dl
{

VariablePtr Function::operator()(const VariablePtr &input)
{
    auto x      = input->data;
    auto y      = this->Forward(x);
    auto output = std::make_shared<Variable>(y);
    output->SetCreator(shared_from_this());
    this->input  = input;
    this->output = output;
    return output;
}

NdArray Square::Forward(const NdArray &x)
{
    return af::pow(x, 2);
}

NdArray Square::Backward(const NdArray &gy)
{
    auto x  = this->input->data;
    auto gx = 2.0 * x * gy;
    return gx;
}

NdArray Exp::Forward(const NdArray &x)
{
    return af::exp(x);
}

NdArray Exp::Backward(const NdArray &gy)
{
    auto x  = this->input->data;
    auto gx = af::exp(x) * gy;
    return gx;
}

// 数值微分，求函数 f 在 x 处的导数
NdArray NumericalDiff(std::function<Variable(Variable)> f, const Variable &x, data_t eps)
{
    auto x0 = Variable(x.data - eps);
    auto x1 = Variable(x.data + eps);
    auto y0 = f(x0);
    auto y1 = f(x1);
    return (y1.data - y0.data) / (2 * eps);
}

}  // namespace dl
