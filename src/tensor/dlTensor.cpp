#include "dlTensor.h"

namespace dl
{
Variable::Variable(const NdArray &data) : data(data)
{
    // double grad_val = 1.0;
    // grad            = std::make_shared<NdArray>(af::constant(grad_val, {1}));
}

Variable::~Variable() {}

void Variable::SetCreator(const FunctionPtr &func)
{
    creator = func;
}

void Variable::Backward()
{
    if (!this->grad)
    {
        double grad_val = 1.0;
        auto dims       = this->data.dims();
        grad            = std::make_shared<NdArray>(af::constant(grad_val, dims));
    }

    auto funcs = std::vector<FunctionPtr>({this->creator});
    while (!funcs.empty())
    {
        auto f = funcs.back();
        funcs.pop_back();

        auto gys = NdArrayPtrList();
        for (const auto &o : f->outputs)
        {
            gys.emplace_back(o->grad);
        }
        auto gxs = f->Backward(gys);

        if (gxs.size() != f->inputs.size())
        {
            loge("backward error!, gxs size {}, inputs size {}", gxs.size(), f->inputs.size());
            exit(1);
        }

        for (size_t i = 0; i < gxs.size(); i++)
        {
            auto x  = f->inputs[i];
            auto gx = gxs[i];

            x->grad = gx;
            if (x->creator)
            {
                funcs.push_back(x->creator);
            }
        }
    }

    return;
}

void Variable::ClearGrad()
{
    grad = nullptr;
}

void Variable::Print()
{
    af::print("", data);
};

// 变量转换，未来考虑去掉
VariablePtrList AsVariablePtrList(VariablePtr data)
{
    VariablePtrList l;
    l.push_back(data);
    return l;
}

NdArrayPtrList AsDLArrayPtrList(NdArray data)
{
    NdArrayPtrList l;
    l.push_back(AsDLArrayPtr(data));
    return l;
}

NdArrayPtr AsDLArrayPtr(NdArray data)
{
    return std::make_shared<NdArray>(data);
}

VariablePtr AsVariablePtr(NdArrayPtr data)
{
    return std::make_shared<Variable>(*data);
}

VariablePtr AsVariablePtr(Variable &data)
{
    return std::make_shared<Variable>(data);
}

}  // namespace dl
