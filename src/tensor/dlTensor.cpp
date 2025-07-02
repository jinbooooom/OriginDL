#include "dlTensor.h"

namespace dl
{
Variable::Variable(const NdArray &data) : data(data)
{
    double grad_val = 1.0;
    grad            = std::make_shared<NdArray>(af::constant(grad_val, {1}));
}

Variable::~Variable() {}

void Variable::SetCreator(const FunctionPtr &func)
{
    creator = func;
}

void Variable::Backward()
{
    // auto funcs = std::vector<FunctionPtr>({this->creator});
    // while (!funcs.empty())
    // {
    //     auto f = funcs.back();
    //     funcs.pop_back();
    //     auto x  = f->inputs;
    //     auto y  = f->outputs;
    //     x->grad = std::make_shared<NdArray>(f->Backward(*y->grad));

    //     if (x->creator != nullptr)
    //     {
    //         funcs.push_back(x->creator);
    //     }
    // }

    return;
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

//  NdArrayPtrList AsNdArrayPtrList(VariablePtr data)
//  {
//     NdArrayPtrList l;
//     l.push_back(data);
//     return l;
//  }

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
