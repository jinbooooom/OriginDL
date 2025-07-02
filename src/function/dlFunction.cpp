#include "dlFunction.h"

namespace dl
{

VariablePtr Function::operator()(const VariablePtr &input)
{
    auto x  = input->data;
    auto xs = NdArrayPtrList();
    xs.push_back(AsDLArrayPtr(x));
    auto y      = this->Forward(xs);
    auto output = std::make_shared<Variable>(*y[0]);
    output->SetCreator(shared_from_this());
    this->inputs  = AsVariablePtrList(input);
    this->outputs = AsVariablePtrList(output);
    return output;
}

VariablePtrList Function::operator()(const VariablePtrList &inputs)
{
    auto xs = NdArrayPtrList();
    for (const auto &i : inputs)
    {
        xs.push_back(AsDLArrayPtr(i->data));
    }

    auto ys      = this->Forward(xs);
    auto outputs = VariablePtrList();
    for (const auto &y : ys)
    {
        auto o = AsVariablePtr(y);
        o->SetCreator(shared_from_this());
        outputs.push_back(o);
    }

    this->inputs  = inputs;
    this->outputs = std::move(outputs);
    return this->outputs;
}

}  // namespace dl
