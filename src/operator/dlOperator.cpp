#include "dlOperator.h"

namespace dl
{

VariablePtrList Operator::operator()(const VariablePtr &input)
{
    auto outputs = (*this)(VariablePtrList({input}));
    return outputs;
}

VariablePtrList Operator::operator()(const VariablePtrList &inputs)
{
    auto xs = NdArrayPtrList();
    for (const auto &i : inputs)
    {
        xs.push_back(AsDLArrayPtr(i->data_));
    }

    auto ys      = this->Forward(xs);
    auto outputs = VariablePtrList();
    for (const auto &y : ys)
    {
        auto o = AsVariablePtr(y);
        o->SetCreator(shared_from_this());
        outputs.push_back(o);
    }

    int maxGen = 0;
    for (auto &e : inputs)
    {
        if (e->generation_ > maxGen)
        {
            maxGen = e->generation_;
        }
    }
    this->generation_ = maxGen;

    this->inputs_ = inputs;
    // this->outputs_ = std::move(outputs);
    this->outputs_.clear();
    for (const auto &o : outputs)
    {
        VariableWPtr w = o;
        this->outputs_.push_back(w);
    }

    return outputs;
}

}  // namespace dl
