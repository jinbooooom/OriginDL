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
        xs.push_back(as_dl_array_ptr(i->data_));
    }

    auto ys      = this->forward(xs);
    auto outputs = VariablePtrList();
    for (const auto &y : ys)
    {
        auto o = as_variable_ptr(y);
        o->set_creator(shared_from_this());
        outputs.push_back(o);
    }

    int max_gen = 0;
    for (auto &e : inputs)
    {
        if (e->generation_ > max_gen)
        {
            max_gen = e->generation_;
        }
    }
    this->generation_ = max_gen;

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
