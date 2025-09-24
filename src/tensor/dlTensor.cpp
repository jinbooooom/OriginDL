#include "base/dlException.h"
#include "originDL.h"

namespace dl
{
Variable::Variable(const NdArray &data) : data_(data), generation_(0) {}

Variable::~Variable() {}

void Variable::set_creator(const FunctionPtr &func)
{
    creator_    = func;
    generation_ = creator_->generation_ + 1;
}

void Variable::backward()
{
    if (!this->grad_)
    {
        double grad_val = 1.0;
        auto dims       = this->data_.dims();
        grad_           = std::make_shared<NdArray>(af::constant(grad_val, dims));
    }

    auto funcs    = std::list<FunctionPtr>();
    auto func_set = std::set<FunctionPtr>();  // 考虑到多输出的情况下

    auto add_func = [&funcs, &func_set](const FunctionPtr &f) {
        if (func_set.find(f) == func_set.end())
        {
            funcs.push_back(f);
            func_set.insert(f);
            funcs.sort(
                [](const FunctionPtr &lhs, const FunctionPtr &rhs) { return lhs->generation_ < rhs->generation_; });
        }
    };

    add_func(this->creator_);

    while (!funcs.empty())
    {
        auto f = funcs.back();
        funcs.pop_back();

        auto gys = NdArrayPtrList();
        for (const auto &o : f->outputs_)
        {
            // 通过 lock() 升级为 shared_ptr 并检查有效性
            if (auto o_ptr = o.lock())
            {
                gys.emplace_back(o_ptr->grad_);
            }
            else
            {
                DL_CRITICAL_THROW("backward error!, output is nullptr");
            }
        }
        auto gxs = f->backward(gys);

        if (gxs.size() != f->inputs_.size())
        {
            DL_ERROR_THROW("backward error!, gxs size " + std::to_string(gxs.size()) + ", inputs size " +
                           std::to_string(f->inputs_.size()));
        }

        for (size_t i = 0; i < gxs.size(); i++)
        {
            auto x  = f->inputs_[i];
            auto gx = gxs[i];

            if (!x->grad_)
            {
                x->grad_ = gx;
            }
            else
            {
                x->grad_ = as_dl_array_ptr(*(x->grad_) + (*gx));
            }

            if (x->creator_)
            {
                add_func(x->creator_);
            }
        }
    }

    return;
}

void Variable::clear_grad()
{
    grad_ = nullptr;
}

VariablePtr Variable::reshape(const af::dim4 shape)
{
    auto p = as_variable_ptr(*this);
    return ::dl::reshape(p, shape);
}

VariablePtr Variable::transpose()
{
    auto p = as_variable_ptr(*this);
    return ::dl::transpose(p);
};

void Variable::print(std::string desc)
{
    af::print(desc.c_str(), data_);
};

// 变量转换，未来考虑去掉
VariablePtrList as_variable_ptr_list(VariablePtr data)
{
    VariablePtrList l;
    l.push_back(data);
    return l;
}

NdArrayPtrList as_dl_array_ptr_list(NdArray data)
{
    NdArrayPtrList l;
    l.push_back(as_dl_array_ptr(data));
    return l;
}

NdArrayPtr as_dl_array_ptr(NdArray data)
{
    return std::make_shared<NdArray>(data);
}

VariablePtr as_variable_ptr(NdArrayPtr data)
{
    return std::make_shared<Variable>(*data);
}

VariablePtr as_variable_ptr(Variable &data)
{
    return std::make_shared<Variable>(data);
}

}  // namespace dl
