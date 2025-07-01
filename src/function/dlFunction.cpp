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

}  // namespace dl
