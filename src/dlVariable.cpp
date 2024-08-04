#include "dlVariable.h"

namespace dl
{
Variable::Variable(const NdArray &data) : data(data) {}

Variable::~Variable() {}

void Variable::Print(){
    std::cout << data << std::endl;
};
}  // namespace dl