#include "dlLoss.h"

namespace dl
{

VariablePtr MeanSquaredError(const VariablePtr &x0, const VariablePtr &x1)
{
    // auto diff = x0 - x1;
    // return F::sum(diff ^ 2) / diff->size();
}

#define mse MeanSquaredError;

}  // namespace dl
