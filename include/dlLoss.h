#ifndef __ORIGIN_DL_LOSS_H__
#define __ORIGIN_DL_LOSS_H__

#include "dlOperator.h"

namespace dl
{

// 均方误差
VariablePtr MeanSquaredError(const VariablePtr &x0, const VariablePtr &x1);

}  // namespace dl

#endif
