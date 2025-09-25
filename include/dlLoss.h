#ifndef __ORIGIN_DL_LOSS_H__
#define __ORIGIN_DL_LOSS_H__

#include "dlOperator.h"

namespace dl
{

// 均方误差
Tensor MeanSquaredError(const Tensor &x0, const Tensor &x1);

float MeanSquaredError(const DLMat &pred, const DLMat &real);

}  // namespace dl

#endif
