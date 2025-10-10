#ifndef __ORIGIN_DL_BACKEND_MAT_UTILS_H__
#define __ORIGIN_DL_BACKEND_MAT_UTILS_H__

#include "types.h"

namespace origin
{
namespace utils
{
// 临时注释掉ArrayFire相关函数
// #if MAT_BACKEND == ARRAYFIRE
// af::array BroadcastTo(const af::array &src, const af::dim4 &targetShape);
// af::array SumTo(const af::array &src, const af::dim4 &targetShape);
// #endif
}  // namespace utils
}  // namespace origin

#endif
