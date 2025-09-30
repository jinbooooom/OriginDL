#ifndef __ORIGIN_DL_BACKEND_H__
#define __ORIGIN_DL_BACKEND_H__

#include <arrayfire.h>
#include "array_fire_mat.h"
#include "mat.h"

namespace origin
{

// 后端选择机制
#ifndef MAT_BACKEND
#    define MAT_BACKEND ARRAYFIRE
#endif

// 根据后端选择矩阵类型
#if MAT_BACKEND == ARRAYFIRE
using Mat_t = ArrayFireMat;
#elif MAT_BACKEND == EIGEN
using Mat_t = EigenMat;  // 未来扩展
#elif MAT_BACKEND == CUSTOM
using Mat_t = CustomMat;  // 未来扩展
#else
using Mat_t = ArrayFireMat;  // 默认使用ArrayFire
#endif

using NdArray = af::array;
using DLMat   = af::array;

}  // namespace origin

#endif
