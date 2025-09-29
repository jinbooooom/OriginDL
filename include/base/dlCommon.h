#ifndef __ORIGIN_DL_COMMON_H__
#define __ORIGIN_DL_COMMON_H__

#include <cmath>
#include <functional>
#include <list>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include <arrayfire.h>
#include "../mat/dlArrayFireMat.h"
#include "../mat/dlMat.h"
#include "../mat/dlShape.h"
#include "dlLog.h"
#include "dlTypes.h"

namespace dl
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

class Operator;
using NdArrayPtr     = std::shared_ptr<NdArray>;
using NdArrayPtrList = std::vector<NdArrayPtr>;
using NdArrayList    = std::vector<NdArray>;
using FunctionPtr    = std::shared_ptr<Operator>;

extern void print(const NdArray &data);

}  // namespace dl

#endif
