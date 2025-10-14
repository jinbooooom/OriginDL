#ifndef __ORIGIN_DL_BACKEND_H__
#define __ORIGIN_DL_BACKEND_H__

/*
该文件会包含具体的矩阵计算后端头文件，在include的时候要分外注意。
*/

#include "mat.h"

// 支持多种后端选择
#ifdef MAT_BACKEND
#    if MAT_BACKEND == 0  // ARRAYFIRE
#        include <arrayfire.h>
#        include "array_fire_mat.h"
#    elif MAT_BACKEND == 1  // TORCH
#        include <torch/torch.h>
#        include "torch/torch_mat.h"
#    elif MAT_BACKEND == 2  // ORIGIN
#        include "origin/origin_mat.h"
#    elif MAT_BACKEND == 3       // EIGEN
#        include "eigen_mat.h"   // 未来扩展
#    elif MAT_BACKEND == 4       // CUSTOM
#        include "custom_mat.h"  // 未来扩展
#    endif
#else
// 默认使用TORCH后端
#    include <torch/torch.h>
#    include "torch/torch_mat.h"
#endif

namespace origin
{

// 根据后端选择对应的Mat类型
#ifdef MAT_BACKEND
#    if MAT_BACKEND == 0  // ARRAYFIRE
using Mat_t = ArrayFireMat;
#    elif MAT_BACKEND == 1  // TORCH
using Mat_t = TorchMat;
#    elif MAT_BACKEND == 2  // ORIGIN
using Mat_t = OriginMat;
#    elif MAT_BACKEND == 3  // EIGEN
using Mat_t = EigenMat;  // 未来扩展
#    elif MAT_BACKEND == 4  // CUSTOM
using Mat_t = CustomMat;  // 未来扩展
#    else
using Mat_t = TorchMat;  // 默认使用TorchMat
#    endif
#else
using Mat_t = TorchMat;  // 默认使用TorchMat
#endif

}  // namespace origin

#endif
