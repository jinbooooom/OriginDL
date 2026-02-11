#ifndef __ORIGIN_DL_BACKEND_H__
#define __ORIGIN_DL_BACKEND_H__

/*
该文件会包含具体的矩阵计算后端头文件，在include的时候要分外注意。
*/

#include "mat.h"

// 支持多种后端选择
#ifdef MAT_BACKEND
#    if MAT_BACKEND == 0  // ORIGIN（默认）
#        include "origin/origin_mat.h"
#    elif MAT_BACKEND == 1  // TORCH
#        include <torch/torch.h>
#        include "torch/torch_mat.h"
#    endif
#else
// 默认使用ORIGIN后端
#    include "origin/origin_mat.h"
#endif

namespace origin
{

// 根据后端选择对应的Mat类型
#ifdef MAT_BACKEND
#    if MAT_BACKEND == 0  // ORIGIN（默认）
using Mat_t = OriginMat;
#    elif MAT_BACKEND == 1  // TORCH
using Mat_t = TorchMat;
#    else
using Mat_t = OriginMat;  // 默认使用OriginMat
#    endif
#else
using Mat_t = OriginMat;  // 默认使用OriginMat
#endif

}  // namespace origin

#endif
