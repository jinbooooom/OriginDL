#ifndef __ORIGIN_DL_BACKEND_H__
#define __ORIGIN_DL_BACKEND_H__

#include "backend_constants.h"
#include "mat.h"

// 硬编码使用TORCH后端
// #if MAT_BACKEND == ARRAYFIRE
// #include <arrayfire.h>
// #include "array_fire_mat.h"
// #elif MAT_BACKEND == TORCH
#include <torch/torch.h>
#include "torch/torch_mat.h"
// #elif MAT_BACKEND == EIGEN
// #include "eigen_mat.h"  // 未来扩展
// #elif MAT_BACKEND == CUSTOM
// #include "custom_mat.h"  // 未来扩展
// #endif

namespace origin
{

// 硬编码使用TORCH后端
// #ifndef MAT_BACKEND
// #    define MAT_BACKEND ARRAYFIRE
// #endif

// 硬编码使用TorchMat
// #if MAT_BACKEND == ARRAYFIRE
// using Mat_t = ArrayFireMat;
// #elif MAT_BACKEND == TORCH
using Mat_t = TorchMat;
// #elif MAT_BACKEND == EIGEN
// using Mat_t = EigenMat;  // 未来扩展
// #elif MAT_BACKEND == CUSTOM
// using Mat_t = CustomMat;  // 未来扩展
// #else
// using Mat_t = TorchMat;  // 默认使用TorchMat
// #endif

// 硬编码使用torch::Tensor类型别名
// #if MAT_BACKEND == ARRAYFIRE
// using NdArray = af::array;
// using DLMat   = af::array;
// #elif MAT_BACKEND == TORCH
using NdArray = torch::Tensor;
using DLMat   = torch::Tensor;
// #else
// // 默认使用TorchMat
// using NdArray = torch::Tensor;
// using DLMat   = torch::Tensor;
// #endif

}  // namespace origin

#endif
