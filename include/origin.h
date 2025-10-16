#ifndef __ORIGIN_DL_H__
#define __ORIGIN_DL_H__

// 包含所有必要的头文件，让用户只需要包含这一个文件
#include "origin/core/operator.h"
#include "origin/core/tensor.h"

// 如果启用了CUDA，暴露CUDA命名空间
#ifdef WITH_CUDA
#    include "origin/mat/origin/cuda/cuda_utils.cuh"
#endif

#endif  // __ORIGIN_DL_H__