#ifndef __ORIGIN_DL_H__
#define __ORIGIN_DL_H__

// 包含所有必要的头文件，让用户只需要包含这一个文件
#include "origin/core/operator.h"
#include "origin/core/parameter.h"
#include "origin/core/tensor.h"

// 神经网络模块
#include "origin/nn/layer.h"
#include "origin/nn/layers/linear.h"
#include "origin/nn/module.h"
#include "origin/nn/sequential.h"

// 优化器
#include "origin/optim/optimizer.h"
#include "origin/optim/sgd.h"

// 如果启用了CUDA，暴露CUDA命名空间
#ifdef WITH_CUDA
#    include "origin/cuda/cuda.h"
#endif

#endif  // __ORIGIN_DL_H__