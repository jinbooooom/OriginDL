#ifndef __ORIGIN_DL_H__
#define __ORIGIN_DL_H__

// ============================================================================
// Core 核心模块
// ============================================================================
#include "origin/core/config.h"
#include "origin/core/operator.h"
#include "origin/core/parameter.h"
#include "origin/core/tensor.h"
#include "origin/mat/shape.h"

// ============================================================================
// Neural Network 神经网络模块
// ============================================================================
#include "origin/nn/layer.h"
#include "origin/nn/module.h"
#include "origin/nn/sequential.h"

// 常用层
#include "origin/nn/layers/batch_norm2d.h"
#include "origin/nn/layers/conv2d.h"
#include "origin/nn/layers/flatten.h"
#include "origin/nn/layers/linear.h"
#include "origin/nn/layers/max_pool2d.h"
#include "origin/nn/layers/relu.h"

// ============================================================================
// Optimizer 优化器
// ============================================================================
#include "origin/optim/adam.h"
#include "origin/optim/hooks.h"
#include "origin/optim/optimizer.h"
#include "origin/optim/sgd.h"

// ============================================================================
// Data 数据模块
// ============================================================================
#include "origin/data/dataloader.h"
#include "origin/data/dataset.h"
#include "origin/data/mnist.h"

// ============================================================================
// 模型保存加载模块
// ============================================================================
#include "origin/io/checkpoint.h"
#include "origin/io/model_io.h"

// ============================================================================
// PNNX 模型推理模块
// ============================================================================
#include "origin/pnnx/pnnx_graph.h"

// ============================================================================
// Utils 工具模块
// ============================================================================
#include "origin/utils/log.h"
#include "origin/utils/metrics.h"

// ============================================================================
// CUDA 支持
// ============================================================================
#include "origin/cuda/cuda.h"

#endif  // __ORIGIN_DL_H__