#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <memory>
#include <random>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cuda/cuda_ops.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cuda
{

/**
 * @brief CUDA dropout kernel（前向传播）
 * @details 每个线程处理一个元素
 * @param scale 缩放因子 scale = 1/(1-p)，在主机端计算后传入
 *              注意：不在核函数内计算 scale 的原因：
 *              1. scale 对所有线程是常量，避免每个线程重复计算除法
 *              2. 除法操作在 GPU 上相对昂贵，主机端计算一次更高效
 *              3. 标量参数通过常量内存/寄存器传递，开销很小
 */
template <typename T>
__global__ void dropout_kernel(const T *__restrict__ x,
                               T *__restrict__ y,
                               float *__restrict__ mask,
                               size_t n,
                               float p,
                               float scale,
                               unsigned long long seed)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        // 使用线程ID和全局索引生成随机数
        curandState state;
        curand_init(seed + idx, 0, 0, &state);
        float rand_val = curand_uniform(&state);

        if (rand_val < p)
        {
            mask[idx] = 0.0f;
            y[idx]    = T(0);
        }
        else
        {
            mask[idx] = scale;
            y[idx]    = static_cast<T>(static_cast<float>(x[idx]) * scale);
        }
    }
}

/**
 * @brief CUDA dropout_backward kernel（反向传播）
 * @details 每个线程处理一个元素
 */
template <typename T>
__global__ void dropout_backward_kernel(const T *__restrict__ gy,
                                        const float *__restrict__ mask,
                                        T *__restrict__ gx,
                                        size_t n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        gx[idx] = static_cast<T>(static_cast<float>(gy[idx]) * mask[idx]);
    }
}

/**
 * @brief CUDA dropout：Dropout 前向传播
 * @param x 输入张量
 * @param p dropout 概率
 * @param training 是否为训练模式
 * @param mask 输出参数：保存 dropout mask
 * @return 输出张量
 */
std::unique_ptr<Mat> dropout(const OriginMat &x, float p, bool training, OriginMat *mask)
{
    if (unlikely(p < 0.0f || p >= 1.0f))
    {
        THROW_INVALID_ARG("Dropout: p must be in [0, 1), but got {}", p);
    };

    auto x_shape = x.shape();
    auto result  = std::make_unique<OriginMat>(x_shape, x.dtype(), x.device());

    const void *x_data = x.storage()->data();
    void *y_data       = result->storage()->data();
    if (!training)
    {
        // 推理模式：直接返回输入
        size_t data_size = x_shape.elements() * element_size(x.dtype());
        CUDA_CHECK(cudaMemcpyAsync(y_data, x_data, data_size, cudaMemcpyDeviceToDevice));
        return result;
    }

    // 训练模式：生成 dropout mask
    OriginMat *mask_ptr = nullptr;
    std::unique_ptr<OriginMat> mask_unique;
    if (mask != nullptr)
    {
        mask_ptr = mask;
    }
    else
    {
        mask_unique = std::make_unique<OriginMat>(x_shape, DataType::kFloat32, x.device());
        mask_ptr    = mask_unique.get();
    }

    void *mask_data = mask_ptr->storage()->data();

    // 生成随机种子
    std::random_device rd;
    std::mt19937 gen(rd());
    unsigned long long seed = gen();

    float scale = 1.0f / (1.0f - p);

    // 计算线程块和网格大小
    const size_t threads_per_block = 256;
    const size_t num_elements      = x_shape.elements();
    const size_t num_blocks        = (num_elements + threads_per_block - 1) / threads_per_block;

    // 使用类型分发器执行 dropout 操作
    device_common::TypeDispatcher::dispatch_void(x.dtype(), [&]<typename T>() {
        dropout_kernel<T><<<num_blocks, threads_per_block>>>(static_cast<const T *>(x_data), static_cast<T *>(y_data),
                                                             static_cast<float *>(mask_data), num_elements, p, scale,
                                                             seed);
    });

    return result;
}

/**
 * @brief CUDA dropout_backward：Dropout 反向传播
 * @param gy 输出梯度
 * @param mask dropout mask
 * @return 输入梯度
 */
std::unique_ptr<Mat> dropout_backward(const OriginMat &gy, const OriginMat &mask)
{
    if (unlikely(gy.shape() != mask.shape()))
    {
        THROW_INVALID_ARG("Dropout backward: gradient shape {} must match mask shape {}", gy.shape().to_string(),
                          mask.shape().to_string());
    }

    VALIDATE_SAME_CUDA_DEVICE(gy, mask);

    auto gy_shape = gy.shape();
    auto result   = std::make_unique<OriginMat>(gy_shape, gy.dtype(), gy.device());

    const void *gy_data   = gy.storage()->data();
    const void *mask_data = mask.storage()->data();
    void *gx_data         = result->storage()->data();

    // 计算线程块和网格大小
    const size_t threads_per_block = 256;
    const size_t num_elements      = gy_shape.elements();
    const size_t num_blocks        = (num_elements + threads_per_block - 1) / threads_per_block;

    // 使用类型分发器执行反向传播操作
    device_common::TypeDispatcher::dispatch_void(gy.dtype(), [&]<typename T>() {
        dropout_backward_kernel<T><<<num_blocks, threads_per_block>>>(static_cast<const T *>(gy_data),
                                                                      static_cast<const float *>(mask_data),
                                                                      static_cast<T *>(gx_data), num_elements);
    });

    return result;
}

}  // namespace cuda
}  // namespace origin
