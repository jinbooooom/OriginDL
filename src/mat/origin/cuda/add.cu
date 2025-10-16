#include <cuda_runtime.h>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/cuda/cuda_broadcast.cuh"
#include "origin/mat/origin/cuda/device_validation.cuh"
#include "origin/utils/exception.h"

namespace origin
{
namespace cuda
{

// 模板化的CUDA内核
template <typename T>
__global__ void add_kernel(const T *a, const T *b, T *c, size_t n)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

// 类型分发器 - 编译时特化
template <typename T>
void launch_add_kernel(const T *a, const T *b, T *c, size_t n, cudaStream_t stream = 0)
{
    dim3 block = get_optimal_block_size(n);
    dim3 grid  = get_optimal_grid_size(n, block);

    add_kernel<T><<<grid, block, 0, stream>>>(a, b, c, n);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("CUDA add kernel launch failed: {}", cudaGetErrorString(err));
    }
}

// 运行时类型分发
void dispatch_add(DataType dtype, const void *a, const void *b, void *c, size_t n, cudaStream_t stream = 0)
{
    switch (dtype)
    {
        case DataType::kFloat32:
            launch_add_kernel<float>(static_cast<const float *>(a), static_cast<const float *>(b),
                                     static_cast<float *>(c), n, stream);
            break;
        case DataType::kFloat64:
            launch_add_kernel<double>(static_cast<const double *>(a), static_cast<const double *>(b),
                                      static_cast<double *>(c), n, stream);
            break;
        case DataType::kInt32:
            launch_add_kernel<int32_t>(static_cast<const int32_t *>(a), static_cast<const int32_t *>(b),
                                       static_cast<int32_t *>(c), n, stream);
            break;
        case DataType::kInt8:
            launch_add_kernel<int8_t>(static_cast<const int8_t *>(a), static_cast<const int8_t *>(b),
                                      static_cast<int8_t *>(c), n, stream);
            break;
        default:
            THROW_INVALID_ARG("Unsupported data type for CUDA add operation: {}", dtype_to_string(dtype));
    }
}

// add算子实现
std::unique_ptr<Mat> add(const OriginMat &a, const OriginMat &b)
{
    // 验证输入 - 支持广播
    Shape result_shape = compute_broadcast_shape(a, b);

    if (a.dtype() != b.dtype())
    {
        THROW_INVALID_ARG("Data type mismatch in CUDA add: {} vs {}", dtype_to_string(a.dtype()),
                          dtype_to_string(b.dtype()));
    }

    // 使用高效的组合设备检查
    validation::validate_same_cuda_device(a, b, "add");

    // 创建结果张量
    std::unique_ptr<OriginMat> result(new OriginMat(result_shape, a.dtype(), a.device()));

    // 获取数据指针
    const void *a_data = a.storage()->data();
    const void *b_data = b.storage()->data();
    void *c_data       = result->storage()->data();

    // 启动CUDA内核
    dispatch_add(a.dtype(), a_data, b_data, c_data, a.elements());

    // 同步等待完成
    cudaDeviceSynchronize();

    return result;
}

}  // namespace cuda
}  // namespace origin
