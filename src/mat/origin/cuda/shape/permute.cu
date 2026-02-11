#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include "origin/mat/basic_types.h"
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
 * @brief CUDA permute kernel
 * @tparam T 数据类型
 */
template <typename T>
__global__ void permute_kernel(const T *__restrict__ input,
                               T *__restrict__ output,
                               const size_t *input_shape,
                               const size_t *input_strides,
                               const size_t *output_strides,
                               const int *dims,
                               size_t ndim,
                               size_t total_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements)
    {
        // 将输出索引转换为坐标
        size_t remaining = idx;
        size_t out_coords[8];  // 支持最多8维
        for (size_t d = 0; d < ndim; ++d)
        {
            out_coords[d] = remaining / output_strides[d];
            remaining %= output_strides[d];
        }

        // 将输出坐标映射回输入坐标
        size_t in_coords[8];
        for (size_t d = 0; d < ndim; ++d)
        {
            in_coords[dims[d]] = out_coords[d];
        }

        // 计算输入索引
        size_t in_idx = 0;
        for (size_t d = 0; d < ndim; ++d)
        {
            in_idx += in_coords[d] * input_strides[d];
        }

        output[idx] = input[in_idx];
    }
}

/**
 * @brief CUDA permute：按照指定顺序重新排列张量的维度
 * @param mat 输入矩阵
 * @param dims 新的维度顺序，例如 {0, 2, 3, 1} 表示将维度 0,1,2,3 重新排列为 0,2,3,1
 * @return 重排后的矩阵
 */
std::unique_ptr<Mat> permute(const OriginMat &mat, const std::vector<int> &dims)
{
    auto input_shape = mat.shape();
    size_t ndim      = input_shape.size();

    if (unlikely(dims.size() != ndim))
    {
        THROW_INVALID_ARG("permute: dims size {} does not match input dimension {}", dims.size(), ndim);
    }

    if (unlikely(ndim > 8))
    {
        THROW_INVALID_ARG("permute: supports up to 8 dimensions, but got {}", ndim);
    }

    // 验证 dims 的有效性
    std::vector<bool> used(ndim, false);
    for (int dim : dims)
    {
        if (unlikely(dim < 0 || dim >= static_cast<int>(ndim)))
        {
            THROW_INVALID_ARG("permute: invalid dimension {} (must be in [0, {}))", dim, ndim);
        }
        if (unlikely(used[dim]))
        {
            THROW_INVALID_ARG("permute: duplicate dimension {}", dim);
        }
        used[dim] = true;
    }

    // 计算输出形状
    std::vector<size_t> output_dims;
    output_dims.reserve(ndim);
    for (int dim : dims)
    {
        output_dims.push_back(input_shape[dim]);
    }
    Shape output_shape(output_dims);

    // 创建输出矩阵
    auto result = std::make_unique<OriginMat>(output_shape, mat.dtype(), mat.device());

    // 计算输入和输出的步长
    auto compute_strides = [](const Shape &shape) {
        std::vector<size_t> strides(shape.size());
        size_t stride = 1;
        for (int i = shape.size() - 1; i >= 0; --i)
        {
            strides[i] = stride;
            stride *= shape[i];
        }
        return strides;
    };

    auto input_strides  = compute_strides(input_shape);
    auto output_strides = compute_strides(output_shape);

    // 在 GPU 上分配临时内存存储形状和步长信息
    size_t shape_size  = ndim * sizeof(size_t);
    size_t stride_size = ndim * sizeof(size_t);
    size_t dims_size   = ndim * sizeof(int);

    size_t *d_input_shape    = nullptr;
    size_t *d_input_strides  = nullptr;
    size_t *d_output_strides = nullptr;
    int *d_dims              = nullptr;

    CUDA_CHECK(cudaMalloc(&d_input_shape, shape_size));
    CUDA_CHECK(cudaMalloc(&d_input_strides, stride_size));
    CUDA_CHECK(cudaMalloc(&d_output_strides, stride_size));
    CUDA_CHECK(cudaMalloc(&d_dims, dims_size));

    CUDA_CHECK(cudaMemcpy(d_input_shape, input_shape.dims().data(), shape_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_input_strides, input_strides.data(), stride_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_strides, output_strides.data(), stride_size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dims, dims.data(), dims_size, cudaMemcpyHostToDevice));

    // 计算线程块和网格大小
    const size_t threads_per_block = 256;
    size_t total_elements          = mat.elements();
    const size_t num_blocks        = (total_elements + threads_per_block - 1) / threads_per_block;

    // 使用类型分发器执行 permute 操作
    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        permute_kernel<T><<<num_blocks, threads_per_block>>>(mat.data_ptr<T>(), result->data_ptr<T>(), d_input_shape,
                                                             d_input_strides, d_output_strides, d_dims, ndim,
                                                             total_elements);
    });

    // 清理临时内存
    CUDA_CHECK(cudaFree(d_input_shape));
    CUDA_CHECK(cudaFree(d_input_strides));
    CUDA_CHECK(cudaFree(d_output_strides));
    CUDA_CHECK(cudaFree(d_dims));

    CUDA_CHECK_ASYNC();
    return result;
}

}  // namespace cuda
}  // namespace origin
