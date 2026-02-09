#include "origin/mat/origin/cuda/cuda_kernels.cuh"

#ifdef __CUDACC__
namespace origin
{
namespace cuda
{

/**
 * @brief 启动索引写入内核（单个元素）实现
 * @note 此函数在 .cu 文件中实现，用 nvcc 编译，避免在头文件中使用丑陋的宏
 */
template <typename T>
__host__ void launch_index_put_kernel(T *data, size_t index, T value, cudaStream_t stream)
{
    // 使用1个block，1个thread来写入单个元素
    index_put_kernel<T><<<1, 1, 0, stream>>>(data, index, value);
}

// 显式实例化 launch_index_put_kernel 的常用类型，确保链接器能找到这些符号
// 这些函数在 .cpp 文件中被调用，但实现在 .cu 文件中用 nvcc 编译
template void launch_index_put_kernel<float>(float *, size_t, float, cudaStream_t);
template void launch_index_put_kernel<double>(double *, size_t, double, cudaStream_t);
template void launch_index_put_kernel<int8_t>(int8_t *, size_t, int8_t, cudaStream_t);
template void launch_index_put_kernel<int32_t>(int32_t *, size_t, int32_t, cudaStream_t);
template void launch_index_put_kernel<int64_t>(int64_t *, size_t, int64_t, cudaStream_t);
template void launch_index_put_kernel<uint8_t>(uint8_t *, size_t, uint8_t, cudaStream_t);

/**
 * @brief 启动 clone kernel：按逻辑顺序拷贝非连续张量实现
 * @note 此函数在 .cu 文件中实现，用 nvcc 编译，避免在头文件中使用丑陋的宏
 */
template <typename T>
__host__ void launch_clone_kernel(const T *src,
                                   T *dst,
                                   const size_t *shape,
                                   const size_t *src_strides,
                                   const size_t *output_strides,
                                   size_t ndim,
                                   size_t total_elements,
                                   cudaStream_t stream)
{
    dim3 block = get_optimal_block_size(total_elements);
    dim3 grid  = get_optimal_grid_size(total_elements, block);

    clone_kernel<T><<<grid, block, 0, stream>>>(src, dst, shape, src_strides, output_strides, ndim, total_elements);
}

// 显式实例化 launch_clone_kernel 的常用类型，确保链接器能找到这些符号
// 这些函数在 .cpp 文件中被调用，但实现在 .cu 文件中用 nvcc 编译
template void launch_clone_kernel<float>(const float *, float *, const size_t *, const size_t *, const size_t *,
                                         size_t, size_t, cudaStream_t);
template void launch_clone_kernel<double>(const double *, double *, const size_t *, const size_t *, const size_t *,
                                          size_t, size_t, cudaStream_t);
template void launch_clone_kernel<int8_t>(const int8_t *, int8_t *, const size_t *, const size_t *, const size_t *,
                                          size_t, size_t, cudaStream_t);
template void launch_clone_kernel<int32_t>(const int32_t *, int32_t *, const size_t *, const size_t *, const size_t *,
                                           size_t, size_t, cudaStream_t);
template void launch_clone_kernel<uint8_t>(const uint8_t *, uint8_t *, const size_t *, const size_t *, const size_t *,
                                            size_t, size_t, cudaStream_t);
template void launch_clone_kernel<long>(const long *, long *, const size_t *, const size_t *, const size_t *,
                                        size_t, size_t, cudaStream_t);
template void launch_clone_kernel<unsigned long>(const unsigned long *, unsigned long *, const size_t *,
                                                  const size_t *, const size_t *, size_t, size_t, cudaStream_t);

}  // namespace cuda
}  // namespace origin
#endif  // __CUDACC__
