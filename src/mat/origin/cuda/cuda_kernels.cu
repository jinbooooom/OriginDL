#include "origin/mat/origin/cuda/cuda_kernels.cuh"

#ifdef __CUDACC__
namespace origin {
namespace cuda {

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

} // namespace cuda
} // namespace origin
#endif // __CUDACC__
