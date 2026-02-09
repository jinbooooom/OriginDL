/**
 * @file cuda_ops.cu
 * @brief CUDA 非计算类算子实现
 * 
 * ============================================================================
 * 文件功能说明
 * ============================================================================
 * 
 * 本文件承担非计算类 CUDA 算子的实现，类似于 add.cu 但按功能分类而非按算子分类。
 * 
 * 架构位置：
 * - origin_mat.cpp (封装层)
 *   ↓ 包含
 * - cuda_ops.cuh (所有 CUDA 算子的接口声明)
 *   ↓ 声明
 * - cuda_ops.cu (本文件：非计算类算子实现：clone、index_put)
 * - add.cu, divide.cu 等 (计算类算子实现)
 *   ↓ 都包含
 * - cuda_kernels.cuh (kernel 定义，只在 .cu 文件中使用)
 * 
 * ============================================================================
 */

#include "origin/mat/origin/cuda/cuda_ops.cuh"
#include "origin/mat/origin/cuda/cuda_kernels.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/origin_mat_utils.h"
#include "origin/utils/exception.h"

#ifdef __CUDACC__
namespace origin
{
namespace cuda
{

// ============================================================================
// Kernel 定义（仅供本文件内部使用）
// ============================================================================

/**
 * @brief 索引写入内核（单个元素）
 * @tparam T 数据类型
 * @param data 数据指针
 * @param index 线性索引
 * @param value 要写入的值
 */
template <typename T>
__global__ void index_put_kernel(T *data, size_t index, T value)
{
    // 只写入指定索引位置的值
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        data[index] = value;
    }
}

/**
 * @brief CUDA clone kernel：按逻辑顺序拷贝非连续张量
 * @tparam T 数据类型
 * @param src 源数据指针 
 * @param dst 目标数据指针（连续存储）
 * @param shape 源张量的形状（与 src_strides 对应）
 * @param src_strides 源张量的 strides（可能非连续）
 * @param output_strides 输出张量的连续 strides（用于计算逻辑索引，对应源张量的 shape）
 * @param ndim 维度数
 * @param total_elements 元素总数
 *
 * ============================================================================
 * 算法原理
 * ============================================================================
 *
 * 该 kernel 的核心目标是将非连续张量按逻辑顺序拷贝到连续内存中。
 * 关键思想是：将输出位置的线性索引转换为源张量的物理内存偏移。
 *
 * 算法分为三个步骤：
 *
 * 1. 线性索引 -> 多维坐标
 *    使用输出张量的连续 strides，将线性索引 idx 转换为多维坐标 (i, j, k, ...)
 *    这些坐标对应源张量的逻辑位置
 *
 * 2. 多维坐标 -> 源张量物理偏移
 *    使用相同的坐标和源张量的 strides 计算源张量中的物理内存偏移
 *
 * 3. 拷贝元素
 *    从源张量的物理位置读取，写入到目标张量的连续位置
 *
 * ============================================================================
 * 步骤1详解：线性索引 -> 多维坐标
 * ============================================================================
 *
 * 这是算法的核心步骤，需要理解 strides 的含义和计算方法。
 *
 * 1.1 Strides 的含义
 *    对于形状为 [d0, d1, d2, ..., d(n-1)] 的张量，连续 strides 的计算方式：
 *        strides[n-1] = 1
 *        strides[i] = strides[i+1] * d(i+1)  (从后往前计算)
 *
 *    例如：shape = [3, 2]
 *        strides[1] = 1
 *        strides[0] = strides[1] * 2 = 1 * 2 = 2
 *        所以 output_strides = [2, 1]
 *
 * 1.2 线性索引到多维坐标的转换原理
 *    对于连续存储的张量，线性索引 idx 与多维坐标 (i0, i1, i2, ...) 的关系：
 *        idx = i0 * strides[0] + i1 * strides[1] + i2 * strides[2] + ...
 *
 *    要从 idx 反推出坐标，需要"反向"计算：
 *        i0 = idx / strides[0]           (整数除法)
 *        remainder = idx % strides[0]    (余数)
 *        i1 = remainder / strides[1]
 *        remainder = remainder % strides[1]
 *        i2 = remainder / strides[2]
 *        ...
 *
 * 1.3 算法实现
 *    代码中的实现：
 *        remaining = idx
 *        for d = 0 to ndim-1:
 *            coords[d] = remaining / output_strides[d]
 *            remaining = remaining % output_strides[d]
 *
 * 1.4 具体示例（shape = [3, 2], output_strides = [2, 1]）
 *
 *    idx = 0:
 *        d=0: coords[0] = 0 / 2 = 0, remaining = 0 % 2 = 0
 *        d=1: coords[1] = 0 / 1 = 0, remaining = 0 % 1 = 0
 *        结果: coords = [0, 0] -> 位置 (0, 0)
 *
 *    idx = 1:
 *        d=0: coords[0] = 1 / 2 = 0, remaining = 1 % 2 = 1
 *        d=1: coords[1] = 1 / 1 = 1, remaining = 1 % 1 = 0
 *        结果: coords = [0, 1] -> 位置 (0, 1)
 *
 *    idx = 2:
 *        d=0: coords[0] = 2 / 2 = 1, remaining = 2 % 2 = 0
 *        d=1: coords[1] = 0 / 1 = 0, remaining = 0 % 1 = 0
 *        结果: coords = [1, 0] -> 位置 (1, 0)
 *
 *    idx = 3:
 *        d=0: coords[0] = 3 / 2 = 1, remaining = 3 % 2 = 1
 *        d=1: coords[1] = 1 / 1 = 1, remaining = 1 % 1 = 0
 *        结果: coords = [1, 1] -> 位置 (1, 1)
 *
 *    idx = 4:
 *        d=0: coords[0] = 4 / 2 = 2, remaining = 4 % 2 = 0
 *        d=1: coords[1] = 0 / 1 = 0, remaining = 0 % 1 = 0
 *        结果: coords = [2, 0] -> 位置 (2, 0)
 *
 *    idx = 5:
 *        d=0: coords[0] = 5 / 2 = 2, remaining = 5 % 2 = 1
 *        d=1: coords[1] = 1 / 1 = 1, remaining = 1 % 1 = 0
 *        结果: coords = [2, 1] -> 位置 (2, 1)
 *
 * 1.5 为什么使用 output_strides 而不是 shape？
 *    因为 strides 包含了维度大小的信息，可以直接用于计算坐标。
 *    使用 strides 的好处是：
 *    - 直接反映了内存布局
 *    - 计算效率高（只需要除法和取模）
 *    - 适用于任意维度的张量
 *
 * 1.6 三维示例（shape = [2, 2, 2], output_strides = [4, 2, 1]）
 *
 *    idx = 0:  coords = [0, 0, 0] -> (0, 0, 0)
 *    idx = 1:  coords = [0, 0, 1] -> (0, 0, 1)
 *    idx = 2:  coords = [0, 1, 0] -> (0, 1, 0)
 *    idx = 3:  coords = [0, 1, 1] -> (0, 1, 1)
 *    idx = 4:  coords = [1, 0, 0] -> (1, 0, 0)
 *    idx = 5:  coords = [1, 0, 1] -> (1, 0, 1)
 *    idx = 6:  coords = [1, 1, 0] -> (1, 1, 0)
 *    idx = 7:  coords = [1, 1, 1] -> (1, 1, 1)
 *
 *    验证：idx = 5 = 1*4 + 0*2 + 1*1 = 4 + 0 + 1 = 5 
 *
 * ============================================================================
 * 完整示例：转置后 reshape 的场景
 * ============================================================================
 *
 * 假设有一个 2×3 的张量经过转置，然后需要 reshape：
 *
 * 步骤1：原始张量
 *   原始数据（内存顺序）: [1, 2, 3, 4, 5, 6]
 *   原始形状: [2, 3]
 *   原始逻辑顺序: [[1, 2, 3], [4, 5, 6]]
 *
 * 步骤2：转置操作
 *   转置后形状: [3, 2]  <- 这是源张量的 shape（传入 kernel）
 *   转置后逻辑顺序: [[1, 4], [2, 5], [3, 6]]
 *   转置后内存顺序: [1, 2, 3, 4, 5, 6] (未变，但strides变了)
 *   转置后strides: [1, 3]  <- 这是 src_strides（非连续）
 *
 * 步骤3：clone_kernel 拷贝过程（将非连续张量转为连续）
 *   输出形状: [3, 2]  <- 与源张量相同（因为 contiguous 保持形状）
 *   output_strides: [2, 1]  <- 连续 strides
 *
 *   拷贝过程：
 *
 *   | idx | coords | src_offset计算 | 逻辑值 | 物理位置 | 拷贝到 dst[idx] |
 *   |-----|--------|----------------|--------|----------|----------------|
 *   | 0   | (0,0)  | 0*1+0*3=0      | 1      | src[0]   | dst[0] = 1     |
 *   | 1   | (0,1)  | 0*1+1*3=3      | 4      | src[3]   | dst[1] = 4     |
 *   | 2   | (1,0)  | 1*1+0*3=1      | 2      | src[1]   | dst[2] = 2     |
 *   | 3   | (1,1)  | 1*1+1*3=4      | 5      | src[4]   | dst[3] = 5     |
 *   | 4   | (2,0)  | 2*1+0*3=2      | 3      | src[2]   | dst[4] = 3     |
 *   | 5   | (2,1)  | 2*1+1*3=5      | 6      | src[5]   | dst[5] = 6     |
 *
 *   结果：dst = [1, 4, 2, 5, 3, 6]（按逻辑顺序，连续存储）
 *
 * 步骤4：reshape 操作（在 contiguous 之后）
 *   对连续副本进行 reshape，例如 reshape 为 [6]
 *   结果: [1, 4, 2, 5, 3, 6]（零拷贝，只是改变视图）
 *
 */
template <typename T>
__global__ void clone_kernel(const T *__restrict__ src,
                             T *__restrict__ dst,
                             const size_t *shape,
                             const size_t *src_strides,
                             const size_t *output_strides,
                             size_t ndim,
                             size_t total_elements)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < total_elements)
    {
        // 步骤1：将线性索引转换为多维坐标（按输出形状）
        // 使用输出张量的连续 strides，将线性索引 idx 转换为多维坐标
        size_t coords[8];  // 支持最多8维
        size_t remaining = idx;
        for (size_t d = 0; d < ndim; ++d)
        {
            coords[d] = remaining / output_strides[d];
            remaining %= output_strides[d];
        }

        // 步骤2：计算源张量的物理偏移（使用实际的 strides）
        // 使用相同的坐标和源张量的 strides 计算源张量中的物理内存偏移
        size_t src_offset = 0;
        for (size_t d = 0; d < ndim; ++d)
        {
            src_offset += coords[d] * src_strides[d];
        }

        // 步骤3：拷贝元素（目标位置是连续的，所以直接使用 idx）
        // 从源张量的物理位置读取，写入到目标张量的连续位置
        dst[idx] = src[src_offset];
    }
}

// ============================================================================
// 内部实现：kernel 启动函数（仅供本文件内部使用）
// ============================================================================

/**
 * @brief 启动索引写入内核（单个元素）实现
 * @note 此函数在 .cu 文件中实现，用 nvcc 编译，仅供本文件内部使用
 */
template <typename T>
__host__ void launch_index_put_kernel(T *data, size_t index, T value, cudaStream_t stream)
{
    // 使用1个block，1个thread来写入单个元素
    index_put_kernel<T><<<1, 1, 0, stream>>>(data, index, value);
}

/**
 * @brief 启动 clone kernel：按逻辑顺序拷贝非连续张量实现
 * @note 此函数在 .cu 文件中实现，用 nvcc 编译，仅供本文件内部使用
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

// ============================================================================
// 非计算类算子实现
// ============================================================================

/**
 * @brief CUDA clone：深拷贝张量（支持非连续张量）
 * @param mat 输入矩阵
 * @return 拷贝后的矩阵（连续存储）
 */
std::unique_ptr<Mat> clone(const OriginMat &mat)
{
    // 深拷贝：创建新的 Storage 并复制数据（真正的独立副本）
    size_t data_size = mat.elements() * element_size(mat.dtype());
    auto new_storage = Storage::create(data_size, mat.device().type(), mat.device().index());

    // 如果张量是连续的，可以直接使用 memcpy（快速路径）
    if (mat.is_contiguous())
    {
        // 先同步，确保所有之前的异步kernel操作完成
        // 然后再复制数据，确保复制的是最新的数据
        cudaDeviceSynchronize();
        cudaError_t err = cudaMemcpy(new_storage->data(), mat.storage()->data(), data_size, cudaMemcpyDeviceToDevice);
        if (err != cudaSuccess)
        {
            THROW_RUNTIME_ERROR("CUDA memory copy failed in clone: {}", cudaGetErrorString(err));
        }
    }
    else
    {
        // 对于非连续张量，需要按逻辑顺序拷贝（使用 strides）
        // 计算目标张量的连续 strides（用于写入和构造 OriginMat）
        auto output_strides = utils::compute_strides(mat.shape());

        // CUDA 版本：使用 kernel 按逻辑顺序拷贝
        cudaDeviceSynchronize();

        device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
            const T *src_data = static_cast<const T *>(mat.storage()->data());
            T *dst_data       = static_cast<T *>(new_storage->data());

            size_t total_elements = mat.elements();
            size_t ndim           = mat.shape().size();

            std::vector<size_t> shape_vec(mat.shape().dims().begin(), mat.shape().dims().end());
            size_t shape_size = ndim * sizeof(size_t);
            size_t total_size = 3 * shape_size;  // 三个数组的总大小

            size_t *d_combined = nullptr;
            CUDA_CHECK(cudaMalloc(&d_combined, total_size));

            size_t *d_shape          = d_combined;
            size_t *d_strides        = d_combined + ndim;
            size_t *d_output_strides = d_combined + 2 * ndim;

            // 复制数据到连续内存的不同区域
            CUDA_CHECK(cudaMemcpy(d_shape, shape_vec.data(), shape_size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_strides, mat.strides().data(), shape_size, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_output_strides, output_strides.data(), shape_size, cudaMemcpyHostToDevice));

            launch_clone_kernel<T>(src_data, dst_data, d_shape, d_strides, d_output_strides, ndim, total_elements, 0);

            CUDA_CHECK_ASYNC();

            CUDA_CHECK(cudaFree(d_combined));
        });

        // 使用已计算的 output_strides 创建 OriginMat，避免构造函数中重复计算
        return std::make_unique<OriginMat>(new_storage, mat.shape(), output_strides, mat.dtype());
    }

    // 创建新的 OriginMat，使用新的 Storage（连续情况，构造函数会计算 strides）
    return std::make_unique<OriginMat>(new_storage, mat.shape(), mat.dtype());
}

/**
 * @brief CUDA index_put：根据多维索引写入单个元素
 * @param mat 输入/输出矩阵（原地修改）
 * @param indices 多维索引
 * @param value 要写入的标量值
 */
void index_put(OriginMat &mat, std::initializer_list<size_t> indices, const Scalar &value)
{
    if (unlikely(indices.size() != mat.shape().size()))
    {
        THROW_INVALID_ARG("Index count ({}) does not match tensor dimension ({}). Indices: {}, Shape: {}",
                          indices.size(), mat.shape().size(), "[indices]", mat.shape().to_string());
    }

    // 验证每个索引值并计算内存偏移（使用 strides，支持非连续内存）
    size_t offset = 0;
    size_t i      = 0;
    for (auto idx : indices)
    {
        if (unlikely(idx >= mat.shape()[i]))
        {
            THROW_INVALID_ARG("Index {} out of range for dimension {} (size: {}). Indices: {}, Shape: {}", idx, i,
                              mat.shape()[i], "[indices]", mat.shape().to_string());
        }
        offset += idx * mat.strides()[i];
        ++i;
    }

    void *data_ptr = mat.storage()->data();
    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        T val   = value.to<T>();
        T *data = static_cast<T *>(data_ptr);
        launch_index_put_kernel<T>(data, offset, val, 0);
    });
}

}  // namespace cuda
}  // namespace origin
#endif  // __CUDACC__
