#include "origin/mat/origin/cuda/cuda_kernels.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/cuda/device_validation.cuh"
#include "origin/utils/exception.h"

namespace origin
{
namespace cuda
{

/**
 * @brief CUDA标量乘法算子实现
 * @details 将标量值与张量每个元素相乘
 */

/**
 * @brief 类型分发器 - 编译时特化
 * @tparam T 数据类型
 * @param a 输入矩阵A
 * @param scalar 标量值
 * @param c 输出矩阵C
 * @param n 元素总数
 * @param stream CUDA流
 */
template <typename T>
void launch_multiply_scalar_kernel(const T *a, T scalar, T *c, size_t n, cudaStream_t stream = 0)
{
    // 根据数据大小选择最优的内核
    // 使用标量内核启动器
    launch_scalar_kernel<T, MultiplyOp>(a, scalar, c, n, MultiplyOp{}, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("CUDA multiply_scalar kernel launch failed: {}", cudaGetErrorString(err));
    }
}

/**
 * @brief 运行时类型分发
 * @param dtype 数据类型
 * @param a 输入矩阵A的设备指针
 * @param scalar 标量值
 * @param c 输出矩阵C的设备指针
 * @param n 元素总数
 * @param stream CUDA流
 */
void dispatch_multiply_scalar(DataType dtype, const void *a, data_t scalar, void *c, size_t n, cudaStream_t stream = 0)
{
    switch (dtype)
    {
        case DataType::kFloat32:
            launch_multiply_scalar_kernel<float>(static_cast<const float *>(a), static_cast<float>(scalar),
                                                 static_cast<float *>(c), n, stream);
            break;
        case DataType::kFloat64:
            launch_multiply_scalar_kernel<double>(static_cast<const double *>(a), static_cast<double>(scalar),
                                                  static_cast<double *>(c), n, stream);
            break;
        case DataType::kInt32:
            launch_multiply_scalar_kernel<int32_t>(static_cast<const int32_t *>(a), static_cast<int32_t>(scalar),
                                                   static_cast<int32_t *>(c), n, stream);
            break;
        case DataType::kInt8:
            launch_multiply_scalar_kernel<int8_t>(static_cast<const int8_t *>(a), static_cast<int8_t>(scalar),
                                                  static_cast<int8_t *>(c), n, stream);
            break;
        default:
            THROW_INVALID_ARG("Unsupported data type for CUDA multiply_scalar operation: {}", dtype_to_string(dtype));
    }
}

/**
 * @brief multiply_scalar算子实现
 * @param mat 输入矩阵
 * @param scalar 标量值
 * @return 标量乘法结果矩阵
 */
std::unique_ptr<Mat> multiply_scalar(const OriginMat &mat, data_t scalar)
{
    // 验证输入
    // 使用统一的设备检查
    validation::validate_cuda_device(mat, "multiply_scalar");

    // 创建结果张量
    auto result = std::unique_ptr<OriginMat>(new OriginMat(mat.shape(), mat.dtype(), mat.device()));

    // 获取数据指针
    const void *a_data = mat.storage()->data();
    void *c_data       = result->storage()->data();

    // 启动CUDA内核
    dispatch_multiply_scalar(mat.dtype(), a_data, scalar, c_data, mat.elements());

    // 同步等待完成
    cudaDeviceSynchronize();

    return result;
}

}  // namespace cuda
}  // namespace origin
