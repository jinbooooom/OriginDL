#include <cuda_runtime.h>
#include <cstring>
#include <type_traits>
#include <vector>
#include "origin/mat/origin/cuda/cuda_ops.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/conv_utils.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cuda
{

// ==================== CUDA Kernels ====================

/**
 * @brief im2col CUDA kernel：将图像转换为列矩阵
 * @tparam T 数据类型
 */
template <typename T>
__global__ void im2col_kernel(const T *__restrict__ img,
                              T *__restrict__ col,
                              size_t N,
                              size_t C,
                              size_t H,
                              size_t W,
                              int KH,
                              int KW,
                              int SH,
                              int SW,
                              int PH,
                              int PW,
                              int OH,
                              int OW,
                              size_t padded_H,
                              size_t padded_W,
                              bool to_matrix)
{
    // 计算全局线程索引
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (to_matrix)
    {
        // 输出形状: (N*OH*OW, C*KH*KW)
        size_t out_rows = N * OH * OW;
        size_t out_cols = C * KH * KW;

        if (idx < out_rows * out_cols)
        {
            size_t row_idx = idx / out_cols;
            size_t col_idx = idx % out_cols;

            size_t n     = row_idx / (OH * OW);
            size_t oh_ow = row_idx % (OH * OW);
            int oh       = oh_ow / OW;
            int ow       = oh_ow % OW;

            size_t c     = col_idx / (KH * KW);
            size_t kh_kw = col_idx % (KH * KW);
            int kh       = kh_kw / KW;
            int kw       = kh_kw % KW;

            int h_idx = kh + SH * oh;
            int w_idx = kw + SW * ow;

            if (h_idx >= 0 && h_idx < static_cast<int>(padded_H) && w_idx >= 0 && w_idx < static_cast<int>(padded_W))
            {
                size_t img_idx = n * C * padded_H * padded_W + c * padded_H * padded_W + h_idx * padded_W + w_idx;
                col[idx]       = img[img_idx];
            }
            else
            {
                col[idx] = static_cast<T>(0);
            }
        }
    }
    else
    {
        // 输出形状: (N, C, KH, KW, OH, OW)
        size_t total_elements = N * C * KH * KW * OH * OW;
        if (idx < total_elements)
        {
            size_t n   = idx / (C * KH * KW * OH * OW);
            size_t rem = idx % (C * KH * KW * OH * OW);
            size_t c   = rem / (KH * KW * OH * OW);
            rem        = rem % (KH * KW * OH * OW);
            int kh     = rem / (KW * OH * OW);
            rem        = rem % (KW * OH * OW);
            int kw     = rem / (OH * OW);
            rem        = rem % (OH * OW);
            int oh     = rem / OW;
            int ow     = rem % OW;

            int h_idx = kh + SH * oh;
            int w_idx = kw + SW * ow;

            if (h_idx >= 0 && h_idx < static_cast<int>(padded_H) && w_idx >= 0 && w_idx < static_cast<int>(padded_W))
            {
                size_t img_idx = n * C * padded_H * padded_W + c * padded_H * padded_W + h_idx * padded_W + w_idx;
                col[idx]       = img[img_idx];
            }
            else
            {
                col[idx] = static_cast<T>(0);
            }
        }
    }
}

/**
 * @brief col2im CUDA kernel：将列矩阵转换回图像形状
 * @tparam T 数据类型
 */
template <typename T>
__global__ void col2im_kernel(const T *__restrict__ col,
                              T *__restrict__ img,
                              size_t N,
                              size_t C,
                              size_t H,
                              size_t W,
                              int KH,
                              int KW,
                              int SH,
                              int SW,
                              int PH,
                              int PW,
                              int OH,
                              int OW,
                              size_t padded_H,
                              size_t padded_W,
                              bool to_matrix)
{
    // 计算全局线程索引
    size_t idx            = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = N * C * padded_H * padded_W;

    if (idx < total_elements)
    {
        img[idx] = static_cast<T>(0);  // 初始化为0
    }

    __syncthreads();

    // 计算需要处理的元素总数
    size_t col_elements = to_matrix ? (N * OH * OW * C * KH * KW) : (N * C * KH * KW * OH * OW);

    // 每个线程处理一个输出位置
    for (size_t col_idx = idx; col_idx < col_elements; col_idx += blockDim.x * gridDim.x)
    {
        size_t n, c, kh, kw, oh, ow;

        if (to_matrix)
        {
            // 从 (N*OH*OW, C*KH*KW) 形状计算索引
            size_t row_idx     = col_idx / (C * KH * KW);
            size_t col_col_idx = col_idx % (C * KH * KW);

            n            = row_idx / (OH * OW);
            size_t oh_ow = row_idx % (OH * OW);
            oh           = oh_ow / OW;
            ow           = oh_ow % OW;

            c            = col_col_idx / (KH * KW);
            size_t kh_kw = col_col_idx % (KH * KW);
            kh           = kh_kw / KW;
            kw           = kh_kw % KW;
        }
        else
        {
            // 从 (N, C, KH, KW, OH, OW) 形状计算索引
            n          = col_idx / (C * KH * KW * OH * OW);
            size_t rem = col_idx % (C * KH * KW * OH * OW);
            c          = rem / (KH * KW * OH * OW);
            rem        = rem % (KH * KW * OH * OW);
            kh         = rem / (KW * OH * OW);
            rem        = rem % (KW * OH * OW);
            kw         = rem / (OH * OW);
            rem        = rem % (OH * OW);
            oh         = rem / OW;
            ow         = rem % OW;
        }

        int h_idx = kh + SH * oh;
        int w_idx = kw + SW * ow;

        if (h_idx >= 0 && h_idx < static_cast<int>(padded_H) && w_idx >= 0 && w_idx < static_cast<int>(padded_W))
        {
            size_t img_idx = n * C * padded_H * padded_W + c * padded_H * padded_W + h_idx * padded_W + w_idx;
            // atomicAdd 只支持 float 和 double
            if constexpr (std::is_same_v<T, float>)
            {
                atomicAdd(&img[img_idx], col[col_idx]);
            }
            else if constexpr (std::is_same_v<T, double>)
            {
                atomicAdd(&img[img_idx], col[col_idx]);
            }
            else
            {
                // 对于其他类型，使用原子操作
                // 对于整数类型，CUDA 也支持 atomicAdd
                if constexpr (std::is_integral_v<T>)
                {
                    atomicAdd(reinterpret_cast<unsigned long long *>(&img[img_idx]),
                              static_cast<unsigned long long>(col[col_idx]));
                }
                else
                {
                    // 对于其他类型，转换为 float 进行原子操作
                    float *img_float = reinterpret_cast<float *>(&img[img_idx]);
                    float val        = static_cast<float>(col[col_idx]);
                    atomicAdd(img_float, val);
                }
            }
        }
    }
}

/**
 * @brief 填充图像 CUDA kernel：将原始图像复制到填充位置
 * @tparam T 数据类型
 */
template <typename T>
__global__ void pad_image_kernel(const T *__restrict__ src,
                                 T *__restrict__ dst,
                                 size_t N,
                                 size_t C,
                                 size_t H,
                                 size_t W,
                                 size_t padded_H,
                                 size_t padded_W,
                                 int PH,
                                 int PW)
{
    size_t idx            = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = N * C * H * W;

    if (idx < total_elements)
    {
        size_t n   = idx / (C * H * W);
        size_t rem = idx % (C * H * W);
        size_t c   = rem / (H * W);
        rem        = rem % (H * W);
        size_t h   = rem / W;
        size_t w   = rem % W;

        size_t src_idx = n * C * H * W + c * H * W + h * W + w;
        size_t dst_h   = h + PH;
        size_t dst_w   = w + PW;
        size_t dst_idx = n * C * padded_H * padded_W + c * padded_H * padded_W + dst_h * padded_W + dst_w;

        dst[dst_idx] = src[src_idx];
    }
}

/**
 * @brief 转置 CUDA kernel：从 (N, OH, OW, OC) 到 (N, OC, OH, OW)
 * @tparam T 数据类型
 */
template <typename T>
__global__ void transpose_conv_output_kernel(const T *__restrict__ src,
                                             T *__restrict__ dst,
                                             size_t N,
                                             size_t OC,
                                             int OH,
                                             int OW)
{
    size_t idx            = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = N * OC * OH * OW;

    if (idx < total_elements)
    {
        size_t n   = idx / (OC * OH * OW);
        size_t rem = idx % (OC * OH * OW);
        size_t oc  = rem / (OH * OW);
        rem        = rem % (OH * OW);
        int oh     = rem / OW;
        int ow     = rem % OW;

        // 从 (N, OH, OW, OC) 读取
        size_t src_idx = n * OH * OW * OC + oh * OW * OC + ow * OC + oc;
        dst[idx]       = src[src_idx];
    }
}

/**
 * @brief 移除填充 CUDA kernel：从填充图像中提取原始图像
 * @tparam T 数据类型
 */
template <typename T>
__global__ void unpad_image_kernel(const T *__restrict__ src,
                                   T *__restrict__ dst,
                                   size_t N,
                                   size_t C,
                                   size_t H,
                                   size_t W,
                                   size_t padded_H,
                                   size_t padded_W,
                                   int PH,
                                   int PW)
{
    size_t idx            = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = N * C * H * W;

    if (idx < total_elements)
    {
        size_t n   = idx / (C * H * W);
        size_t rem = idx % (C * H * W);
        size_t c   = rem / (H * W);
        rem        = rem % (H * W);
        size_t h   = rem / W;
        size_t w   = rem % W;

        size_t src_h   = h + PH;
        size_t src_w   = w + PW;
        size_t src_idx = n * C * padded_H * padded_W + c * padded_H * padded_W + src_h * padded_W + src_w;
        size_t dst_idx = n * C * H * W + c * H * W + h * W + w;

        dst[dst_idx] = src[src_idx];
    }
}

// ==================== 内部辅助函数 ====================

namespace
{

/**
 * @brief im2col 内部实现（CUDA版本）
 */
std::unique_ptr<Mat> im2col_impl(const OriginMat &img,
                                 std::pair<int, int> kernel_size,
                                 std::pair<int, int> stride,
                                 std::pair<int, int> pad,
                                 bool to_matrix)
{
    auto img_shape = img.shape();

    // 检查输入形状：必须是 (N, C, H, W)
    if (img_shape.size() != 4)
    {
        THROW_INVALID_ARG("im2col: input must be 4D (N, C, H, W), but got shape {}", img_shape.to_string());
    }

    size_t N = img_shape[0];
    size_t C = img_shape[1];
    size_t H = img_shape[2];
    size_t W = img_shape[3];

    int KH = kernel_size.first;
    int KW = kernel_size.second;
    int SH = stride.first;
    int SW = stride.second;
    int PH = pad.first;
    int PW = pad.second;

    // 计算输出尺寸
    int OH = get_conv_outsize(static_cast<int>(H), KH, SH, PH);
    int OW = get_conv_outsize(static_cast<int>(W), KW, SW, PW);

    if (OH <= 0 || OW <= 0)
    {
        THROW_INVALID_ARG(
            "im2col: invalid output size OH={}, OW={} for input H={}, W={}, kernel=({},{}), "
            "stride=({},{}), pad=({},{})",
            OH, OW, H, W, KH, KW, SH, SW, PH, PW);
    }

    // 创建填充后的图像
    size_t padded_H = H + 2 * PH + SH - 1;
    size_t padded_W = W + 2 * PW + SW - 1;
    auto padded_img = std::make_unique<OriginMat>(Shape{N, C, padded_H, padded_W}, img.dtype(), img.device());

    // 在 GPU 上填充图像
    device_common::TypeDispatcher::dispatch_void(img.dtype(), [&]<typename T>() {
        const T *img_data = img.data_ptr<T>();
        T *padded_data    = padded_img->data_ptr<T>();

        // 初始化填充图像为0
        cudaMemset(padded_data, 0, N * C * padded_H * padded_W * sizeof(T));
        CUDA_CHECK(cudaGetLastError());

        // 使用 CUDA kernel 复制原始图像到填充位置
        size_t total_elements = N * C * H * W;
        int threads_per_block = 256;
        int num_blocks        = (total_elements + threads_per_block - 1) / threads_per_block;

        pad_image_kernel<T>
            <<<num_blocks, threads_per_block>>>(img_data, padded_data, N, C, H, W, padded_H, padded_W, PH, PW);
        CUDA_CHECK_ASYNC();
    });

    // 创建输出矩阵
    Shape out_shape;
    if (to_matrix)
    {
        out_shape = Shape{N * OH * OW, C * KH * KW};
    }
    else
    {
        out_shape = Shape{
            N, C, static_cast<size_t>(KH), static_cast<size_t>(KW), static_cast<size_t>(OH), static_cast<size_t>(OW)};
    }
    auto result = std::make_unique<OriginMat>(out_shape, img.dtype(), img.device());

    // 启动 CUDA kernel
    device_common::TypeDispatcher::dispatch_void(img.dtype(), [&]<typename T>() {
        const T *padded_data = padded_img->data_ptr<T>();
        T *col_data          = result->data_ptr<T>();

        size_t total_elements = to_matrix ? (N * OH * OW * C * KH * KW) : (N * C * KH * KW * OH * OW);
        int threads_per_block = 256;
        int num_blocks        = (total_elements + threads_per_block - 1) / threads_per_block;

        im2col_kernel<T><<<num_blocks, threads_per_block>>>(padded_data, col_data, N, C, H, W, KH, KW, SH, SW, PH, PW,
                                                            OH, OW, padded_H, padded_W, to_matrix);
    });

    CUDA_CHECK_ASYNC();

    return result;
}

/**
 * @brief col2im 内部实现（CUDA版本）
 */
std::unique_ptr<Mat> col2im_impl(const OriginMat &col,
                                 const Shape &input_shape,
                                 std::pair<int, int> kernel_size,
                                 std::pair<int, int> stride,
                                 std::pair<int, int> pad,
                                 bool to_matrix)
{
    // 检查输入形状
    if (input_shape.size() != 4)
    {
        THROW_INVALID_ARG("col2im: input_shape must be 4D (N, C, H, W), but got shape {}", input_shape.to_string());
    }

    auto col_shape = col.shape();

    size_t N = input_shape[0];
    size_t C = input_shape[1];
    size_t H = input_shape[2];
    size_t W = input_shape[3];

    int KH = kernel_size.first;
    int KW = kernel_size.second;
    int SH = stride.first;
    int SW = stride.second;
    int PH = pad.first;
    int PW = pad.second;

    // 计算输出尺寸
    int OH = get_conv_outsize(static_cast<int>(H), KH, SH, PH);
    int OW = get_conv_outsize(static_cast<int>(W), KW, SW, PW);

    // 创建输出图像（带填充）
    size_t padded_H = H + 2 * PH + SH - 1;
    size_t padded_W = W + 2 * PW + SW - 1;
    auto padded_img = std::make_unique<OriginMat>(Shape{N, C, padded_H, padded_W}, col.dtype(), col.device());

    // 初始化填充图像为0
    device_common::TypeDispatcher::dispatch_void(col.dtype(), [&]<typename T>() {
        T *padded_data = padded_img->data_ptr<T>();
        cudaMemset(padded_data, 0, N * C * padded_H * padded_W * sizeof(T));
    });

    // 启动 CUDA kernel 进行 col2im
    device_common::TypeDispatcher::dispatch_void(col.dtype(), [&]<typename T>() {
        const T *col_data = col.data_ptr<T>();
        T *padded_data    = padded_img->data_ptr<T>();

        size_t col_elements   = to_matrix ? (N * OH * OW * C * KH * KW) : (N * C * KH * KW * OH * OW);
        int threads_per_block = 256;
        int num_blocks        = (col_elements + threads_per_block - 1) / threads_per_block;

        col2im_kernel<T><<<num_blocks, threads_per_block>>>(col_data, padded_data, N, C, H, W, KH, KW, SH, SW, PH, PW,
                                                            OH, OW, padded_H, padded_W, to_matrix);
    });

    CUDA_CHECK_ASYNC();

    // 移除填充，得到最终输出
    auto result = std::make_unique<OriginMat>(input_shape, col.dtype(), col.device());

    device_common::TypeDispatcher::dispatch_void(col.dtype(), [&]<typename T>() {
        const T *padded_data = padded_img->data_ptr<T>();
        T *result_data       = result->data_ptr<T>();

        // 使用 CUDA kernel 移除填充
        size_t total_elements = N * C * H * W;
        int threads_per_block = 256;
        int num_blocks        = (total_elements + threads_per_block - 1) / threads_per_block;

        unpad_image_kernel<T>
            <<<num_blocks, threads_per_block>>>(padded_data, result_data, N, C, H, W, padded_H, padded_W, PH, PW);
        CUDA_CHECK_ASYNC();
    });

    return result;
}

}  // anonymous namespace

// ==================== 对外接口 ====================

std::unique_ptr<Mat> im2col(const OriginMat &img,
                            std::pair<int, int> kernel_size,
                            std::pair<int, int> stride,
                            std::pair<int, int> pad,
                            bool to_matrix)
{
    return im2col_impl(img, kernel_size, stride, pad, to_matrix);
}

std::unique_ptr<Mat> col2im(const OriginMat &col,
                            const Shape &input_shape,
                            std::pair<int, int> kernel_size,
                            std::pair<int, int> stride,
                            std::pair<int, int> pad,
                            bool to_matrix)
{
    return col2im_impl(col, input_shape, kernel_size, stride, pad, to_matrix);
}

// ==================== conv2d 实现 ====================

std::unique_ptr<Mat> conv2d(const OriginMat &x,
                            const OriginMat &W,
                            const OriginMat *b,
                            std::pair<int, int> stride,
                            std::pair<int, int> pad)
{
    // 输入验证
    if (x.shape().size() != 4)
    {
        THROW_INVALID_ARG("conv2d: x must be 4D (N, C, H, W), but got shape {}", x.shape().to_string());
    }
    if (W.shape().size() != 4)
    {
        THROW_INVALID_ARG("conv2d: W must be 4D (OC, C, KH, KW), but got shape {}", W.shape().to_string());
    }

    size_t N    = x.shape()[0];
    size_t C    = x.shape()[1];
    size_t H    = x.shape()[2];
    size_t W_in = x.shape()[3];

    size_t OC   = W.shape()[0];
    size_t C_in = W.shape()[1];
    size_t KH   = W.shape()[2];
    size_t KW   = W.shape()[3];

    // 检查通道数是否匹配
    if (C != C_in)
    {
        THROW_INVALID_ARG("conv2d: channel mismatch - x has {} channels, but W expects {} channels", C, C_in);
    }

    // 计算输出尺寸
    int OH = get_conv_outsize(static_cast<int>(H), static_cast<int>(KH), stride.first, pad.first);
    int OW = get_conv_outsize(static_cast<int>(W_in), static_cast<int>(KW), stride.second, pad.second);

    if (OH <= 0 || OW <= 0)
    {
        THROW_INVALID_ARG(
            "conv2d: invalid output size OH={}, OW={} for input H={}, W={}, kernel=({},{}), "
            "stride=({},{}), pad=({},{})",
            OH, OW, H, W_in, KH, KW, stride.first, stride.second, pad.first, pad.second);
    }

    // 检查偏置
    if (b != nullptr)
    {
        if (b->shape().size() != 1 || b->shape()[0] != OC)
        {
            THROW_INVALID_ARG("conv2d: b must be 1D with size {}, but got shape {}", OC, b->shape().to_string());
        }
    }

    // 创建输出 Mat，形状为 (N, OC, OH, OW)
    Shape out_shape{N, OC, static_cast<size_t>(OH), static_cast<size_t>(OW)};
    auto result = std::make_unique<OriginMat>(out_shape, x.dtype(), x.device());

    // 使用 im2col + matmul 实现卷积
    // 1. im2col: (N, C, H, W) -> (N*OH*OW, C*KH*KW)
    auto col = im2col_impl(x, std::make_pair(static_cast<int>(KH), static_cast<int>(KW)), stride, pad, true);

    // 2. 将卷积核 reshape 为 (OC, C*KH*KW) 并转置为 (C*KH*KW, OC)
    auto W_reshaped          = W.reshape(Shape{OC, C * KH * KW});
    auto W_T                 = W_reshaped->transpose();
    const OriginMat &W_T_mat = static_cast<const OriginMat &>(*W_T);

    // 3. 矩阵乘法: col @ W_T -> (N*OH*OW, OC)
    const OriginMat &col_mat = static_cast<const OriginMat &>(*col);
    auto y_flat              = col_mat.matmul(W_T_mat);

    // 4. 添加偏置（如果存在）
    if (b != nullptr)
    {
        // 广播偏置: (OC,) -> (N*OH*OW, OC)
        auto b_broadcast            = b->broadcast_to(Shape{N * static_cast<size_t>(OH) * static_cast<size_t>(OW), OC});
        const OriginMat &y_flat_mat = static_cast<const OriginMat &>(*y_flat);
        const OriginMat &b_broadcast_mat = static_cast<const OriginMat &>(*b_broadcast);
        y_flat                           = y_flat_mat.operator+(b_broadcast_mat);
    }

    // 5. Reshape 并转置: (N*OH*OW, OC) -> (N, OH, OW, OC) -> (N, OC, OH, OW)
    const OriginMat &y_flat_mat = static_cast<const OriginMat &>(*y_flat);
    auto y_reshaped             = y_flat_mat.reshape(Shape{N, static_cast<size_t>(OH), static_cast<size_t>(OW), OC});

    // 使用 CUDA kernel 进行转置
    device_common::TypeDispatcher::dispatch_void(x.dtype(), [&]<typename T>() {
        const OriginMat &y_reshaped_mat = static_cast<const OriginMat &>(*y_reshaped);
        const T *src_data               = y_reshaped_mat.data_ptr<T>();
        T *dst_data                     = result->data_ptr<T>();

        size_t total_elements = N * OC * OH * OW;
        int threads_per_block = 256;
        int num_blocks        = (total_elements + threads_per_block - 1) / threads_per_block;

        transpose_conv_output_kernel<T><<<num_blocks, threads_per_block>>>(src_data, dst_data, N, OC, OH, OW);
    });

    CUDA_CHECK_ASYNC();

    return result;
}

std::vector<std::unique_ptr<Mat>> conv2d_backward(const OriginMat &gy,
                                                  const OriginMat &x,
                                                  const OriginMat &W,
                                                  const OriginMat *b,
                                                  std::pair<int, int> stride,
                                                  std::pair<int, int> pad)
{
    // 输入验证
    if (gy.shape().size() != 4)
    {
        THROW_INVALID_ARG("conv2d_backward: gy must be 4D (N, OC, OH, OW), but got shape {}", gy.shape().to_string());
    }
    if (x.shape().size() != 4)
    {
        THROW_INVALID_ARG("conv2d_backward: x must be 4D (N, C, H, W), but got shape {}", x.shape().to_string());
    }
    if (W.shape().size() != 4)
    {
        THROW_INVALID_ARG("conv2d_backward: W must be 4D (OC, C, KH, KW), but got shape {}", W.shape().to_string());
    }

    size_t N  = x.shape()[0];
    size_t C  = x.shape()[1];
    size_t OC = W.shape()[0];
    size_t KH = W.shape()[2];
    size_t KW = W.shape()[3];

    int OH = static_cast<int>(gy.shape()[2]);
    int OW = static_cast<int>(gy.shape()[3]);

    std::vector<std::unique_ptr<Mat>> grads;

    // 1. 计算 gW (卷积核梯度)
    // gy 形状: (N, OC, OH, OW) -> reshape 为 (N*OH*OW, OC)
    auto gy_reshaped = gy.reshape(Shape{N * static_cast<size_t>(OH) * static_cast<size_t>(OW), OC});
    // 转置为 (OC, N*OH*OW)
    auto gy_T                 = gy_reshaped->transpose();
    const OriginMat &gy_T_mat = static_cast<const OriginMat &>(*gy_T);

    // 使用 im2col 将输入转换为列矩阵
    auto col = im2col_impl(x, std::make_pair(static_cast<int>(KH), static_cast<int>(KW)), stride, pad, true);
    // col 形状: (N*OH*OW, C*KH*KW)
    const OriginMat &col_mat = static_cast<const OriginMat &>(*col);

    // gW = gy_T @ col，然后 reshape 为 (OC, C, KH, KW)
    auto gW_flat = gy_T_mat.matmul(col_mat);
    // gW_flat 形状: (OC, C*KH*KW)
    const OriginMat &gW_flat_mat = static_cast<const OriginMat &>(*gW_flat);
    auto gW                      = gW_flat_mat.reshape(Shape{OC, C, KH, KW});
    grads.push_back(std::move(gW));

    // 2. 计算 gb (偏置梯度)
    if (b != nullptr)
    {
        // gb = gy.sum(axis=(0, 2, 3))
        // gy 形状: (N, OC, OH, OW)
        // 方法：依次对维度2(OH), 2(OW), 0(N)求和
        auto gy_sum_h                 = gy.sum(2);  // sum over OH, shape: (N, OC, OW)
        const OriginMat &gy_sum_h_mat = static_cast<const OriginMat &>(*gy_sum_h);
        auto gy_sum_w                 = gy_sum_h_mat.sum(2);  // sum over OW, shape: (N, OC)
        const OriginMat &gy_sum_w_mat = static_cast<const OriginMat &>(*gy_sum_w);
        auto gb_mat                   = gy_sum_w_mat.sum(0);  // sum over N, shape: (OC,)
        auto gb                       = std::unique_ptr<OriginMat>(static_cast<OriginMat *>(gb_mat.release()));
        // 强制reshape到(OC,)，确保形状正确
        auto gb_reshaped = gb->reshape(Shape{OC});
        grads.push_back(std::move(gb_reshaped));
    }

    // 3. 计算 gx (输入梯度)
    // 将卷积核 reshape
    auto W_reshaped = W.reshape(Shape{OC, C * KH * KW});
    // gy_reshaped 形状: (N*OH*OW, OC)
    const OriginMat &gy_reshaped_mat = static_cast<const OriginMat &>(*gy_reshaped);
    const OriginMat &W_reshaped_mat  = static_cast<const OriginMat &>(*W_reshaped);
    // gx_col = gy_reshaped @ W_reshaped
    auto gx_col = gy_reshaped_mat.matmul(W_reshaped_mat);
    // gx_col 形状: (N*OH*OW, C*KH*KW)

    // 使用 col2im 转换回图像形状
    const OriginMat &gx_col_mat = static_cast<const OriginMat &>(*gx_col);
    auto gx = col2im_impl(gx_col_mat, x.shape(), std::make_pair(static_cast<int>(KH), static_cast<int>(KW)), stride,
                          pad, true);
    grads.insert(grads.begin(), std::move(gx));  // 插入到开头，顺序为 {gx, gW, [gb]}

    return grads;
}

}  // namespace cuda
}  // namespace origin
