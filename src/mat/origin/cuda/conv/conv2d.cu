#include "origin/mat/origin/cuda/cuda_ops.cuh"
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/cuda/cublas_wrapper.h"
#ifdef HAVE_CUDNN
#include "origin/mat/origin/cuda/cudnn_wrapper.h"
#include <cudnn.h>
#endif
#include "origin/mat/origin/origin_mat.h"
#include "origin/utils/exception.h"
#include "origin/utils/conv_utils.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include <vector>
#include <cstring>
#include <type_traits>
#include <cuda_runtime.h>

// 前向声明 GPU matmul 函数
namespace origin {
namespace cuda {
std::unique_ptr<Mat> matmul(const OriginMat &a, const OriginMat &b);
}  // namespace cuda
}  // namespace origin

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
__global__ void im2col_kernel(const T *__restrict__ img, T *__restrict__ col, size_t N, size_t C, size_t H, size_t W,
                               int KH, int KW, int SH, int SW, int PH, int PW, int OH, int OW, size_t padded_H,
                               size_t padded_W, bool to_matrix)
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

            size_t n = row_idx / (OH * OW);
            size_t oh_ow = row_idx % (OH * OW);
            int oh = oh_ow / OW;
            int ow = oh_ow % OW;

            size_t c = col_idx / (KH * KW);
            size_t kh_kw = col_idx % (KH * KW);
            int kh = kh_kw / KW;
            int kw = kh_kw % KW;

            // h_idx 和 w_idx 的计算：对于输出位置 (oh, ow)，卷积核覆盖的输入区域是
            // h_start = oh * SH - PH (相对于原始输入图像，可能为负数)
            // w_start = ow * SW - PW
            // 卷积核内的位置 (kh, kw) 对应的输入位置是：
            // h = h_start + kh = oh * SH - PH + kh (相对于原始输入图像，可能为负数)
            // w = w_start + kw = ow * SW - PW + kw
            // 在 padded_img 中，原始图像从 (PH, PW) 开始，所以：
            // padded_h = h + PH = oh * SH - PH + kh + PH = oh * SH + kh
            // padded_w = w + PW = ow * SW - PW + kw + PW = ow * SW + kw
            // 所以 h_idx = kh + SH * oh = oh * SH + kh 已经是 padded_img 中的索引了
            int h_idx = kh + SH * oh;
            int w_idx = kw + SW * ow;

            if (h_idx >= 0 && h_idx < static_cast<int>(padded_H) && w_idx >= 0 && w_idx < static_cast<int>(padded_W))
            {
                size_t img_idx = n * C * padded_H * padded_W + c * padded_H * padded_W + h_idx * padded_W + w_idx;
                col[idx] = img[img_idx];
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
            size_t n = idx / (C * KH * KW * OH * OW);
            size_t rem = idx % (C * KH * KW * OH * OW);
            size_t c = rem / (KH * KW * OH * OW);
            rem = rem % (KH * KW * OH * OW);
            int kh = rem / (KW * OH * OW);
            rem = rem % (KW * OH * OW);
            int kw = rem / (OH * OW);
            rem = rem % (OH * OW);
            int oh = rem / OW;
            int ow = rem % OW;

            // h_idx 和 w_idx 的计算：对于输出位置 (oh, ow)，卷积核覆盖的输入区域是
            // h_start = oh * SH - PH (相对于原始输入图像，可能为负数)
            // w_start = ow * SW - PW
            // 卷积核内的位置 (kh, kw) 对应的输入位置是：
            // h = h_start + kh = oh * SH - PH + kh (相对于原始输入图像，可能为负数)
            // w = w_start + kw = ow * SW - PW + kw
            // 在 padded_img 中，原始图像从 (PH, PW) 开始，所以：
            // padded_h = h + PH = oh * SH - PH + kh + PH = oh * SH + kh
            // padded_w = w + PW = ow * SW - PW + kw + PW = ow * SW + kw
            // 所以 h_idx = kh + SH * oh = oh * SH + kh 已经是 padded_img 中的索引了
            int h_idx = kh + SH * oh;
            int w_idx = kw + SW * ow;

            if (h_idx >= 0 && h_idx < static_cast<int>(padded_H) && w_idx >= 0 && w_idx < static_cast<int>(padded_W))
            {
                size_t img_idx = n * C * padded_H * padded_W + c * padded_H * padded_W + h_idx * padded_W + w_idx;
                col[idx] = img[img_idx];
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
__global__ void col2im_kernel(const T *__restrict__ col, T *__restrict__ img, size_t N, size_t C, size_t H, size_t W,
                               int KH, int KW, int SH, int SW, int PH, int PW, int OH, int OW, size_t padded_H,
                               size_t padded_W, bool to_matrix)
{
    // 计算全局线程索引
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
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
            size_t row_idx = col_idx / (C * KH * KW);
            size_t col_col_idx = col_idx % (C * KH * KW);

            n = row_idx / (OH * OW);
            size_t oh_ow = row_idx % (OH * OW);
            oh = oh_ow / OW;
            ow = oh_ow % OW;

            c = col_col_idx / (KH * KW);
            size_t kh_kw = col_col_idx % (KH * KW);
            kh = kh_kw / KW;
            kw = kh_kw % KW;
        }
        else
        {
            // 从 (N, C, KH, KW, OH, OW) 形状计算索引
            n = col_idx / (C * KH * KW * OH * OW);
            size_t rem = col_idx % (C * KH * KW * OH * OW);
            c = rem / (KH * KW * OH * OW);
            rem = rem % (KH * KW * OH * OW);
            kh = rem / (KW * OH * OW);
            rem = rem % (KW * OH * OW);
            kw = rem / (OH * OW);
            rem = rem % (OH * OW);
            oh = rem / OW;
            ow = rem % OW;
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
                    atomicAdd(reinterpret_cast<unsigned long long *>(&img[img_idx]), static_cast<unsigned long long>(col[col_idx]));
                }
                else
                {
                    // 对于其他类型，转换为 float 进行原子操作
                    float *img_float = reinterpret_cast<float *>(&img[img_idx]);
                    float val = static_cast<float>(col[col_idx]);
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
__global__ void pad_image_kernel(const T *__restrict__ src, T *__restrict__ dst, size_t N, size_t C, size_t H, size_t W,
                                  size_t padded_H, size_t padded_W, int PH, int PW)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = N * C * H * W;

    if (idx < total_elements)
    {
        size_t n = idx / (C * H * W);
        size_t rem = idx % (C * H * W);
        size_t c = rem / (H * W);
        rem = rem % (H * W);
        size_t h = rem / W;
        size_t w = rem % W;

        size_t src_idx = n * C * H * W + c * H * W + h * W + w;
        size_t dst_h = h + PH;
        size_t dst_w = w + PW;
        size_t dst_idx = n * C * padded_H * padded_W + c * padded_H * padded_W + dst_h * padded_W + dst_w;

        dst[dst_idx] = src[src_idx];
    }
}

/**
 * @brief 转换权重从行主序到列主序（用于 cuDNN）
 * @tparam T 数据类型
 * @param src 源数据（行主序）
 * @param dst 目标数据（列主序）
 * @param OC 输出通道数
 * @param C 输入通道数
 * @param KH 卷积核高度
 * @param KW 卷积核宽度
 */
template <typename T>
__global__ void convert_filter_row_to_col_major_kernel(const T *__restrict__ src, T *__restrict__ dst,
                                                       size_t OC, size_t C, int KH, int KW)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = OC * C * KH * KW;

    if (idx < total_elements)
    {
        // 计算列主序索引 (oc, c, kh, kw)
        size_t oc = idx % OC;
        size_t rem = idx / OC;
        size_t c = rem % C;
        rem = rem / C;
        int kh = rem % KH;
        int kw = rem / KH;

        // 行主序索引: oc * (C*KH*KW) + c * (KH*KW) + kh * KW + kw
        size_t row_major_idx = oc * (C * KH * KW) + c * (KH * KW) + kh * KW + kw;
        dst[idx] = src[row_major_idx];
    }
}

/**
 * @brief 转置 CUDA kernel：从 (N, OH, OW, OC) 到 (N, OC, OH, OW)
 * @tparam T 数据类型
 */
template <typename T>
__global__ void transpose_conv_output_kernel(const T *__restrict__ src, T *__restrict__ dst, size_t N, size_t OC,
                                              int OH, int OW)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = N * OC * OH * OW;

    if (idx < total_elements)
    {
        size_t n = idx / (OC * OH * OW);
        size_t rem = idx % (OC * OH * OW);
        size_t oc = rem / (OH * OW);
        rem = rem % (OH * OW);
        int oh = rem / OW;
        int ow = rem % OW;

        // 从 (N, OH, OW, OC) 读取
        size_t src_idx = n * OH * OW * OC + oh * OW * OC + ow * OC + oc;
        dst[idx] = src[src_idx];
    }
}

/**
 * @brief 移除填充 CUDA kernel：从填充图像中提取原始图像
 * @tparam T 数据类型
 */
template <typename T>
__global__ void unpad_image_kernel(const T *__restrict__ src, T *__restrict__ dst, size_t N, size_t C, size_t H,
                                    size_t W, size_t padded_H, size_t padded_W, int PH, int PW)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t total_elements = N * C * H * W;

    if (idx < total_elements)
    {
        size_t n = idx / (C * H * W);
        size_t rem = idx % (C * H * W);
        size_t c = rem / (H * W);
        rem = rem % (H * W);
        size_t h = rem / W;
        size_t w = rem % W;

        size_t src_h = h + PH;
        size_t src_w = w + PW;
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
std::unique_ptr<Mat> im2col_impl(const OriginMat &img, std::pair<int, int> kernel_size, std::pair<int, int> stride,
                                 std::pair<int, int> pad, bool to_matrix)
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
        THROW_INVALID_ARG("im2col: invalid output size OH={}, OW={} for input H={}, W={}, kernel=({},{}), "
                         "stride=({},{}), pad=({},{})",
                         OH, OW, H, W, KH, KW, SH, SW, PH, PW);
    }

    // 创建填充后的图像
    // 标准公式：padded_H = H + 2 * PH
    // 但是，为了确保 stride>1 时所有输出位置都能正确访问，需要额外的空间
    // 对于 stride>1，最后一个输出位置可能需要访问到 H + PH + (OH-1)*SH + KH - 1
    // 所以需要：padded_H >= H + 2*PH + (OH-1)*SH + KH - 1 - (H + PH) = PH + (OH-1)*SH + KH - 1
    // 但是，这太复杂了。实际上，标准的 padded_H = H + 2 * PH 应该就足够了。
    // 让我先尝试使用标准公式，看看是否能解决问题。
    size_t padded_H = H + 2 * PH;
    size_t padded_W = W + 2 * PW;
    auto padded_img = std::make_unique<OriginMat>(Shape{N, C, padded_H, padded_W}, img.dtype(), img.device());

    // 在 GPU 上填充图像
    device_common::TypeDispatcher::dispatch_void(img.dtype(), [&]<typename T>() {
        const T *img_data = img.data_ptr<T>();
        T *padded_data = padded_img->data_ptr<T>();

        // 初始化填充图像为0
        cudaMemset(padded_data, 0, N * C * padded_H * padded_W * sizeof(T));
        CUDA_CHECK(cudaGetLastError());

        // 使用 CUDA kernel 复制原始图像到填充位置
        size_t total_elements = N * C * H * W;
        int threads_per_block = 256;
        int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

        pad_image_kernel<T><<<num_blocks, threads_per_block>>>(img_data, padded_data, N, C, H, W, padded_H, padded_W,
                                                               PH, PW);
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
        out_shape = Shape{N, C, static_cast<size_t>(KH), static_cast<size_t>(KW), static_cast<size_t>(OH),
                          static_cast<size_t>(OW)};
    }
    auto result = std::make_unique<OriginMat>(out_shape, img.dtype(), img.device());

    // 启动 CUDA kernel
    device_common::TypeDispatcher::dispatch_void(img.dtype(), [&]<typename T>() {
        const T *padded_data = padded_img->data_ptr<T>();
        T *col_data = result->data_ptr<T>();

        size_t total_elements = to_matrix ? (N * OH * OW * C * KH * KW) : (N * C * KH * KW * OH * OW);
        int threads_per_block = 256;
        int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

        im2col_kernel<T><<<num_blocks, threads_per_block>>>(padded_data, col_data, N, C, H, W, KH, KW, SH, SW, PH, PW,
                                                            OH, OW, padded_H, padded_W, to_matrix);
    });

    CUDA_CHECK_ASYNC();

    return result;
}

/**
 * @brief col2im 内部实现（CUDA版本）
 */
std::unique_ptr<Mat> col2im_impl(const OriginMat &col, const Shape &input_shape, std::pair<int, int> kernel_size,
                                  std::pair<int, int> stride, std::pair<int, int> pad, bool to_matrix)
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
    // 标准公式：padded_H = H + 2 * PH
    size_t padded_H = H + 2 * PH;
    size_t padded_W = W + 2 * PW;
    auto padded_img = std::make_unique<OriginMat>(Shape{N, C, padded_H, padded_W}, col.dtype(), col.device());

    // 初始化填充图像为0
    device_common::TypeDispatcher::dispatch_void(col.dtype(), [&]<typename T>() {
        T *padded_data = padded_img->data_ptr<T>();
        cudaMemset(padded_data, 0, N * C * padded_H * padded_W * sizeof(T));
    });

    // 启动 CUDA kernel 进行 col2im
    device_common::TypeDispatcher::dispatch_void(col.dtype(), [&]<typename T>() {
        const T *col_data = col.data_ptr<T>();
        T *padded_data = padded_img->data_ptr<T>();

        size_t col_elements = to_matrix ? (N * OH * OW * C * KH * KW) : (N * C * KH * KW * OH * OW);
        int threads_per_block = 256;
        int num_blocks = (col_elements + threads_per_block - 1) / threads_per_block;

        col2im_kernel<T><<<num_blocks, threads_per_block>>>(col_data, padded_data, N, C, H, W, KH, KW, SH, SW, PH, PW,
                                                              OH, OW, padded_H, padded_W, to_matrix);
    });

    CUDA_CHECK_ASYNC();

    // 移除填充，得到最终输出
    auto result = std::make_unique<OriginMat>(input_shape, col.dtype(), col.device());

    device_common::TypeDispatcher::dispatch_void(col.dtype(), [&]<typename T>() {
        const T *padded_data = padded_img->data_ptr<T>();
        T *result_data = result->data_ptr<T>();

        // 使用 CUDA kernel 移除填充
        size_t total_elements = N * C * H * W;
        int threads_per_block = 256;
        int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

        unpad_image_kernel<T><<<num_blocks, threads_per_block>>>(padded_data, result_data, N, C, H, W, padded_H,
                                                                  padded_W, PH, PW);
        CUDA_CHECK_ASYNC();
    });

    return result;
}

}  // anonymous namespace

// ==================== 对外接口 ====================

std::unique_ptr<Mat> im2col(const OriginMat &img, std::pair<int, int> kernel_size, std::pair<int, int> stride,
                            std::pair<int, int> pad, bool to_matrix)
{
    return im2col_impl(img, kernel_size, stride, pad, to_matrix);
}

std::unique_ptr<Mat> col2im(const OriginMat &col, const Shape &input_shape, std::pair<int, int> kernel_size,
                            std::pair<int, int> stride, std::pair<int, int> pad, bool to_matrix)
{
    return col2im_impl(col, input_shape, kernel_size, stride, pad, to_matrix);
}

// ==================== conv2d 实现 ====================

std::unique_ptr<Mat> conv2d(const OriginMat &x, const OriginMat &W, const OriginMat *b, std::pair<int, int> stride,
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

    size_t N = x.shape()[0];
    size_t C = x.shape()[1];
    size_t H = x.shape()[2];
    size_t W_in = x.shape()[3];

    size_t OC = W.shape()[0];
    size_t C_in = W.shape()[1];
    size_t KH = W.shape()[2];
    size_t KW = W.shape()[3];

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
        THROW_INVALID_ARG("conv2d: invalid output size OH={}, OW={} for input H={}, W={}, kernel=({},{}), "
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

#ifdef HAVE_CUDNN
    // cuDNN 和 cuBLAS 已禁用，避免行列主序转换问题
    // 直接使用 im2col + CPU matmul 实现
    // if (x.dtype() == DataType::kFloat32)
    // {
    //     // cuDNN 代码已注释，避免数据布局转换问题
    // }
#endif

    // 回退到 im2col + cuBLAS GEMM 实现（当 cuDNN 不可用或数据类型不支持时）
    // 1. im2col: (N, C, H, W) -> (N*OH*OW, C*KH*KW)
    auto col = im2col_impl(x, std::make_pair(static_cast<int>(KH), static_cast<int>(KW)), stride, pad, true);

    // 调试：检查 im2col 的输入和输出（仅对 model.0.conv: N=4, C=3, OC=16, OH=160, OW=160, stride=2）
    if (N == 4 && C == 3 && OC == 16 && OH == 160 && OW == 160 && stride.first == 2 && stride.second == 2) {
        // 检查输入数据
        auto x_cpu = x.to_device(Device(DeviceType::kCPU, 0));
        const OriginMat &x_cpu_mat = static_cast<const OriginMat &>(*x_cpu);
        auto x_data = x_cpu_mat.to_vector<float>();
        std::cout << "\n=== DEBUG: conv2d input (model.0.conv) ===" << std::endl;
        std::cout << "Input shape: " << x.shape().to_string() << std::endl;
        std::cout << "Input first 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), x_data.size()); ++i) {
            std::cout << x_data[i] << " ";
        }
        std::cout << std::endl;
        // 检查输入 [0,0,0,0] 到 [0,0,0,9]
        std::cout << "Input [0,0,0,0] to [0,0,0,9]: ";
        for (int w_idx = 0; w_idx < 10; ++w_idx) {
            size_t idx = static_cast<size_t>(0) * C * H * W_in + static_cast<size_t>(0) * H * W_in + static_cast<size_t>(0) * W_in + static_cast<size_t>(w_idx);
            if (idx < x_data.size()) {
                std::cout << x_data[idx] << " ";
            }
        }
        std::cout << std::endl;
        
        // 检查 im2col 输出
        const OriginMat &col_mat_debug = static_cast<const OriginMat &>(*col);
        auto col_cpu = col_mat_debug.to_device(Device(DeviceType::kCPU, 0));
        const OriginMat &col_cpu_mat = static_cast<const OriginMat &>(*col_cpu);
        auto col_data = col_cpu_mat.to_vector<float>();
        std::cout << "\n=== DEBUG: im2col output (model.0.conv) ===" << std::endl;
        std::cout << "Shape: " << col_mat_debug.shape().to_string() << " (expected: {102400, 108})" << std::endl;
        std::cout << "First 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), col_data.size()); ++i) {
            std::cout << col_data[i] << " ";
        }
        std::cout << std::endl;
        // 检查 col[0,0] 到 col[9,0] (第一列的前10个值)
        size_t col_cols = col_mat_debug.shape()[1];
        std::cout << "col[0,0] to col[9,0] (col=0): ";
        for (int row = 0; row < 10; ++row) {
            size_t idx = row * col_cols + 0;
            if (idx < col_data.size()) {
                std::cout << col_data[idx] << " ";
            }
        }
        std::cout << std::endl;
        // 检查 col[0,1] 到 col[9,1] (第二列的前10个值)
        std::cout << "col[0,1] to col[9,1] (col=1): ";
        for (int row = 0; row < 10; ++row) {
            size_t idx = row * col_cols + 1;
            if (idx < col_data.size()) {
                std::cout << col_data[idx] << " ";
            }
        }
        std::cout << std::endl;
        // 检查 col[0,2] 到 col[9,2] (第三列的前10个值)
        std::cout << "col[0,2] to col[9,2] (col=2): ";
        for (int row = 0; row < 10; ++row) {
            size_t idx = row * col_cols + 2;
            if (idx < col_data.size()) {
                std::cout << col_data[idx] << " ";
            }
        }
        std::cout << std::endl;
        // 检查 col[0,:] 的前20个值
        std::cout << "col[0,:] first 20 values: ";
        for (int col_idx = 0; col_idx < 20; ++col_idx) {
            size_t idx = 0 * col_cols + col_idx;
            if (idx < col_data.size()) {
                std::cout << col_data[idx] << " ";
            }
        }
        std::cout << std::endl;
        // 检查 col[1,:] 的前20个值
        std::cout << "col[1,:] first 20 values: ";
        for (int col_idx = 0; col_idx < 20; ++col_idx) {
            size_t idx = 1 * col_cols + col_idx;
            if (idx < col_data.size()) {
                std::cout << col_data[idx] << " ";
            }
        }
        std::cout << std::endl;
        // 检查 col[2,:] 的前20个值
        std::cout << "col[2,:] first 20 values: ";
        for (int col_idx = 0; col_idx < 20; ++col_idx) {
            size_t idx = 2 * col_cols + col_idx;
            if (idx < col_data.size()) {
                std::cout << col_data[idx] << " ";
            }
        }
        std::cout << std::endl;
    }
    
    // 调试：检查 im2col 的输入和输出（仅对 model.1.conv: N=4, C=16, OC=32, OH=80, OW=80, stride=2）
    if (N == 4 && C == 16 && OC == 32 && OH == 80 && OW == 80 && stride.first == 2 && stride.second == 2) {
        // 检查输入数据
        auto x_cpu = x.to_device(Device(DeviceType::kCPU, 0));
        const OriginMat &x_cpu_mat = static_cast<const OriginMat &>(*x_cpu);
        auto x_data = x_cpu_mat.to_vector<float>();
        std::cout << "\n=== DEBUG: conv2d input (model.1.conv) ===" << std::endl;
        std::cout << "Input shape: " << x.shape().to_string() << std::endl;
        std::cout << "Input first 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), x_data.size()); ++i) {
            std::cout << x_data[i] << " ";
        }
        std::cout << std::endl;
        // 检查输入 [0,0,0,0] 到 [0,0,0,9]
        std::cout << "Input [0,0,0,0] to [0,0,0,9]: ";
        for (int w_idx = 0; w_idx < 10; ++w_idx) {
            size_t idx = static_cast<size_t>(0) * C * H * W_in + static_cast<size_t>(0) * H * W_in + static_cast<size_t>(0) * W_in + static_cast<size_t>(w_idx);
            if (idx < x_data.size()) {
                std::cout << x_data[idx] << " ";
            }
        }
        std::cout << std::endl;
    }
    
    // 调试：检查 im2col 的输入和输出（仅对 model.2.cv1.conv: N=4, C=32, OC=16, OH=80, OW=80, kernel=1x1）
    if (N == 4 && C == 32 && OC == 16 && OH == 80 && OW == 80 && KH == 1 && KW == 1 && pad.first == 0 && pad.second == 0) {
        // 检查输入数据
        auto x_cpu = x.to_device(Device(DeviceType::kCPU, 0));
        const OriginMat &x_cpu_mat = static_cast<const OriginMat &>(*x_cpu);
        auto x_data = x_cpu_mat.to_vector<float>();
        std::cout << "\n=== DEBUG: conv2d input (model.2.cv1.conv) ===" << std::endl;
        std::cout << "Input shape: " << x.shape().to_string() << std::endl;
        std::cout << "Input first 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), x_data.size()); ++i) {
            std::cout << x_data[i] << " ";
        }
        std::cout << std::endl;
        // 检查输入 [0,0,0,0] 到 [0,0,0,9]
        std::cout << "Input [0,0,0,0] to [0,0,0,9]: ";
        for (int w_idx = 0; w_idx < 10; ++w_idx) {
            size_t idx = static_cast<size_t>(0) * C * H * W_in + static_cast<size_t>(0) * H * W_in + static_cast<size_t>(0) * W_in + static_cast<size_t>(w_idx);
            if (idx < x_data.size()) {
                std::cout << x_data[idx] << " ";
            }
        }
        std::cout << std::endl;
    }
    
    // 调试：检查 im2col 的输入和输出（仅对 model.2.cv2.conv: N=4, C=32, OC=16, OH=80, OW=80, kernel=1x1）
    if (N == 4 && C == 32 && OC == 16 && OH == 80 && OW == 80 && KH == 1 && KW == 1) {
        // 检查输入数据
        auto x_cpu = x.to_device(Device(DeviceType::kCPU, 0));
        const OriginMat &x_cpu_mat = static_cast<const OriginMat &>(*x_cpu);
        auto x_data = x_cpu_mat.to_vector<float>();
        std::cout << "\n=== DEBUG: conv2d input (model.2.cv2.conv) ===" << std::endl;
        std::cout << "Input shape: " << x.shape().to_string() << std::endl;
        std::cout << "Input first 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), x_data.size()); ++i) {
            std::cout << x_data[i] << " ";
        }
        std::cout << std::endl;
        // 检查输入 [0,0,0,0] 到 [0,0,0,9]
        std::cout << "Input [0,0,0,0] to [0,0,0,9]: ";
        for (int w_idx = 0; w_idx < 10; ++w_idx) {
            size_t idx = static_cast<size_t>(0) * C * H * W_in + static_cast<size_t>(0) * H * W_in + static_cast<size_t>(0) * W_in + static_cast<size_t>(w_idx);
            if (idx < x_data.size()) {
                std::cout << x_data[idx] << " ";
            }
        }
        std::cout << std::endl;
    }
    
    // 调试：检查 im2col 的输入和输出（仅对 model.2.cv3.conv: N=4, C=32, OC=32, OH=80, OW=80, kernel=1x1）
    if (N == 4 && C == 32 && OC == 32 && OH == 80 && OW == 80 && KH == 1 && KW == 1) {
        // 检查输入数据
        auto x_cpu = x.to_device(Device(DeviceType::kCPU, 0));
        const OriginMat &x_cpu_mat = static_cast<const OriginMat &>(*x_cpu);
        auto x_data = x_cpu_mat.to_vector<float>();
        std::cout << "\n=== DEBUG: conv2d input (model.2.cv3.conv) ===" << std::endl;
        std::cout << "Input shape: " << x.shape().to_string() << std::endl;
        std::cout << "Input first 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), x_data.size()); ++i) {
            std::cout << x_data[i] << " ";
        }
        std::cout << std::endl;
        // 检查输入 [0,0,0,0] 到 [0,0,0,9]
        std::cout << "Input [0,0,0,0] to [0,0,0,9]: ";
        for (int w_idx = 0; w_idx < 10; ++w_idx) {
            size_t idx = static_cast<size_t>(0) * C * H * W_in + static_cast<size_t>(0) * H * W_in + static_cast<size_t>(0) * W_in + static_cast<size_t>(w_idx);
            if (idx < x_data.size()) {
                std::cout << x_data[idx] << " ";
            }
        }
        std::cout << std::endl;
    }
    
    // 调试：检查 im2col 的输入和输出（仅对 model.3.conv: N=4, C=32, OC=64, OH=40, OW=40, stride=2）
    if (N == 4 && C == 32 && OC == 64 && OH == 40 && OW == 40 && stride.first == 2 && stride.second == 2) {
        // 检查输入数据
        auto x_cpu = x.to_device(Device(DeviceType::kCPU, 0));
        const OriginMat &x_cpu_mat = static_cast<const OriginMat &>(*x_cpu);
        auto x_data = x_cpu_mat.to_vector<float>();
        std::cout << "\n=== DEBUG: conv2d input (model.3.conv) ===" << std::endl;
        std::cout << "Input shape: " << x.shape().to_string() << std::endl;
        std::cout << "Input first 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), x_data.size()); ++i) {
            std::cout << x_data[i] << " ";
        }
        std::cout << std::endl;
        // 检查输入 [0,0,0,0] 到 [0,0,0,9]
        std::cout << "Input [0,0,0,0] to [0,0,0,9]: ";
        for (int w_idx = 0; w_idx < 10; ++w_idx) {
            size_t idx = static_cast<size_t>(0) * C * H * W_in + static_cast<size_t>(0) * H * W_in + static_cast<size_t>(0) * W_in + static_cast<size_t>(w_idx);
            if (idx < x_data.size()) {
                std::cout << x_data[idx] << " ";
            }
        }
        std::cout << std::endl;
    }
    
    // 调试：检查 im2col 的输入和输出（仅对 model.2.cv2.conv: N=4, C=32, OC=16, OH=80, OW=80, kernel=1x1）
    if (N == 4 && C == 32 && OC == 16 && OH == 80 && OW == 80 && KH == 1 && KW == 1) {
        // 检查输入数据
        auto x_cpu = x.to_device(Device(DeviceType::kCPU, 0));
        const OriginMat &x_cpu_mat = static_cast<const OriginMat &>(*x_cpu);
        auto x_data = x_cpu_mat.to_vector<float>();
        std::cout << "\n=== DEBUG: conv2d input (model.2.cv2.conv) ===" << std::endl;
        std::cout << "Input shape: " << x.shape().to_string() << std::endl;
        std::cout << "Input first 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), x_data.size()); ++i) {
            std::cout << x_data[i] << " ";
        }
        std::cout << std::endl;
        // 检查输入 [0,0,0,0] 到 [0,0,0,9]
        std::cout << "Input [0,0,0,0] to [0,0,0,9]: ";
        for (int w_idx = 0; w_idx < 10; ++w_idx) {
            size_t idx = static_cast<size_t>(0) * C * H * W_in + static_cast<size_t>(0) * H * W_in + static_cast<size_t>(0) * W_in + static_cast<size_t>(w_idx);
            if (idx < x_data.size()) {
                std::cout << x_data[idx] << " ";
            }
        }
        std::cout << std::endl;
        std::cout << "Conv params: kernel=(" << KH << "," << KW << "), stride=(" << stride.first << "," << stride.second << "), padding=(" << pad.first << "," << pad.second << ")" << std::endl;
    }
    
    // 调试：检查 im2col 的输入和输出（仅对 model.4.m.1.cv1.conv: N=4, C=32, OC=32, OH=40, OW=40, kernel=1x1）
    if (N == 4 && C == 32 && OC == 32 && OH == 40 && OW == 40 && KH == 1 && KW == 1) {
        // 检查输入数据
        auto x_cpu = x.to_device(Device(DeviceType::kCPU, 0));
        const OriginMat &x_cpu_mat = static_cast<const OriginMat &>(*x_cpu);
        auto x_data = x_cpu_mat.to_vector<float>();
        std::cout << "\n=== DEBUG: conv2d input (model.4.m.1.cv1.conv) ===" << std::endl;
        std::cout << "Input shape: " << x.shape().to_string() << std::endl;
        std::cout << "Input first 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), x_data.size()); ++i) {
            std::cout << x_data[i] << " ";
        }
        std::cout << std::endl;
        // 检查输入 [0,0,0,0] 到 [0,0,0,9]
        std::cout << "Input [0,0,0,0] to [0,0,0,9]: ";
        for (int w_idx = 0; w_idx < 10; ++w_idx) {
            size_t idx = static_cast<size_t>(0) * C * H * W_in + static_cast<size_t>(0) * H * W_in + static_cast<size_t>(0) * W_in + static_cast<size_t>(w_idx);
            if (idx < x_data.size()) {
                std::cout << x_data[idx] << " ";
            }
        }
        std::cout << std::endl;
    }
    
    // 调试：检查 im2col 的输入和输出（仅对 model.4.m.1.cv2.conv: N=4, C=32, OC=32, OH=40, OW=40, kernel=3x3）
    if (N == 4 && C == 32 && OC == 32 && OH == 40 && OW == 40 && KH == 3 && KW == 3) {
        // 检查输入数据
        auto x_cpu = x.to_device(Device(DeviceType::kCPU, 0));
        const OriginMat &x_cpu_mat = static_cast<const OriginMat &>(*x_cpu);
        auto x_data = x_cpu_mat.to_vector<float>();
        std::cout << "\n=== DEBUG: conv2d input (model.4.m.1.cv2.conv) ===" << std::endl;
        std::cout << "Input shape: " << x.shape().to_string() << std::endl;
        std::cout << "Input first 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), x_data.size()); ++i) {
            std::cout << x_data[i] << " ";
        }
        std::cout << std::endl;
        // 检查输入 [0,0,0,0] 到 [0,0,0,9]
        std::cout << "Input [0,0,0,0] to [0,0,0,9]: ";
        for (int w_idx = 0; w_idx < 10; ++w_idx) {
            size_t idx = static_cast<size_t>(0) * C * H * W_in + static_cast<size_t>(0) * H * W_in + static_cast<size_t>(0) * W_in + static_cast<size_t>(w_idx);
            if (idx < x_data.size()) {
                std::cout << x_data[idx] << " ";
            }
        }
        std::cout << std::endl;
    }
    
    // 调试：检查 im2col 的输入和输出（仅对 model.4.cv1.conv: N=4, C=64, OC=32, OH=40, OW=40）
    if (N == 4 && C == 64 && OC == 32 && OH == 40 && OW == 40) {
        // 检查输入数据
        auto x_cpu = x.to_device(Device(DeviceType::kCPU, 0));
        const OriginMat &x_cpu_mat = static_cast<const OriginMat &>(*x_cpu);
        auto x_data = x_cpu_mat.to_vector<float>();
        std::cout << "\n=== DEBUG: conv2d input (model.4.cv1.conv) ===" << std::endl;
        std::cout << "Input shape: " << x.shape().to_string() << std::endl;
        std::cout << "Input first 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), x_data.size()); ++i) {
            std::cout << x_data[i] << " ";
        }
        std::cout << std::endl;
        // 检查输入 [0,0,0,0] 到 [0,0,0,9]
        std::cout << "Input [0,0,0,0] to [0,0,0,9]: ";
        for (int w = 0; w < 10; ++w) {
            size_t idx = static_cast<size_t>(0) * C * H * W_in + static_cast<size_t>(0) * H * W_in + static_cast<size_t>(0) * W_in + static_cast<size_t>(w);
            if (idx < x_data.size()) {
                std::cout << x_data[idx] << " ";
            }
        }
        std::cout << std::endl;
        
        // 检查 im2col 输出
        const OriginMat &col_mat_debug = static_cast<const OriginMat &>(*col);
        auto col_cpu = col_mat_debug.to_device(Device(DeviceType::kCPU, 0));
        const OriginMat &col_cpu_mat = static_cast<const OriginMat &>(*col_cpu);
        auto col_data = col_cpu_mat.to_vector<float>();
        std::cout << "\n=== DEBUG: im2col output (model.4.cv1.conv) ===" << std::endl;
        std::cout << "Shape: " << col_mat_debug.shape().to_string() << " (expected: {6400, C*KH*KW})" << std::endl;
        std::cout << "First 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), col_data.size()); ++i) {
            std::cout << col_data[i] << " ";
        }
        std::cout << std::endl;
        // 检查 col[0,0] 到 col[9,0] (第一列的前10个值)
        size_t col_cols = col_mat_debug.shape()[1];
        std::cout << "col[0,0] to col[9,0] (col=0): ";
        for (int row = 0; row < 10; ++row) {
            size_t idx = row * col_cols + 0;
            if (idx < col_data.size()) {
                std::cout << col_data[idx] << " ";
            }
        }
        std::cout << std::endl;
        // 分析：col[0,0] 对应 (n=0, oh=0, ow=0, c=0, kh=0, kw=0) -> h_idx=0, w_idx=0
        // col[1,0] 对应 (n=0, oh=0, ow=1, c=0, kh=0, kw=0) -> h_idx=0, w_idx=SW
        // 如果 SW=1，那么 col[0,0] 访问输入 [0,0,0,0]，col[1,0] 访问输入 [0,0,0,1]
        std::cout << "Analysis: col[0,0] accesses input [0,0,0,0], col[1,0] accesses input [0,0,0,SW]" << std::endl;
    }

    // 2. 将卷积核 reshape 为 (OC, C*KH*KW)
    auto W_reshaped = W.reshape(Shape{OC, C * KH * KW});
    const OriginMat &W_reshaped_mat = static_cast<const OriginMat &>(*W_reshaped);
    const OriginMat &col_mat = static_cast<const OriginMat &>(*col);

    // 3. 直接使用 cuBLAS GEMM 进行矩阵乘法: col @ W^T -> (N*OH*OW, OC)
    // col: (N*OH*OW, C*KH*KW), W: (OC, C*KH*KW), 需要计算 col @ W^T
    // 在列主序中，这等价于计算 C^T = W^T @ col^T
    size_t M = N * OH * OW;  // col 的行数
    size_t K = C * KH * KW;  // 公共维度
    size_t N_out = OC;       // 输出通道数
    
    Shape y_flat_shape{M, N_out};
    auto y_flat = std::make_unique<OriginMat>(y_flat_shape, x.dtype(), x.device());
    
    // 使用 GPU matmul 实现（避免 cuBLAS 的行列主序转换问题）
    // 计算: y_flat = col @ W^T
        auto W_T = W_reshaped_mat.transpose();
        const OriginMat &W_T_mat = static_cast<const OriginMat &>(*W_T);
    
    // 使用 GPU matmul
    if (col_mat.device().type() == DeviceType::kCUDA && W_T_mat.device().type() == DeviceType::kCUDA)
    {
        auto y_flat_matmul = cuda::matmul(col_mat, W_T_mat);
        y_flat = std::unique_ptr<OriginMat>(static_cast<OriginMat*>(y_flat_matmul.release()));
    }
    else
    {
        // 回退到 CPU matmul
        auto y_flat_matmul = col_mat.matmul(W_T_mat);
        y_flat = std::unique_ptr<OriginMat>(static_cast<OriginMat*>(y_flat_matmul.release()));
    }

    // 调试：检查 matmul 的输出（仅对 model.0.conv: N=4, C=3, OC=16, OH=160, OW=160, stride=2）
    if (N == 4 && C == 3 && OC == 16 && OH == 160 && OW == 160 && stride.first == 2 && stride.second == 2) {
        const OriginMat &y_flat_mat = static_cast<const OriginMat &>(*y_flat);
        auto y_flat_cpu = y_flat_mat.to_device(Device(DeviceType::kCPU, 0));
        const OriginMat &y_flat_cpu_mat = static_cast<const OriginMat &>(*y_flat_cpu);
        auto y_flat_data = y_flat_cpu_mat.to_vector<float>();
        std::cout << "\n=== DEBUG: matmul output (model.0.conv) ===" << std::endl;
        std::cout << "Shape: " << y_flat_mat.shape().to_string() << " (expected: {102400, 16})" << std::endl;
        std::cout << "First 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), y_flat_data.size()); ++i) {
            std::cout << y_flat_data[i] << " ";
        }
        std::cout << std::endl;
        // 检查 matmul[0,0] 到 matmul[9,0] (col=0)
        std::cout << "matmul[0,0] to matmul[9,0] (col=0): ";
        for (int row = 0; row < 10; ++row) {
            size_t idx = row * OC + 0;
            if (idx < y_flat_data.size()) {
                std::cout << y_flat_data[idx] << " ";
            }
        }
        std::cout << std::endl;
    }
    
    // 调试：检查 matmul 的输出（仅对 model.1.conv: N=4, C=16, OC=32, OH=80, OW=80, stride=2）
    if (N == 4 && C == 16 && OC == 32 && OH == 80 && OW == 80 && stride.first == 2 && stride.second == 2) {
        const OriginMat &y_flat_mat = static_cast<const OriginMat &>(*y_flat);
        auto y_flat_cpu = y_flat_mat.to_device(Device(DeviceType::kCPU, 0));
        const OriginMat &y_flat_cpu_mat = static_cast<const OriginMat &>(*y_flat_cpu);
        auto y_flat_data = y_flat_cpu_mat.to_vector<float>();
        std::cout << "\n=== DEBUG: matmul output (model.1.conv) ===" << std::endl;
        std::cout << "Shape: " << y_flat_mat.shape().to_string() << " (expected: {25600, 32})" << std::endl;
        std::cout << "First 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), y_flat_data.size()); ++i) {
            std::cout << y_flat_data[i] << " ";
        }
        std::cout << std::endl;
        // 检查 matmul[0,0] 到 matmul[9,0] (col=0)
        std::cout << "matmul[0,0] to matmul[9,0] (col=0): ";
        for (int row = 0; row < 10; ++row) {
            size_t idx = row * OC + 0;
            if (idx < y_flat_data.size()) {
                std::cout << y_flat_data[idx] << " ";
            }
        }
        std::cout << std::endl;
    }
    
    // 调试：检查 matmul 的输出（仅对 model.2.cv1.conv: N=4, C=32, OC=16, OH=80, OW=80, kernel=1x1）
    if (N == 4 && C == 32 && OC == 16 && OH == 80 && OW == 80 && KH == 1 && KW == 1 && pad.first == 0 && pad.second == 0) {
        const OriginMat &y_flat_mat = static_cast<const OriginMat &>(*y_flat);
        auto y_flat_cpu = y_flat_mat.to_device(Device(DeviceType::kCPU, 0));
        const OriginMat &y_flat_cpu_mat = static_cast<const OriginMat &>(*y_flat_cpu);
        auto y_flat_data = y_flat_cpu_mat.to_vector<float>();
        std::cout << "\n=== DEBUG: matmul output (model.2.cv1.conv) ===" << std::endl;
        std::cout << "Shape: " << y_flat_mat.shape().to_string() << " (expected: {25600, 16})" << std::endl;
        std::cout << "First 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), y_flat_data.size()); ++i) {
            std::cout << y_flat_data[i] << " ";
        }
        std::cout << std::endl;
        // 检查 matmul[0,0] 到 matmul[9,0] (col=0)
        std::cout << "matmul[0,0] to matmul[9,0] (col=0): ";
        for (int row = 0; row < 10; ++row) {
            size_t idx = row * OC + 0;
            if (idx < y_flat_data.size()) {
                std::cout << y_flat_data[idx] << " ";
            }
        }
        std::cout << std::endl;
    }
    
    // 调试：检查 matmul 的输出（仅对 model.2.cv2.conv: N=4, C=32, OC=16, OH=80, OW=80, kernel=1x1）
    if (N == 4 && C == 32 && OC == 16 && OH == 80 && OW == 80 && KH == 1 && KW == 1) {
        const OriginMat &y_flat_mat = static_cast<const OriginMat &>(*y_flat);
        auto y_flat_cpu = y_flat_mat.to_device(Device(DeviceType::kCPU, 0));
        const OriginMat &y_flat_cpu_mat = static_cast<const OriginMat &>(*y_flat_cpu);
        auto y_flat_data = y_flat_cpu_mat.to_vector<float>();
        std::cout << "\n=== DEBUG: matmul output (model.2.cv2.conv) ===" << std::endl;
        std::cout << "Shape: " << y_flat_mat.shape().to_string() << " (expected: {25600, 16})" << std::endl;
        std::cout << "First 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), y_flat_data.size()); ++i) {
            std::cout << y_flat_data[i] << " ";
        }
        std::cout << std::endl;
        // 检查 matmul[0,0] 到 matmul[9,0] (col=0)
        std::cout << "matmul[0,0] to matmul[9,0] (col=0): ";
        for (int row = 0; row < 10; ++row) {
            size_t idx = row * OC + 0;
            if (idx < y_flat_data.size()) {
                std::cout << y_flat_data[idx] << " ";
            }
        }
        std::cout << std::endl;
    }
    
    // 调试：检查 matmul 的输出（仅对 model.2.cv3.conv: N=4, C=32, OC=32, OH=80, OW=80, kernel=1x1）
    if (N == 4 && C == 32 && OC == 32 && OH == 80 && OW == 80 && KH == 1 && KW == 1) {
        const OriginMat &y_flat_mat = static_cast<const OriginMat &>(*y_flat);
        auto y_flat_cpu = y_flat_mat.to_device(Device(DeviceType::kCPU, 0));
        const OriginMat &y_flat_cpu_mat = static_cast<const OriginMat &>(*y_flat_cpu);
        auto y_flat_data = y_flat_cpu_mat.to_vector<float>();
        std::cout << "\n=== DEBUG: matmul output (model.2.cv3.conv) ===" << std::endl;
        std::cout << "Shape: " << y_flat_mat.shape().to_string() << " (expected: {25600, 32})" << std::endl;
        std::cout << "First 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), y_flat_data.size()); ++i) {
            std::cout << y_flat_data[i] << " ";
        }
        std::cout << std::endl;
        // 检查 matmul[0,0] 到 matmul[9,0] (col=0)
        std::cout << "matmul[0,0] to matmul[9,0] (col=0): ";
        for (int row = 0; row < 10; ++row) {
            size_t idx = row * OC + 0;
            if (idx < y_flat_data.size()) {
                std::cout << y_flat_data[idx] << " ";
            }
        }
        std::cout << std::endl;
    }
    
    // 调试：检查 matmul 的输出（仅对 model.3.conv: N=4, C=32, OC=64, OH=40, OW=40, stride=2）
    if (N == 4 && C == 32 && OC == 64 && OH == 40 && OW == 40 && stride.first == 2 && stride.second == 2) {
        const OriginMat &y_flat_mat = static_cast<const OriginMat &>(*y_flat);
        auto y_flat_cpu = y_flat_mat.to_device(Device(DeviceType::kCPU, 0));
        const OriginMat &y_flat_cpu_mat = static_cast<const OriginMat &>(*y_flat_cpu);
        auto y_flat_data = y_flat_cpu_mat.to_vector<float>();
        std::cout << "\n=== DEBUG: matmul output (model.3.conv) ===" << std::endl;
        std::cout << "Shape: " << y_flat_mat.shape().to_string() << " (expected: {6400, 64})" << std::endl;
        std::cout << "First 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), y_flat_data.size()); ++i) {
            std::cout << y_flat_data[i] << " ";
        }
        std::cout << std::endl;
        // 检查 matmul[0,0] 到 matmul[9,0] (col=0)
        std::cout << "matmul[0,0] to matmul[9,0] (col=0): ";
        for (int row = 0; row < 10; ++row) {
            size_t idx = row * OC + 0;
            if (idx < y_flat_data.size()) {
                std::cout << y_flat_data[idx] << " ";
            }
        }
        std::cout << std::endl;
    }
    
    // 调试：检查 matmul 的输出（仅对 model.2.cv2.conv: N=4, C=16, OC=16, OH=80, OW=80）
    if (N == 4 && C == 16 && OC == 16 && OH == 80 && OW == 80) {
        const OriginMat &y_flat_mat = static_cast<const OriginMat &>(*y_flat);
        auto y_flat_cpu = y_flat_mat.to_device(Device(DeviceType::kCPU, 0));
        const OriginMat &y_flat_cpu_mat = static_cast<const OriginMat &>(*y_flat_cpu);
        auto y_flat_data = y_flat_cpu_mat.to_vector<float>();
        std::cout << "\n=== DEBUG: matmul output (model.2.cv2.conv) ===" << std::endl;
        std::cout << "Shape: " << y_flat_mat.shape().to_string() << " (expected: {25600, 16})" << std::endl;
        std::cout << "First 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), y_flat_data.size()); ++i) {
            std::cout << y_flat_data[i] << " ";
        }
        std::cout << std::endl;
        // 检查 matmul[0,0] 到 matmul[9,0] (col=0)
        std::cout << "matmul[0,0] to matmul[9,0] (col=0): ";
        for (int row = 0; row < 10; ++row) {
            size_t idx = row * OC + 0;
            if (idx < y_flat_data.size()) {
                std::cout << y_flat_data[idx] << " ";
            }
        }
        std::cout << std::endl;
        // 检查 matmul[9,0] 到 matmul[9,15] (row=9, all cols)
        std::cout << "matmul[9,0] to matmul[9,15] (row=9, all cols): ";
        for (int col = 0; col < 16; ++col) {
            size_t idx = 9 * OC + col;
            if (idx < y_flat_data.size()) {
                std::cout << y_flat_data[idx] << " ";
            }
        }
        std::cout << std::endl;
    }
    
    // 调试：检查 matmul 的输出（仅对 model.4.m.1.cv1.conv: N=4, C=32, OC=32, OH=40, OW=40, kernel=1x1）
    if (N == 4 && C == 32 && OC == 32 && OH == 40 && OW == 40 && KH == 1 && KW == 1) {
        const OriginMat &y_flat_mat = static_cast<const OriginMat &>(*y_flat);
        auto y_flat_cpu = y_flat_mat.to_device(Device(DeviceType::kCPU, 0));
        const OriginMat &y_flat_cpu_mat = static_cast<const OriginMat &>(*y_flat_cpu);
        auto y_flat_data = y_flat_cpu_mat.to_vector<float>();
        std::cout << "\n=== DEBUG: matmul output (model.4.m.1.cv1.conv) ===" << std::endl;
        std::cout << "Shape: " << y_flat_mat.shape().to_string() << " (expected: {6400, 32})" << std::endl;
        std::cout << "First 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), y_flat_data.size()); ++i) {
            std::cout << y_flat_data[i] << " ";
        }
        std::cout << std::endl;
        // 检查 matmul[0,0] 到 matmul[9,0] (col=0)
        std::cout << "matmul[0,0] to matmul[9,0] (col=0): ";
        for (int row = 0; row < 10; ++row) {
            size_t idx = row * OC + 0;
            if (idx < y_flat_data.size()) {
                std::cout << y_flat_data[idx] << " ";
            }
        }
        std::cout << std::endl;
    }
    
    // 调试：检查 matmul 的输出（仅对 model.4.m.1.cv2.conv: N=4, C=32, OC=32, OH=40, OW=40, kernel=3x3）
    if (N == 4 && C == 32 && OC == 32 && OH == 40 && OW == 40 && KH == 3 && KW == 3) {
        const OriginMat &y_flat_mat = static_cast<const OriginMat &>(*y_flat);
        auto y_flat_cpu = y_flat_mat.to_device(Device(DeviceType::kCPU, 0));
        const OriginMat &y_flat_cpu_mat = static_cast<const OriginMat &>(*y_flat_cpu);
        auto y_flat_data = y_flat_cpu_mat.to_vector<float>();
        std::cout << "\n=== DEBUG: matmul output (model.4.m.1.cv2.conv) ===" << std::endl;
        std::cout << "Shape: " << y_flat_mat.shape().to_string() << " (expected: {6400, 32})" << std::endl;
        std::cout << "First 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), y_flat_data.size()); ++i) {
            std::cout << y_flat_data[i] << " ";
        }
        std::cout << std::endl;
        // 检查 matmul[0,0] 到 matmul[9,0] (col=0)
        std::cout << "matmul[0,0] to matmul[9,0] (col=0): ";
        for (int row = 0; row < 10; ++row) {
            size_t idx = row * OC + 0;
            if (idx < y_flat_data.size()) {
                std::cout << y_flat_data[idx] << " ";
            }
        }
        std::cout << std::endl;
    }
    
    // 调试：检查 matmul 的输出（仅对 model.4.cv1.conv: N=4, C=64, OC=32, OH=40, OW=40）
    if (N == 4 && C == 64 && OC == 32 && OH == 40 && OW == 40) {
        const OriginMat &y_flat_mat = static_cast<const OriginMat &>(*y_flat);
        auto y_flat_cpu = y_flat_mat.to_device(Device(DeviceType::kCPU, 0));
        const OriginMat &y_flat_cpu_mat = static_cast<const OriginMat &>(*y_flat_cpu);
        auto y_flat_data = y_flat_cpu_mat.to_vector<float>();
        std::cout << "\n=== DEBUG: matmul output (model.4.cv1.conv) ===" << std::endl;
        std::cout << "Shape: " << y_flat_mat.shape().to_string() << " (expected: {6400, 32})" << std::endl;
        std::cout << "First 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), y_flat_data.size()); ++i) {
            std::cout << y_flat_data[i] << " ";
        }
        std::cout << std::endl;
        // 检查 matmul[0,0], matmul[1,0], matmul[2,0], ..., matmul[9,0]
        std::cout << "matmul[0,0] to matmul[9,0] (col=0): ";
        for (int row = 0; row < 10; ++row) {
            size_t idx = row * OC + 0;
            if (idx < y_flat_data.size()) {
                std::cout << y_flat_data[idx] << " ";
            }
        }
        std::cout << std::endl;
        // 检查 matmul[0,1], matmul[1,1], matmul[2,1], ..., matmul[9,1]
        std::cout << "matmul[0,1] to matmul[9,1] (col=1): ";
        for (int row = 0; row < 10; ++row) {
            size_t idx = row * OC + 1;
            if (idx < y_flat_data.size()) {
                std::cout << y_flat_data[idx] << " ";
            }
        }
        std::cout << std::endl;
    }

    // 4. 添加偏置（如果存在）
    if (b != nullptr)
    {
        // 广播偏置: (OC,) -> (N*OH*OW, OC)
        auto b_broadcast = b->broadcast_to(Shape{N * static_cast<size_t>(OH) * static_cast<size_t>(OW), OC});
        const OriginMat &y_flat_mat = static_cast<const OriginMat &>(*y_flat);
        const OriginMat &b_broadcast_mat = static_cast<const OriginMat &>(*b_broadcast);
        auto y_flat_with_bias = y_flat_mat.operator+(b_broadcast_mat);
        y_flat = std::unique_ptr<OriginMat>(static_cast<OriginMat*>(y_flat_with_bias.release()));
    }

    // 5. Reshape 并转置: (N*OH*OW, OC) -> (N, OH, OW, OC) -> (N, OC, OH, OW)
    const OriginMat &y_flat_mat = static_cast<const OriginMat &>(*y_flat);
    auto y_reshaped = y_flat_mat.reshape(Shape{N, static_cast<size_t>(OH), static_cast<size_t>(OW), OC});
    
    // 调试：检查 reshape 后的数据（仅对 model.4.cv1.conv: N=4, C=64, OC=32, OH=40, OW=40）
    if (N == 4 && C == 64 && OC == 32 && OH == 40 && OW == 40) {
        auto y_reshaped_cpu = y_reshaped->to_device(Device(DeviceType::kCPU, 0));
        const OriginMat &y_reshaped_cpu_mat = static_cast<const OriginMat &>(*y_reshaped_cpu);
        auto y_reshaped_data = y_reshaped_cpu_mat.to_vector<float>();
        std::cout << "\n=== DEBUG: reshape output (model.4.cv1.conv) ===" << std::endl;
        std::cout << "Shape: " << y_reshaped->shape().to_string() << " (expected: {4, 40, 40, 32})" << std::endl;
        std::cout << "First 20 values: ";
        for (size_t i = 0; i < std::min(size_t(20), y_reshaped_data.size()); ++i) {
            std::cout << y_reshaped_data[i] << " ";
        }
        std::cout << std::endl;
        // 检查 [0,0,0,0], [0,0,0,1], [0,0,0,2], ..., [0,0,0,9]
        std::cout << "[0,0,0,0] to [0,0,0,9]: ";
        for (int ow = 0; ow < 10; ++ow) {
            size_t idx = 0 * OH*OW*OC + 0 * OW*OC + ow * OC + 0;
            if (idx < y_reshaped_data.size()) {
                std::cout << y_reshaped_data[idx] << " ";
            }
        }
        std::cout << std::endl;
    }

    // 使用 CUDA kernel 进行转置
    device_common::TypeDispatcher::dispatch_void(x.dtype(), [&]<typename T>() {
        const OriginMat &y_reshaped_mat = static_cast<const OriginMat &>(*y_reshaped);
        const T *src_data = y_reshaped_mat.data_ptr<T>();
        T *dst_data = result->data_ptr<T>();

        size_t total_elements = N * OC * OH * OW;
        int threads_per_block = 256;
        int num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

        transpose_conv_output_kernel<T><<<num_blocks, threads_per_block>>>(src_data, dst_data, N, OC, OH, OW);
    });

    CUDA_CHECK_ASYNC();

    return result;
}

std::vector<std::unique_ptr<Mat>> conv2d_backward(const OriginMat &gy, const OriginMat &x, const OriginMat &W,
                                                    const OriginMat *b, std::pair<int, int> stride,
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

    size_t N = x.shape()[0];
    size_t C = x.shape()[1];
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
    auto gy_T = gy_reshaped->transpose();
    const OriginMat &gy_T_mat = static_cast<const OriginMat &>(*gy_T);

    // 使用 im2col 将输入转换为列矩阵
    auto col = im2col_impl(x, std::make_pair(static_cast<int>(KH), static_cast<int>(KW)), stride, pad, true);
    // col 形状: (N*OH*OW, C*KH*KW)
    const OriginMat &col_mat = static_cast<const OriginMat &>(*col);

    // gW = gy_T @ col，然后 reshape 为 (OC, C, KH, KW)
    auto gW_flat = gy_T_mat.matmul(col_mat);
    // gW_flat 形状: (OC, C*KH*KW)
    const OriginMat &gW_flat_mat = static_cast<const OriginMat &>(*gW_flat);
    auto gW = gW_flat_mat.reshape(Shape{OC, C, KH, KW});
    grads.push_back(std::move(gW));

    // 2. 计算 gb (偏置梯度)
    if (b != nullptr)
    {
        // gb = gy.sum(axis=(0, 2, 3))
        // gy 形状: (N, OC, OH, OW)
        // 方法：依次对维度2(OH), 2(OW), 0(N)求和
        auto gy_sum_h = gy.sum(2);  // sum over OH, shape: (N, OC, OW)
        const OriginMat &gy_sum_h_mat = static_cast<const OriginMat &>(*gy_sum_h);
        auto gy_sum_w = gy_sum_h_mat.sum(2);  // sum over OW, shape: (N, OC)
        const OriginMat &gy_sum_w_mat = static_cast<const OriginMat &>(*gy_sum_w);
        auto gb_mat = gy_sum_w_mat.sum(0);  // sum over N, shape: (OC,)
        auto gb = std::unique_ptr<OriginMat>(static_cast<OriginMat *>(gb_mat.release()));
        // 强制reshape到(OC,)，确保形状正确
        auto gb_reshaped = gb->reshape(Shape{OC});
        grads.push_back(std::move(gb_reshaped));
    }

    // 3. 计算 gx (输入梯度)
    // 将卷积核 reshape
    auto W_reshaped = W.reshape(Shape{OC, C * KH * KW});
    // gy_reshaped 形状: (N*OH*OW, OC)
    const OriginMat &gy_reshaped_mat = static_cast<const OriginMat &>(*gy_reshaped);
    const OriginMat &W_reshaped_mat = static_cast<const OriginMat &>(*W_reshaped);
    // gx_col = gy_reshaped @ W_reshaped
    auto gx_col = gy_reshaped_mat.matmul(W_reshaped_mat);
    // gx_col 形状: (N*OH*OW, C*KH*KW)

    // 使用 col2im 转换回图像形状
    const OriginMat &gx_col_mat = static_cast<const OriginMat &>(*gx_col);
    auto gx = col2im_impl(gx_col_mat, x.shape(), std::make_pair(static_cast<int>(KH), static_cast<int>(KW)), stride, pad, true);
    grads.insert(grads.begin(), std::move(gx));  // 插入到开头，顺序为 {gx, gW, [gb]}

    return grads;
}

}  // namespace cuda
}  // namespace origin

