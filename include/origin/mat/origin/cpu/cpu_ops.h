#ifndef __ORIGIN_DL_CPU_OPS_H__
#define __ORIGIN_DL_CPU_OPS_H__

#include <memory>
#include "origin/mat/origin/../basic_types.h"
#include "origin/mat/origin/origin_mat.h"

namespace origin
{
namespace cpu
{

// === 基础运算 ===
std::unique_ptr<Mat> add(const OriginMat &a, const OriginMat &b);
std::unique_ptr<Mat> subtract(const OriginMat &a, const OriginMat &b);
std::unique_ptr<Mat> multiply(const OriginMat &a, const OriginMat &b);
std::unique_ptr<Mat> divide(const OriginMat &a, const OriginMat &b);
std::unique_ptr<Mat> matmul(const OriginMat &a, const OriginMat &b);

std::unique_ptr<Mat> negate(const OriginMat &mat);

// === 数学函数 ===
std::unique_ptr<Mat> exp(const OriginMat &mat);
std::unique_ptr<Mat> log(const OriginMat &mat);
std::unique_ptr<Mat> sqrt(const OriginMat &mat);
std::unique_ptr<Mat> square(const OriginMat &mat);
std::unique_ptr<Mat> pow(const OriginMat &mat, const Scalar &exponent);

// === 统计函数 ===
std::unique_ptr<Mat> sum(const OriginMat &mat, int axis);

// === 形状操作 ===
std::unique_ptr<Mat> reshape(const OriginMat &mat, const Shape &new_shape);
std::unique_ptr<Mat> transpose(const OriginMat &mat);
std::unique_ptr<Mat> broadcast_to(const OriginMat &mat, const Shape &target_shape);
std::unique_ptr<Mat> sum_to(const OriginMat &mat, const Shape &target_shape);

// === 类型转换 ===
std::unique_ptr<Mat> convert_datatype(const OriginMat &mat, DataType target_type);

// === 卷积相关 ===
// 注意：im2col 和 col2im 是 conv2d 的内部实现，仅供 OriginMat::im2col/col2im 使用
std::unique_ptr<Mat> im2col(const OriginMat &img, std::pair<int, int> kernel_size, std::pair<int, int> stride,
                            std::pair<int, int> pad, bool to_matrix);
std::unique_ptr<Mat> col2im(const OriginMat &col, const Shape &input_shape, std::pair<int, int> kernel_size,
                            std::pair<int, int> stride, std::pair<int, int> pad, bool to_matrix);

/**
 * @brief 完整的卷积操作（前向传播）
 * @param x 输入张量 (N, C, H, W)
 * @param W 卷积核 (OC, C, KH, KW)
 * @param b 偏置 (OC,)，可选，如果为 nullptr 则不添加偏置
 * @param stride 步长 (SH, SW)
 * @param pad 填充 (PH, PW)
 * @return 输出张量 (N, OC, OH, OW)
 */
std::unique_ptr<Mat> conv2d(const OriginMat &x, const OriginMat &W, const OriginMat *b, std::pair<int, int> stride,
                            std::pair<int, int> pad);

/**
 * @brief 完整的卷积反向传播
 * @param gy 输出梯度 (N, OC, OH, OW)
 * @param x 输入张量 (N, C, H, W)
 * @param W 卷积核 (OC, C, KH, KW)
 * @param b 偏置 (OC,)，可选
 * @param stride 步长 (SH, SW)
 * @param pad 填充 (PH, PW)
 * @return 梯度向量：{gx, gW, [gb]}
 */
std::vector<std::unique_ptr<Mat>> conv2d_backward(const OriginMat &gy, const OriginMat &x, const OriginMat &W,
                                                    const OriginMat *b, std::pair<int, int> stride,
                                                    std::pair<int, int> pad);

/**
 * @brief 平均池化操作（前向传播）
 * @param x 输入张量 (N, C, H, W)
 * @param kernel_size 池化核大小 (KH, KW)
 * @param stride 步长 (SH, SW)
 * @param pad 填充 (PH, PW)
 * @return 输出张量 (N, C, OH, OW)
 */
std::unique_ptr<Mat> avg_pool2d(const OriginMat &x, std::pair<int, int> kernel_size, std::pair<int, int> stride,
                                std::pair<int, int> pad);

/**
 * @brief 平均池化反向传播
 * @param gy 输出梯度 (N, C, OH, OW)
 * @param x 输入张量 (N, C, H, W)
 * @param kernel_size 池化核大小 (KH, KW)
 * @param stride 步长 (SH, SW)
 * @param pad 填充 (PH, PW)
 * @return 输入梯度 (N, C, H, W)
 */
std::unique_ptr<Mat> avg_pool2d_backward(const OriginMat &gy, const OriginMat &x, std::pair<int, int> kernel_size,
                                         std::pair<int, int> stride, std::pair<int, int> pad);

/**
 * @brief 最大池化操作（前向传播）
 * @param x 输入张量 (N, C, H, W)
 * @param kernel_size 池化核大小 (KH, KW)
 * @param stride 步长 (SH, SW)
 * @param pad 填充 (PH, PW)
 * @param indices 输出参数：保存每个最大值在窗口内的索引
 * @return 输出张量 (N, C, OH, OW)
 */
std::unique_ptr<Mat> max_pool2d(const OriginMat &x, std::pair<int, int> kernel_size, std::pair<int, int> stride,
                                std::pair<int, int> pad, std::vector<size_t> &indices);

/**
 * @brief 最大池化反向传播
 * @param gy 输出梯度 (N, C, OH, OW)
 * @param x 输入张量 (N, C, H, W)
 * @param kernel_size 池化核大小 (KH, KW)
 * @param stride 步长 (SH, SW)
 * @param pad 填充 (PH, PW)
 * @param indices 前向传播时保存的索引
 * @return 输入梯度 (N, C, H, W)
 */
std::unique_ptr<Mat> max_pool2d_backward(const OriginMat &gy, const OriginMat &x, std::pair<int, int> kernel_size,
                                         std::pair<int, int> stride, std::pair<int, int> pad,
                                         const std::vector<size_t> &indices);

}  // namespace cpu
}  // namespace origin

#endif  // __ORIGIN_DL_CPU_OPS_H__
