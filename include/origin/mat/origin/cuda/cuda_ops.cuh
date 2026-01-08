#ifndef __ORIGIN_DL_CUDA_OPS_H__
#define __ORIGIN_DL_CUDA_OPS_H__

#include <memory>
#include "../../scalar.h"
#include "../origin_mat.h"

namespace origin
{
namespace cuda
{

/**
 * @brief CUDA运算接口声明
 * @details 提供所有CUDA算子的统一接口
 */

// ============================================================================
// 基础二元运算算子
// ============================================================================

/**
 * @brief CUDA加法算子
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 加法结果矩阵
 */
std::unique_ptr<Mat> add(const origin::OriginMat &a, const origin::OriginMat &b);

/**
 * @brief CUDA减法算子
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 减法结果矩阵
 */
std::unique_ptr<Mat> subtract(const origin::OriginMat &a, const origin::OriginMat &b);

/**
 * @brief CUDA乘法算子
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 乘法结果矩阵
 */
std::unique_ptr<Mat> multiply(const origin::OriginMat &a, const origin::OriginMat &b);

/**
 * @brief CUDA除法算子
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 除法结果矩阵
 */
std::unique_ptr<Mat> divide(const origin::OriginMat &a, const origin::OriginMat &b);

// ============================================================================
// 一元运算算子
// ============================================================================

/**
 * @brief CUDA指数算子
 * @param mat 输入矩阵
 * @return 指数运算结果矩阵
 */
std::unique_ptr<Mat> exp(const origin::OriginMat &mat);

/**
 * @brief CUDA对数算子
 * @param mat 输入矩阵
 * @return 对数运算结果矩阵
 */
std::unique_ptr<Mat> log(const origin::OriginMat &mat);

/**
 * @brief CUDA平方根算子
 * @param mat 输入矩阵
 * @return 平方根运算结果矩阵
 */
std::unique_ptr<Mat> sqrt(const origin::OriginMat &mat);

/**
 * @brief CUDA平方算子
 * @param mat 输入矩阵
 * @return 平方运算结果矩阵
 */
std::unique_ptr<Mat> square(const origin::OriginMat &mat);

/**
 * @brief CUDA取负算子
 * @param mat 输入矩阵
 * @return 取负运算结果矩阵
 */
std::unique_ptr<Mat> negate(const origin::OriginMat &mat);

/**
 * @brief CUDA ReLU 激活函数算子
 * @param mat 输入矩阵
 * @return ReLU 运算结果矩阵，y = max(0, x)
 */
std::unique_ptr<Mat> relu(const origin::OriginMat &mat);

// ============================================================================
// 形状操作算子
// ============================================================================

/**
 * @brief CUDA重塑算子
 * @param mat 输入矩阵
 * @param new_shape 新的形状
 * @return 重塑后的矩阵
 */
std::unique_ptr<Mat> reshape(const origin::OriginMat &mat, const Shape &new_shape);

/**
 * @brief CUDA转置算子
 * @param mat 输入矩阵
 * @return 转置后的矩阵
 */
std::unique_ptr<Mat> transpose(const origin::OriginMat &mat);

// ============================================================================
// 高级运算算子
// ============================================================================

/**
 * @brief CUDA矩阵乘法算子
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @return 矩阵乘法结果矩阵
 */
std::unique_ptr<Mat> matmul(const OriginMat &a, const OriginMat &b);

/**
 * @brief CUDA求和算子
 * @param mat 输入矩阵
 * @param axis 求和轴，-1表示所有元素求和
 * @return 求和结果矩阵
 */
std::unique_ptr<Mat> sum(const OriginMat &mat, int axis = -1);

/**
 * @brief CUDA幂运算算子（标量）
 * @param mat 输入矩阵
 * @param exponent 指数
 * @return 幂运算结果矩阵
 */
std::unique_ptr<Mat> pow(const OriginMat &mat, const Scalar &exponent);

/**
 * @brief CUDA广播算子
 * @param mat 输入矩阵
 * @param target_shape 目标形状
 * @return 广播后的矩阵
 */
std::unique_ptr<Mat> broadcast_to(const OriginMat &mat, const Shape &target_shape);

/**
 * @brief CUDA sum_to算子
 * @param mat 输入矩阵
 * @param target_shape 目标形状
 * @return sum_to结果矩阵
 */
std::unique_ptr<Mat> sum_to(const OriginMat &mat, const Shape &target_shape);

/**
 * @brief CUDA类型转换算子
 * @param mat 输入矩阵
 * @param target_type 目标数据类型
 * @return 类型转换后的矩阵
 */
std::unique_ptr<Mat> convert_datatype(const OriginMat &mat, DataType target_type);

// ============================================================================
// 卷积相关操作（内部实现，仅供 OriginMat::im2col/col2im 使用）
// ============================================================================

/**
 * @brief CUDA im2col：将图像转换为列矩阵
 * @param img 输入图像，形状为 (N, C, H, W)
 * @param kernel_size 卷积核大小
 * @param stride 步长
 * @param pad 填充
 * @param to_matrix 是否转换为矩阵形式
 * @return 列矩阵
 */
std::unique_ptr<Mat> im2col(const OriginMat &img,
                            std::pair<int, int> kernel_size,
                            std::pair<int, int> stride,
                            std::pair<int, int> pad,
                            bool to_matrix);

/**
 * @brief CUDA col2im：将列矩阵转换回图像形状
 * @param col 列矩阵
 * @param input_shape 原始输入形状 (N, C, H, W)
 * @param kernel_size 卷积核大小
 * @param stride 步长
 * @param pad 填充
 * @param to_matrix 是否从矩阵形式转换
 * @return 图像矩阵
 */
std::unique_ptr<Mat> col2im(const OriginMat &col,
                            const Shape &input_shape,
                            std::pair<int, int> kernel_size,
                            std::pair<int, int> stride,
                            std::pair<int, int> pad,
                            bool to_matrix);

/**
 * @brief CUDA conv2d：完整的二维卷积操作（前向传播）
 * @param x 输入张量 (N, C, H, W)
 * @param W 卷积核 (OC, C, KH, KW)
 * @param b 偏置 (OC,)，可选，如果为 nullptr 则不添加偏置
 * @param stride 步长 (SH, SW)
 * @param pad 填充 (PH, PW)
 * @return 输出张量 (N, OC, OH, OW)
 */
std::unique_ptr<Mat> conv2d(const OriginMat &x,
                            const OriginMat &W,
                            const OriginMat *b,
                            std::pair<int, int> stride,
                            std::pair<int, int> pad);

/**
 * @brief CUDA conv2d_backward：卷积反向传播
 * @param gy 输出梯度 (N, OC, OH, OW)
 * @param x 输入张量 (N, C, H, W)
 * @param W 卷积核 (OC, C, KH, KW)
 * @param b 偏置 (OC,)，可选
 * @param stride 步长 (SH, SW)
 * @param pad 填充 (PH, PW)
 * @return 梯度向量：{gx, gW, [gb]}
 */
std::vector<std::unique_ptr<Mat>> conv2d_backward(const OriginMat &gy,
                                                  const OriginMat &x,
                                                  const OriginMat &W,
                                                  const OriginMat *b,
                                                  std::pair<int, int> stride,
                                                  std::pair<int, int> pad);

/**
 * @brief CUDA avg_pool2d：平均池化操作（前向传播）
 * @param x 输入张量 (N, C, H, W)
 * @param kernel_size 池化核大小 (KH, KW)
 * @param stride 步长 (SH, SW)
 * @param pad 填充 (PH, PW)
 * @return 输出张量 (N, C, OH, OW)
 */
std::unique_ptr<Mat> avg_pool2d(const OriginMat &x,
                                std::pair<int, int> kernel_size,
                                std::pair<int, int> stride,
                                std::pair<int, int> pad);

/**
 * @brief CUDA avg_pool2d_backward：平均池化反向传播
 * @param gy 输出梯度 (N, C, OH, OW)
 * @param x 输入张量 (N, C, H, W)
 * @param kernel_size 池化核大小 (KH, KW)
 * @param stride 步长 (SH, SW)
 * @param pad 填充 (PH, PW)
 * @return 输入梯度 (N, C, H, W)
 */
std::unique_ptr<Mat> avg_pool2d_backward(const OriginMat &gy,
                                         const OriginMat &x,
                                         std::pair<int, int> kernel_size,
                                         std::pair<int, int> stride,
                                         std::pair<int, int> pad);

/**
 * @brief CUDA max_pool2d：最大池化操作（前向传播）
 * @param x 输入张量 (N, C, H, W)
 * @param kernel_size 池化核大小 (KH, KW)
 * @param stride 步长 (SH, SW)
 * @param pad 填充 (PH, PW)
 * @param indices 输出参数：保存每个最大值在窗口内的索引
 * @return 输出张量 (N, C, OH, OW)
 */
std::unique_ptr<Mat> max_pool2d(const OriginMat &x,
                                std::pair<int, int> kernel_size,
                                std::pair<int, int> stride,
                                std::pair<int, int> pad,
                                std::vector<size_t> &indices);

/**
 * @brief CUDA max_pool2d_backward：最大池化反向传播
 * @param gy 输出梯度 (N, C, OH, OW)
 * @param x 输入张量 (N, C, H, W)
 * @param kernel_size 池化核大小 (KH, KW)
 * @param stride 步长 (SH, SW)
 * @param pad 填充 (PH, PW)
 * @param indices 前向传播时保存的索引
 * @return 输入梯度 (N, C, H, W)
 */
std::unique_ptr<Mat> max_pool2d_backward(const OriginMat &gy,
                                         const OriginMat &x,
                                         std::pair<int, int> kernel_size,
                                         std::pair<int, int> stride,
                                         std::pair<int, int> pad,
                                         const std::vector<size_t> &indices);

}  // namespace cuda
}  // namespace origin

#endif  // __ORIGIN_DL_CUDA_OPS_H__
