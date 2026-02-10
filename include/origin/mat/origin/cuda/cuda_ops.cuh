/**
 * @file cuda_ops.cuh
 * @brief CUDA 算子接口声明头文件
 * 
 * ============================================================================
 * 文件功能说明
 * ============================================================================
 * 
 * 本文件是 CUDA 算子的统一接口声明层，提供所有 CUDA 算子的接口声明。
 * 
 * 架构层次：
 * - origin_mat.cpp (封装层)
 *   ↓ 包含
 * - cuda_ops.cuh (本文件：所有 CUDA 算子的接口声明)
 *   ↓ 声明
 * - cuda_ops.cu (非计算类算子实现：clone、index_put)
 * - add.cu, divide.cu 等 (计算类算子实现)
 *   ↓ 都包含
 * - cuda_kernels.cuh (kernel 定义，只在 .cu 文件中使用)
 * 
 * 使用说明：
 * - origin_mat.cpp 只需包含 cuda_ops.cuh，即可使用所有 CUDA 算子
 * - 所有 CUDA 算子的实现都在对应的 .cu 文件中
 * - cuda_kernels.cuh 只在 .cu 文件中被包含，用于 kernel 定义
 * 
 * ============================================================================
 */

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
 * @brief CUDA加法算子统一实现
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param out 输出矩阵指针，如果为nullptr则创建新矩阵，否则将结果写入out
 * @return 如果out==nullptr则返回新矩阵，否则返回nullptr（结果在out中）
 */
std::unique_ptr<Mat> add(const origin::OriginMat &a, const origin::OriginMat &b, origin::OriginMat *out = nullptr);

/**
 * @brief CUDA减法算子统一实现
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param out 输出矩阵指针，如果为nullptr则创建新矩阵，否则将结果写入out
 * @return 如果out==nullptr则返回新矩阵，否则返回nullptr（结果在out中）
 */
std::unique_ptr<Mat> subtract(const origin::OriginMat &a, const origin::OriginMat &b, origin::OriginMat *out = nullptr);

/**
 * @brief CUDA乘法算子统一实现
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param out 输出矩阵指针，如果为nullptr则创建新矩阵，否则将结果写入out
 * @return 如果out==nullptr则返回新矩阵，否则返回nullptr（结果在out中）
 */
std::unique_ptr<Mat> multiply(const origin::OriginMat &a, const origin::OriginMat &b, origin::OriginMat *out = nullptr);

/**
 * @brief CUDA除法算子统一实现
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param out 输出矩阵指针，如果为nullptr则创建新矩阵，否则将结果写入out
 * @return 如果out==nullptr则返回新矩阵，否则返回nullptr（结果在out中）
 */
std::unique_ptr<Mat> divide(const origin::OriginMat &a, const origin::OriginMat &b, origin::OriginMat *out = nullptr);

// ============================================================================
// 比较运算算子
// ============================================================================

/**
 * @brief CUDA等于比较算子
 * @param mat 输入矩阵
 * @param threshold 比较阈值，可以是标量（shape为{}或{1}）或与输入相同形状的张量
 * @param out 输出矩阵指针，如果为nullptr则创建新矩阵，否则将结果写入out
 * @return 如果out==nullptr则返回新矩阵，否则返回nullptr（结果在out中）
 */
std::unique_ptr<Mat> eq(const origin::OriginMat &mat, const origin::OriginMat &threshold, origin::OriginMat *out = nullptr);

/**
 * @brief CUDA不等于比较算子
 */
std::unique_ptr<Mat> ne(const origin::OriginMat &mat, const origin::OriginMat &threshold, origin::OriginMat *out = nullptr);

/**
 * @brief CUDA小于比较算子
 */
std::unique_ptr<Mat> lt(const origin::OriginMat &mat, const origin::OriginMat &threshold, origin::OriginMat *out = nullptr);

/**
 * @brief CUDA小于等于比较算子
 */
std::unique_ptr<Mat> le(const origin::OriginMat &mat, const origin::OriginMat &threshold, origin::OriginMat *out = nullptr);

/**
 * @brief CUDA大于比较算子
 */
std::unique_ptr<Mat> gt(const origin::OriginMat &mat, const origin::OriginMat &threshold, origin::OriginMat *out = nullptr);

/**
 * @brief CUDA大于等于比较算子
 */
std::unique_ptr<Mat> ge(const origin::OriginMat &mat, const origin::OriginMat &threshold, origin::OriginMat *out = nullptr);

// ============================================================================
// 一元运算算子
// ============================================================================

/**
 * @brief CUDA指数算子统一实现
 * @param mat 输入矩阵
 * @param out 输出矩阵指针，如果为nullptr则创建新矩阵，否则将结果写入out
 * @return 如果out==nullptr则返回新矩阵，否则返回nullptr（结果在out中）
 */
std::unique_ptr<Mat> exp(const origin::OriginMat &mat, origin::OriginMat *out = nullptr);

/**
 * @brief CUDA对数算子统一实现
 * @param mat 输入矩阵
 * @param out 输出矩阵指针，如果为nullptr则创建新矩阵，否则将结果写入out
 * @return 如果out==nullptr则返回新矩阵，否则返回nullptr（结果在out中）
 */
std::unique_ptr<Mat> log(const origin::OriginMat &mat, origin::OriginMat *out = nullptr);

/**
 * @brief CUDA平方根算子统一实现
 * @param mat 输入矩阵
 * @param out 输出矩阵指针，如果为nullptr则创建新矩阵，否则将结果写入out
 * @return 如果out==nullptr则返回新矩阵，否则返回nullptr（结果在out中）
 */
std::unique_ptr<Mat> sqrt(const origin::OriginMat &mat, origin::OriginMat *out = nullptr);

/**
 * @brief CUDA平方算子统一实现
 * @param mat 输入矩阵
 * @param out 输出矩阵指针，如果为nullptr则创建新矩阵，否则将结果写入out
 * @return 如果out==nullptr则返回新矩阵，否则返回nullptr（结果在out中）
 */
std::unique_ptr<Mat> square(const origin::OriginMat &mat, origin::OriginMat *out = nullptr);

/**
 * @brief CUDA取负算子统一实现
 * @param mat 输入矩阵
 * @param out 输出矩阵指针，如果为nullptr则创建新矩阵，否则将结果写入out
 * @return 如果out==nullptr则返回新矩阵，否则返回nullptr（结果在out中）
 */
std::unique_ptr<Mat> negate(const origin::OriginMat &mat, origin::OriginMat *out = nullptr);

/**
 * @brief CUDA ReLU 激活函数算子统一实现
 * @param mat 输入矩阵
 * @param out 输出矩阵指针，如果为nullptr则创建新矩阵，否则将结果写入out
 * @return 如果out==nullptr则返回新矩阵，否则返回nullptr（结果在out中）
 */
std::unique_ptr<Mat> relu(const origin::OriginMat &mat, origin::OriginMat *out = nullptr);

// ============================================================================
// 原地操作算子
// ============================================================================

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

/**
 * @brief CUDA permute：按照指定顺序重新排列张量的维度
 * @param mat 输入矩阵
 * @param dims 新的维度顺序，例如 {0, 2, 3, 1} 表示将维度 0,1,2,3 重新排列为 0,2,3,1
 * @return 重排后的矩阵
 */
std::unique_ptr<Mat> permute(const origin::OriginMat &mat, const std::vector<int> &dims);

/**
 * @brief CUDA cat算子实现
 * @param inputs 输入矩阵列表
 * @param dim 拼接维度
 * @return 拼接结果矩阵
 */
std::unique_ptr<Mat> cat(const std::vector<const origin::OriginMat *> &inputs, int dim);

/**
 * @brief CUDA split：将矩阵沿指定维度分割成多个矩阵（cat 的反向操作）
 * @param input 输入矩阵
 * @param split_sizes 沿 dim 维度的各段大小列表
 * @param dim 分割维度
 * @return 分割后的矩阵列表
 */
std::vector<std::unique_ptr<Mat>> split(const origin::OriginMat &input,
                                        const std::vector<size_t> &split_sizes,
                                        int dim);

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
 * @param keepdim 是否保持维度，默认为false
 * @return 求和结果矩阵
 */
std::unique_ptr<Mat> sum(const OriginMat &mat, int axis = -1, bool keepdim = false);

/**
 * @brief CUDA max：沿指定轴计算最大值
 * @param mat 输入矩阵
 * @param axis 计算轴，-1 表示所有元素
 * @return 最大值结果矩阵
 */
std::unique_ptr<Mat> max(const OriginMat &mat, int axis = -1);

/**
 * @brief CUDA幂运算算子（标量）
 * @param mat 输入矩阵
 * @param exponent 指数
 * @return 幂运算结果矩阵
 */
std::unique_ptr<Mat> pow(const OriginMat &mat, const Scalar &exponent, OriginMat *out = nullptr);

/**
 * @brief CUDA幂函数原地操作算子
 * @param mat 输入/输出矩阵（原地修改）
 * @param exponent 指数
 */
void pow_inplace(OriginMat &mat, const Scalar &exponent);

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
 * @brief CUDA adaptive_avg_pool2d：自适应平均池化操作（前向传播）
 * @param x 输入张量 (N, C, H, W)
 * @param output_size 输出尺寸 (OH, OW)
 * @return 输出张量 (N, C, OH, OW)
 */
std::unique_ptr<Mat> adaptive_avg_pool2d(const OriginMat &x, std::pair<int, int> output_size);

/**
 * @brief CUDA adaptive_avg_pool2d_backward：自适应平均池化反向传播
 * @param gy 输出梯度 (N, C, OH, OW)
 * @param x 输入张量 (N, C, H, W)
 * @param output_size 输出尺寸 (OH, OW)
 * @return 输入梯度 (N, C, H, W)
 */
std::unique_ptr<Mat> adaptive_avg_pool2d_backward(const OriginMat &gy,
                                                  const OriginMat &x,
                                                  std::pair<int, int> output_size);

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

// ============================================================================
// 归一化相关操作
// ============================================================================

/**
 * @brief BatchNorm 前向传播结果结构体
 */
struct BatchNormForwardResult
{
    std::unique_ptr<Mat> y;       // 输出
    std::unique_ptr<Mat> mean;    // 当前 batch 的均值
    std::unique_ptr<Mat> var;     // 当前 batch 的方差
    std::unique_ptr<Mat> x_norm;  // 归一化后的 x
};

/**
 * @brief CUDA batch_norm_forward：BatchNorm 前向传播（返回所有中间结果）
 * @note 只支持浮点类型（float32 或 float64），与 PyTorch 行为一致
 * @param x 输入张量（必须是 float32 或 float64）
 * @param gamma 缩放参数 (weight)，形状为 (C,)，必须与 x 相同的浮点类型
 * @param beta 偏移参数 (bias)，形状为 (C,)，必须与 x 相同的浮点类型
 * @param running_mean 运行均值，形状为 (C,)，必须与 x 相同的浮点类型
 * @param running_var 运行方差，形状为 (C,)，必须与 x 相同的浮点类型
 * @param training 是否为训练模式
 * @param eps 数值稳定性参数
 * @param num_dims 输入张量的总维度数：2=(N,C), 4=(N,C,H,W)
 * @return BatchNormForwardResult 包含输出和中间结果
 */
BatchNormForwardResult batch_norm_forward(const OriginMat &x,
                                          const OriginMat &gamma,
                                          const OriginMat &beta,
                                          const OriginMat &running_mean,
                                          const OriginMat &running_var,
                                          bool training,
                                          float eps,
                                          int num_dims);

/**
 * @brief CUDA batch_norm：BatchNorm 前向传播（只返回输出）
 * @note 只支持浮点类型（float32 或 float64），与 PyTorch 行为一致
 * @param x 输入张量（必须是 float32 或 float64）
 * @param gamma 缩放参数 (weight)，形状为 (C,)，必须与 x 相同的浮点类型
 * @param beta 偏移参数 (bias)，形状为 (C,)，必须与 x 相同的浮点类型
 * @param running_mean 运行均值，形状为 (C,)，必须与 x 相同的浮点类型
 * @param running_var 运行方差，形状为 (C,)，必须与 x 相同的浮点类型
 * @param training 是否为训练模式
 * @param eps 数值稳定性参数
 * @param momentum 动量参数（未使用，为了接口一致性保留）
 * @param num_dims 输入张量的总维度数：2=(N,C), 4=(N,C,H,W)
 * @return 输出张量，形状与输入相同
 */
std::unique_ptr<Mat> batch_norm(const OriginMat &x,
                                const OriginMat &gamma,
                                const OriginMat &beta,
                                const OriginMat &running_mean,
                                const OriginMat &running_var,
                                bool training,
                                float eps,
                                float momentum,
                                int num_dims);

/**
 * @brief CUDA batch_norm_backward：BatchNorm 反向传播
 * @note 只支持浮点类型（float32 或 float64），与 PyTorch 行为一致
 * @param gy 输出梯度，形状与输入 x 相同，必须是浮点类型
 * @param x 输入张量，必须是浮点类型
 * @param gamma 缩放参数 (weight)，形状为 (C,)，必须与 x 相同的浮点类型
 * @param saved_mean 前向传播时保存的均值，形状为 (C,)，必须与 x 相同的浮点类型
 * @param saved_var 前向传播时保存的方差，形状为 (C,)，必须与 x 相同的浮点类型
 * @param saved_x_norm 前向传播时保存的归一化结果，形状与输入 x 相同，必须与 x 相同的浮点类型
 * @param eps 数值稳定性参数
 * @param num_dims 输入张量的总维度数：2=(N,C), 4=(N,C,H,W)
 * @return 梯度向量：{gx, dgamma, dbeta}
 */
std::vector<std::unique_ptr<Mat>> batch_norm_backward(const OriginMat &gy,
                                                      const OriginMat &x,
                                                      const OriginMat &gamma,
                                                      const OriginMat &saved_mean,
                                                      const OriginMat &saved_var,
                                                      const OriginMat &saved_x_norm,
                                                      float eps,
                                                      int num_dims);

// === 上采样相关操作 ===

/**
 * @brief CUDA upsample：上采样操作（最近邻）
 * @param mode 保留接口，当前未使用，仅实现最近邻
 */
std::unique_ptr<Mat> upsample(const OriginMat &x,
                              const Shape &output_shape,
                              int scale_h,
                              int scale_w,
                              const std::string &mode = "nearest");

/**
 * @brief CUDA upsample_backward：上采样反向传播
 * @param gy 输出梯度 (N, C, OH, OW)
 * @param x_shape 输入形状 (N, C, H, W)
 * @param scale_h 高度缩放因子
 * @param scale_w 宽度缩放因子
 * @param mode 插值模式，须与前向一致
 * @return 输入梯度 (N, C, H, W)
 */
std::unique_ptr<Mat> upsample_backward(const OriginMat &gy,
                                       const Shape &x_shape,
                                       int scale_h,
                                       int scale_w,
                                       const std::string &mode = "nearest");

// === Dropout 相关操作 ===

/**
 * @brief CUDA dropout：Dropout 前向传播
 * @param x 输入张量
 * @param p dropout 概率
 * @param training 是否为训练模式
 * @param mask 输出参数：保存 dropout mask
 * @return 输出张量
 */
std::unique_ptr<Mat> dropout(const OriginMat &x, float p, bool training, OriginMat *mask);

/**
 * @brief CUDA dropout_backward：Dropout 反向传播
 * @param gy 输出梯度
 * @param mask dropout mask
 * @return 输入梯度
 */
std::unique_ptr<Mat> dropout_backward(const OriginMat &gy, const OriginMat &mask);

// === 索引和选择操作 ===

/**
 * @brief CUDA gather：根据索引从矩阵中提取值
 * @param input 输入矩阵 (N, C)
 * @param indices 索引向量 (N,)，每个元素在 [0, C) 范围内
 * @return 提取的值 (N,)
 */
std::unique_ptr<Mat> gather(const OriginMat &input, const OriginMat &indices);

/**
 * @brief CUDA one_hot：将索引转换为 one-hot 编码
 * @param indices 索引向量 (N,)，每个元素在 [0, num_classes) 范围内
 * @param num_classes 类别数量
 * @return one-hot 编码矩阵 (N, num_classes)
 */
std::unique_ptr<Mat> one_hot(const OriginMat &indices, int num_classes);

// ============================================================================
// 非计算类算子（内存操作、索引操作等）
// ============================================================================

/**
 * @brief CUDA clone：深拷贝张量（支持非连续张量）
 * @param mat 输入矩阵
 * @return 拷贝后的矩阵（连续存储）
 */
std::unique_ptr<Mat> clone(const origin::OriginMat &mat);

/**
 * @brief CUDA index_put：根据多维索引写入单个元素
 * @param mat 输入/输出矩阵（原地修改）
 * @param indices 多维索引
 * @param value 要写入的标量值
 */
void index_put(origin::OriginMat &mat, std::initializer_list<size_t> indices, const origin::Scalar &value);

// ============================================================================
// 自定义算子
// ============================================================================

/**
 * @brief CUDA yolo_detect_forward：YOLO Detect 前向传播（单个 stage）
 * @param input 输入特征图 (N, C, H, W)
 * @param conv_weight 卷积权重 (OC, C, 1, 1)
 * @param conv_bias 卷积偏置 (OC,)，可选
 * @param grid grid 坐标
 * @param anchor_grid anchor grid 坐标
 * @param stride stride 值
 * @param num_anchors anchor 数量
 * @param num_classes 类别数量
 * @return 输出张量 (N, num_boxes, classes_info)
 */
std::unique_ptr<Mat> yolo_detect_forward(const origin::OriginMat &input,
                                         const origin::OriginMat &conv_weight,
                                         const origin::OriginMat *conv_bias,
                                         const origin::OriginMat &grid,
                                         const origin::OriginMat &anchor_grid,
                                         float stride,
                                         int32_t num_anchors,
                                         int32_t num_classes);

}  // namespace cuda
}  // namespace origin

#endif  // __ORIGIN_DL_CUDA_OPS_H__
