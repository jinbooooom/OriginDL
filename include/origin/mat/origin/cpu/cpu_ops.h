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
/**
 * @brief CPU加法算子统一实现
 * @param a 输入矩阵A
 * @param b 输入矩阵B
 * @param out 输出矩阵指针，如果为nullptr则创建新矩阵，否则将结果写入out
 * @return 如果out==nullptr则返回新矩阵，否则返回nullptr（结果在out中）
 */
std::unique_ptr<Mat> add(const OriginMat &a, const OriginMat &b, OriginMat *out = nullptr);
std::unique_ptr<Mat> subtract(const OriginMat &a, const OriginMat &b, OriginMat *out = nullptr);
std::unique_ptr<Mat> multiply(const OriginMat &a, const OriginMat &b, OriginMat *out = nullptr);
std::unique_ptr<Mat> divide(const OriginMat &a, const OriginMat &b, OriginMat *out = nullptr);
std::unique_ptr<Mat> matmul(const OriginMat &a, const OriginMat &b);


// ============================================================================
// 比较运算算子
// ============================================================================

/**
 * @brief 等于比较算子
 * @param mat 输入矩阵
 * @param threshold 比较阈值，可以是标量（shape为{}或{1}）或与输入相同形状的张量
 * @param out 输出矩阵指针，如果为nullptr则创建新矩阵，否则将结果写入out
 * @return 如果out==nullptr则返回新矩阵，否则返回nullptr（结果在out中）
 */
std::unique_ptr<Mat> eq(const OriginMat &mat, const OriginMat &threshold, OriginMat *out = nullptr);

/**
 * @brief 不等于比较算子
 */
std::unique_ptr<Mat> ne(const OriginMat &mat, const OriginMat &threshold, OriginMat *out = nullptr);

/**
 * @brief 小于比较算子
 */
std::unique_ptr<Mat> lt(const OriginMat &mat, const OriginMat &threshold, OriginMat *out = nullptr);

/**
 * @brief 小于等于比较算子
 */
std::unique_ptr<Mat> le(const OriginMat &mat, const OriginMat &threshold, OriginMat *out = nullptr);

/**
 * @brief 大于比较算子
 */
std::unique_ptr<Mat> gt(const OriginMat &mat, const OriginMat &threshold, OriginMat *out = nullptr);

/**
 * @brief 大于等于比较算子
 */
std::unique_ptr<Mat> ge(const OriginMat &mat, const OriginMat &threshold, OriginMat *out = nullptr);

std::unique_ptr<Mat> negate(const OriginMat &mat, OriginMat *out = nullptr);

// === 数学函数 ===
std::unique_ptr<Mat> exp(const OriginMat &mat, OriginMat *out = nullptr);
std::unique_ptr<Mat> log(const OriginMat &mat, OriginMat *out = nullptr);
std::unique_ptr<Mat> sqrt(const OriginMat &mat, OriginMat *out = nullptr);
std::unique_ptr<Mat> square(const OriginMat &mat, OriginMat *out = nullptr);
std::unique_ptr<Mat> pow(const OriginMat &mat, const Scalar &exponent);
void pow_inplace(OriginMat &mat, const Scalar &exponent);
std::unique_ptr<Mat> relu(const OriginMat &mat, OriginMat *out = nullptr);

// === 统计函数 ===
std::unique_ptr<Mat> sum(const OriginMat &mat, int axis, bool keepdim = false);

/**
 * @brief CPU max：沿指定轴计算最大值
 * @param mat 输入矩阵
 * @param axis 计算轴，-1 表示所有元素
 * @return 最大值结果矩阵
 */
std::unique_ptr<Mat> max(const OriginMat &mat, int axis);

// === 形状操作 ===
std::unique_ptr<Mat> reshape(const OriginMat &mat, const Shape &new_shape);
std::unique_ptr<Mat> transpose(const OriginMat &mat);
/**
 * @brief CPU permute：按照指定顺序重新排列张量的维度
 * @param mat 输入矩阵
 * @param dims 新的维度顺序，例如 {0, 2, 3, 1} 表示将维度 0,1,2,3 重新排列为 0,2,3,1
 * @return 重排后的矩阵
 */
std::unique_ptr<Mat> permute(const OriginMat &mat, const std::vector<int> &dims);
std::unique_ptr<Mat> broadcast_to(const OriginMat &mat, const Shape &target_shape);
std::unique_ptr<Mat> sum_to(const OriginMat &mat, const Shape &target_shape);

/**
 * @brief CPU cat算子实现
 * @param inputs 输入矩阵列表
 * @param dim 拼接维度
 * @return 拼接结果矩阵
 */
std::unique_ptr<Mat> cat(const std::vector<const OriginMat *> &inputs, int dim);

/**
 * @brief CPU split：将矩阵沿指定维度分割成多个矩阵（cat 的反向操作）
 * @param input 输入矩阵
 * @param output_shapes 输出形状列表
 * @param dim 分割维度
 * @return 分割后的矩阵列表
 */
std::vector<std::unique_ptr<Mat>> split(const OriginMat &input, const std::vector<Shape> &output_shapes, int dim);

// === 类型转换 ===
std::unique_ptr<Mat> convert_datatype(const OriginMat &mat, DataType target_type);

// === 卷积相关 ===
// 注意：im2col 和 col2im 是 conv2d 的内部实现，仅供 OriginMat::im2col/col2im 使用
std::unique_ptr<Mat> im2col(const OriginMat &img,
                            std::pair<int, int> kernel_size,
                            std::pair<int, int> stride,
                            std::pair<int, int> pad,
                            bool to_matrix);
std::unique_ptr<Mat> col2im(const OriginMat &col,
                            const Shape &input_shape,
                            std::pair<int, int> kernel_size,
                            std::pair<int, int> stride,
                            std::pair<int, int> pad,
                            bool to_matrix);

/**
 * @brief 完整的卷积操作（前向传播）
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
 * @brief 完整的卷积反向传播
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
 * @brief 平均池化操作（前向传播）
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
 * @brief 平均池化反向传播
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
 * @brief 自适应平均池化操作
 * @param x 输入张量 (N, C, H, W)
 * @param output_size 输出尺寸 (OH, OW)
 * @return 输出张量 (N, C, OH, OW)
 */
std::unique_ptr<Mat> adaptive_avg_pool2d(const OriginMat &x, std::pair<int, int> output_size);

/**
 * @brief 自适应平均池化反向传播
 * @param gy 输出梯度 (N, C, OH, OW)
 * @param x 输入张量 (N, C, H, W)
 * @param output_size 输出尺寸 (OH, OW)
 * @return 输入梯度 (N, C, H, W)
 */
std::unique_ptr<Mat> adaptive_avg_pool2d_backward(const OriginMat &gy,
                                                  const OriginMat &x,
                                                  std::pair<int, int> output_size);

/**
 * @brief 最大池化操作（前向传播）
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
 * @brief 最大池化反向传播
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

// === 归一化相关 ===

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
 * @brief BatchNorm 前向传播（返回所有中间结果）
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
 * @brief BatchNorm 前向传播（只返回输出）
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
 * @brief BatchNorm 反向传播
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
 * @brief CPU upsample：上采样操作（最近邻）
 * @param x 输入张量 (N, C, H, W)
 * @param output_shape 输出形状 (N, C, OH, OW)
 * @param scale_h 高度缩放因子
 * @param scale_w 宽度缩放因子
 * @return 输出张量 (N, C, OH, OW)
 */
std::unique_ptr<Mat> upsample(const OriginMat &x, const Shape &output_shape, int scale_h, int scale_w);

/**
 * @brief CPU upsample_backward：上采样反向传播
 * @param gy 输出梯度 (N, C, OH, OW)
 * @param x_shape 输入形状 (N, C, H, W)
 * @param scale_h 高度缩放因子
 * @param scale_w 宽度缩放因子
 * @return 输入梯度 (N, C, H, W)
 */
std::unique_ptr<Mat> upsample_backward(const OriginMat &gy, const Shape &x_shape, int scale_h, int scale_w);

// === Dropout 相关操作 ===

/**
 * @brief CPU dropout：Dropout 前向传播
 * @param x 输入张量
 * @param p dropout 概率
 * @param training 是否为训练模式
 * @param mask 输出参数：保存 dropout mask
 * @return 输出张量
 */
std::unique_ptr<Mat> dropout(const OriginMat &x, float p, bool training, OriginMat *mask);

/**
 * @brief CPU dropout_backward：Dropout 反向传播
 * @param gy 输出梯度
 * @param mask dropout mask
 * @return 输入梯度
 */
std::unique_ptr<Mat> dropout_backward(const OriginMat &gy, const OriginMat &mask);

// === 索引和选择操作 ===

/**
 * @brief CPU gather：根据索引从矩阵中提取值
 * @param input 输入矩阵 (N, C)
 * @param indices 索引向量 (N,)，每个元素在 [0, C) 范围内
 * @return 提取的值 (N,)
 */
std::unique_ptr<Mat> gather(const OriginMat &input, const OriginMat &indices);

/**
 * @brief CPU one_hot：将索引转换为 one-hot 编码
 * @param indices 索引向量 (N,)，每个元素在 [0, num_classes) 范围内
 * @param num_classes 类别数量
 * @return one-hot 编码矩阵 (N, num_classes)
 */
std::unique_ptr<Mat> one_hot(const OriginMat &indices, int num_classes);

// === 自定义算子 ===

/**
 * @brief CPU yolo_detect_forward：YOLO Detect 前向传播（单个 stage）
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
std::unique_ptr<Mat> yolo_detect_forward(const OriginMat &input,
                                         const OriginMat &conv_weight,
                                         const OriginMat *conv_bias,
                                         const OriginMat &grid,
                                         const OriginMat &anchor_grid,
                                         float stride,
                                         int32_t num_anchors,
                                         int32_t num_classes);

}  // namespace cpu
}  // namespace origin

#endif  // __ORIGIN_DL_CPU_OPS_H__
