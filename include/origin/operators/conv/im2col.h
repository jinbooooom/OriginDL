#ifndef __ORIGIN_DL_IM2COL_H__
#define __ORIGIN_DL_IM2COL_H__

#include "../../core/operator.h"
#include "../../utils/conv_utils.h"

namespace origin
{

/**
 * @brief im2col 算子：将图像转换为列矩阵
 * 
 * 将输入图像 (N, C, H, W) 转换为列矩阵，用于卷积操作
 */
class Im2col : public Operator
{
public:
    std::pair<int, int> kernel_size_;
    std::pair<int, int> stride_;
    std::pair<int, int> pad_;
    bool to_matrix_;
    Shape input_shape_;

    Im2col(std::pair<int, int> kernel_size, std::pair<int, int> stride, std::pair<int, int> pad, bool to_matrix)
        : kernel_size_(kernel_size), stride_(stride), pad_(pad), to_matrix_(to_matrix)
    {
    }

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;
    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief im2col 函数
 * @param x 输入张量，形状为 (N, C, H, W)
 * @param kernel_size 卷积核大小 (int 或 (int, int))
 * @param stride 步长 (int 或 (int, int))，默认 1
 * @param pad 填充 (int 或 (int, int))，默认 0
 * @param to_matrix 是否转换为矩阵形式，默认 true
 * @return 如果 to_matrix=true：形状为 (N*OH*OW, C*KH*KW) 的张量
 *         如果 to_matrix=false：形状为 (N, C, KH, KW, OH, OW) 的张量
 */
Tensor im2col(const Tensor &x, std::pair<int, int> kernel_size, std::pair<int, int> stride = {1, 1},
              std::pair<int, int> pad = {0, 0}, bool to_matrix = true);

Tensor im2col(const Tensor &x, int kernel_size, int stride = 1, int pad = 0, bool to_matrix = true);

}  // namespace origin

#endif  // __ORIGIN_DL_IM2COL_H__

