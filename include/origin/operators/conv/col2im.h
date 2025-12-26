#ifndef __ORIGIN_DL_COL2IM_H__
#define __ORIGIN_DL_COL2IM_H__

#include "../../core/operator.h"
#include "../../utils/conv_utils.h"

namespace origin
{

/**
 * @brief col2im 算子：将列矩阵转换回图像形状
 * 
 * 用于反向传播，将列矩阵转换回原始图像形状 (N, C, H, W)
 */
class Col2im : public Operator
{
public:
    Shape input_shape_;
    std::pair<int, int> kernel_size_;
    std::pair<int, int> stride_;
    std::pair<int, int> pad_;
    bool to_matrix_;

    Col2im(const Shape &input_shape, std::pair<int, int> kernel_size, std::pair<int, int> stride,
           std::pair<int, int> pad, bool to_matrix)
        : input_shape_(input_shape), kernel_size_(kernel_size), stride_(stride), pad_(pad), to_matrix_(to_matrix)
    {
    }

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;
    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief col2im 函数
 * @param col 列矩阵张量
 * @param input_shape 原始输入形状 (N, C, H, W)
 * @param kernel_size 卷积核大小
 * @param stride 步长
 * @param pad 填充
 * @param to_matrix 是否从矩阵形式转换，默认 true
 * @return 形状为 (N, C, H, W) 的张量
 */
Tensor col2im(const Tensor &col, const Shape &input_shape, std::pair<int, int> kernel_size,
              std::pair<int, int> stride = {1, 1}, std::pair<int, int> pad = {0, 0}, bool to_matrix = true);

Tensor col2im(const Tensor &col, const Shape &input_shape, int kernel_size, int stride = 1, int pad = 0,
              bool to_matrix = true);

}  // namespace origin

#endif  // __ORIGIN_DL_COL2IM_H__

