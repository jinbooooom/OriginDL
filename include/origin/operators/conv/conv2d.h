#ifndef __ORIGIN_DL_CONV2D_H__
#define __ORIGIN_DL_CONV2D_H__

#include "../../core/operator.h"
#include "../../utils/conv_utils.h"

namespace origin
{

/**
 * @brief Conv2d 算子：二维卷积操作
 *
 * 输入：
 * - x: 输入张量，形状 (N, C, H, W)
 * - W: 卷积核，形状 (OC, C, KH, KW)
 * - b: 偏置（可选），形状 (OC,)
 *
 * 输出：
 * - y: 形状为 (N, OC, OH, OW) 的张量
 */
class Conv2dOp : public Operator  // TODO：未来增加命名空间，将Conv2dOp改为Conv2d，避免与Conv2d类名冲突
{
public:
    std::pair<int, int> stride_;
    std::pair<int, int> pad_;

    Conv2dOp(std::pair<int, int> stride, std::pair<int, int> pad) : stride_(stride), pad_(pad) {}

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;
    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief Conv2d 函数
 * @param x 输入张量，形状 (N, C, H, W)
 * @param W 卷积核，形状 (OC, C, KH, KW)
 * @param b 偏置（可选），形状 (OC,)
 * @param stride 步长，默认 1
 * @param pad 填充，默认 0
 * @return 形状为 (N, OC, OH, OW) 的张量
 */
Tensor conv2d(const Tensor &x,
              const Tensor &W,
              const Tensor *b            = nullptr,
              std::pair<int, int> stride = {1, 1},
              std::pair<int, int> pad    = {0, 0});

Tensor conv2d(const Tensor &x, const Tensor &W, const Tensor *b, int stride, int pad);

}  // namespace origin

#endif  // __ORIGIN_DL_CONV2D_H__
