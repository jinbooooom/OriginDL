#ifndef __ORIGIN_DL_UPSAMPLE_H__
#define __ORIGIN_DL_UPSAMPLE_H__

#include <string>
#include "../../core/operator.h"

namespace origin
{
namespace functional
{

/**
 * @brief Upsample 上采样算子
 * @details 对输入张量进行上采样（最近邻或双线性插值）。
 *          当前仅支持 4D 输入 (N, C, H, W)；mode 取值为 "nearest" 或 "bilinear"。
 */
class Upsample : public Operator
{
public:
    std::string mode_;                      // "nearest" 或 "bilinear"
    std::pair<float, float> scale_factor_;  // 缩放因子 (scale_h, scale_w)
    std::pair<int, int> size_;              // 目标大小 (H, W)，如果指定则忽略 scale_factor

    Upsample(const std::string &mode = "nearest", std::pair<float, float> scale_factor = {2.0f, 2.0f})
        : mode_(mode), scale_factor_(scale_factor), size_({0, 0})
    {}

    Upsample(const std::string &mode, std::pair<int, int> size) : mode_(mode), scale_factor_({0.0f, 0.0f}), size_(size)
    {}

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;
    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;
};

/**
 * @brief Upsample 函数
 * @param x 输入张量，须为 4D，形状 (N, C, H, W)
 * @param mode 上采样模式，取值 "nearest" 或 "bilinear"（当前未生效，实现均为最近邻，保留供后续扩展）
 * @param scale_factor 缩放因子 (scale_h, scale_w)
 * @return 上采样后的张量
 */
Tensor upsample(const Tensor &x,
                const std::string &mode              = "nearest",
                std::pair<float, float> scale_factor = {2.0f, 2.0f});

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_UPSAMPLE_H__
