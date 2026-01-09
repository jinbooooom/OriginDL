#ifndef __ORIGIN_DL_BATCH_NORM_H__
#define __ORIGIN_DL_BATCH_NORM_H__

#include "../../core/operator.h"

namespace origin
{
namespace functional
{

class BatchNorm : public Operator
{
public:
    bool training_;
    float eps_;
    float momentum_;
    int num_dims_;

    BatchNorm(bool training, float eps, float momentum, int num_dims);

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;

private:
    Tensor saved_mean_;
    Tensor saved_var_;
    Tensor saved_x_norm_;
};

/**
 * @brief 函数式接口：BatchNorm 算子
 * @details 通用的 BatchNorm 实现，支持 BatchNorm1d、BatchNorm2d 和 BatchNorm3d
 * 通过 num_dims 参数区分：
 * - num_dims = 2: BatchNorm1d (输入形状 (N, C))，1D 特征空间（序列）
 * - num_dims = 4: BatchNorm2d (输入形状 (N, C, H, W))，2D 特征空间（图像）
 * - num_dims = 5: BatchNorm3d (输入形状 (N, C, D, H, W))，3D 特征空间（体积）
 * 
 * 注意：num_dims 是输入张量的总维度数，而 "1D"、"2D"、"3D" 指的是特征空间的维度
 * （即除了 batch 维度 N 和 channel 维度 C 外的空间维度数）
 * 
 * @param x 输入张量
 * @param gamma 缩放参数 (weight)，形状为 (C,)
 * @param beta 偏移参数 (bias)，形状为 (C,)
 * @param running_mean 运行均值，形状为 (C,)
 * @param running_var 运行方差，形状为 (C,)
 * @param training 是否为训练模式，默认 false
 * @param eps 数值稳定性参数，默认 1e-5
 * @param momentum 动量参数，用于更新 running_mean 和 running_var，默认 0.1
 * @param num_dims 输入张量的总维度数：2=(N,C), 4=(N,C,H,W), 5=(N,C,D,H,W)，默认 4
 * @return 输出张量，形状与输入相同
 */
Tensor batch_norm(const Tensor &x,
                  const Tensor &gamma,
                  const Tensor &beta,
                  const Tensor &running_mean,
                  const Tensor &running_var,
                  bool training = false,
                  float eps = 1e-5f,
                  float momentum = 0.1f,
                  int num_dims = 4);

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_BATCH_NORM_H__

