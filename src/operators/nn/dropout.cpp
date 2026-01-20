#include <random>
#include <vector>
#include "origin/core/operator.h"
#include "origin/core/tensor.h"
#include "origin/mat/mat.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/origin/cpu/cpu_ops.h"
#include "origin/utils/branch_prediction.h"
#include "origin/utils/exception.h"

#ifdef WITH_CUDA
#include "origin/mat/origin/cuda/cuda_ops.cuh"
#endif

namespace origin
{
namespace functional
{

Dropout::Dropout(float p, bool training) : p_(p), training_(training)
{
    if (unlikely(p < 0.0f || p >= 1.0f))
    {
        THROW_INVALID_ARG("Dropout: p must be in [0, 1), but got {}", p);
    }
}

std::vector<Tensor> Dropout::forward(const std::vector<Tensor> &xs)
{
    if (unlikely(xs.size() != 1))
    {
        THROW_RUNTIME_ERROR("Dropout operator requires exactly 1 input, but got {}", xs.size());
    }

    auto &x = xs[0];

    // 获取 Mat 引用并转换为 OriginMat
    const OriginMat &x_mat = static_cast<const OriginMat &>(mat(x));

    // 创建 mask OriginMat 用于保存 dropout mask
    auto mask_mat_unique = std::make_unique<OriginMat>(x.shape(), DataType::kFloat32, x.device());
    OriginMat *mask_mat = mask_mat_unique.get();

    // 根据设备类型调用对应的实现
    std::unique_ptr<Mat> result;
    if (x.device().type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        result = cuda::dropout(x_mat, p_, training_, mask_mat);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        result = cpu::dropout(x_mat, p_, training_, mask_mat);
    }

    // 保存 mask 用于反向传播
    mask_ = convert_mat_to_tensor(std::move(mask_mat_unique));

    auto y = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(y)};
}

std::vector<Tensor> Dropout::backward(const std::vector<Tensor> &gys)
{
    if (unlikely(gys.size() != 1))
    {
        THROW_RUNTIME_ERROR("Dropout backward requires exactly 1 gradient, but got {}", gys.size());
    }

    auto &gy = gys[0];

    if (!training_)
    {
        // 推理模式：梯度直接传递
        return std::vector<Tensor>{std::move(gy)};
    }

    // 训练模式：根据 mask 计算梯度
    // 获取 Mat 引用并转换为 OriginMat
    const OriginMat &gy_mat  = static_cast<const OriginMat &>(mat(gy));
    const OriginMat &mask_mat = static_cast<const OriginMat &>(mat(mask_));

    // 根据设备类型调用对应的实现
    std::unique_ptr<Mat> result;
    if (gy.device().type() == DeviceType::kCUDA)
    {
#ifdef WITH_CUDA
        result = cuda::dropout_backward(gy_mat, mask_mat);
#else
        THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
    }
    else
    {
        result = cpu::dropout_backward(gy_mat, mask_mat);
    }

    auto gx = convert_mat_to_tensor(std::move(result));
    return std::vector<Tensor>{std::move(gx)};
}

}  // namespace functional
}  // namespace origin
