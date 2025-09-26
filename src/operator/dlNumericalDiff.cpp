#include "dlOperator.h"

namespace dl
{

// 数值微分，求函数 f 在 x 处的导数
NdArray NumericalDiff(std::function<Tensor(Tensor)> f, const Tensor &x, data_t eps)
{
    auto x0     = Tensor(std::move(x.data() - eps));
    auto x1     = Tensor(std::move(x.data() + eps));
    auto y0     = f(x0);
    auto y1     = f(x1);
    auto diff   = y1.data() - y0.data();
    auto result = *diff / (2 * eps);
    return static_cast<Mat_t *>(result.get())->data_;
}

}  // namespace dl
