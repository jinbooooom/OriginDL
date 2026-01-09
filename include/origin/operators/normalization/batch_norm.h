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

}  // namespace functional
}  // namespace origin

#endif  // __ORIGIN_DL_BATCH_NORM_H__

