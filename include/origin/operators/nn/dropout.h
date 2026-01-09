#ifndef __ORIGIN_DL_DROPOUT_H__
#define __ORIGIN_DL_DROPOUT_H__

#include "../../core/operator.h"

namespace origin
{

class Dropout : public Operator
{
public:
    float p_;  // dropout 概率
    bool training_;

    Dropout(float p, bool training);

    std::vector<Tensor> forward(const std::vector<Tensor> &xs) override;

    std::vector<Tensor> backward(const std::vector<Tensor> &gys) override;

private:
    Tensor mask_;
};

}  // namespace origin

#endif  // __ORIGIN_DL_DROPOUT_H__

