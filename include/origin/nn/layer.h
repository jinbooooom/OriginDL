#ifndef __ORIGIN_DL_LAYER_H__
#define __ORIGIN_DL_LAYER_H__

#include "module.h"

namespace origin
{

/**
 * @brief 神经网络层基类
 */
class Layer : public Module
{
public:
    Layer() : Module() {}

    virtual ~Layer() = default;
};

}  // namespace origin

#endif  // __ORIGIN_DL_LAYER_H__
