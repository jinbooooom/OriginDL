#include "origin/core/config.h"

namespace origin
{

namespace Config
{
// 默认启用反向传播
bool enable_backprop = true;
}  // namespace Config

NoGradGuard::NoGradGuard() : old_value_(Config::enable_backprop)
{
    Config::enable_backprop = false;
}

NoGradGuard::~NoGradGuard()
{
    Config::enable_backprop = old_value_;
}

NoGradGuard no_grad()
{
    return NoGradGuard();
}

}  // namespace origin
