#ifndef __ORIGIN_DL_OPTIMIZER_H__
#define __ORIGIN_DL_OPTIMIZER_H__

#include <functional>
#include <memory>
#include <vector>
#include "../core/parameter.h"
#include "../nn/module.h"

namespace origin
{

/**
 * @brief 优化器基类
 */
class Optimizer
{
protected:
    // 目标模型
    Module *target_;

    // Hook列表
    std::vector<std::function<void(std::vector<Parameter *> &)>> hooks_;

    // 参数列表
    std::vector<Parameter *> parameters_;

public:
    /**
     * @brief 构造函数
     * @param target 目标模块
     */
    explicit Optimizer(Module &target);

    /**
     * @brief 虚析构函数
     */
    virtual ~Optimizer() = default;

    /**
     * @brief 执行参数更新
     */
    void step();

    /**
     * @brief 清除梯度
     */
    void zero_grad();

    /**
     * @brief 注册Hook
     * @param hook Hook函数
     */
    void register_hook(std::function<void(std::vector<Parameter *> &)> hook);

    /**
     * @brief 获取参数列表
     * @return 参数列表引用
     */
    std::vector<Parameter *> &parameters() { return parameters_; }

protected:
    /**
     * @brief 更新单个参数
     * @param param 参数引用
     */
    virtual void step_one(Parameter &param) = 0;

private:
    /**
     * @brief 收集参数
     */
    void collect_parameters();
};

}  // namespace origin

#endif  // __ORIGIN_DL_OPTIMIZER_H__
