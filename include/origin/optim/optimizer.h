#ifndef __ORIGIN_DL_OPTIMIZER_H__
#define __ORIGIN_DL_OPTIMIZER_H__

#include <any>
#include <functional>
#include <memory>
#include <unordered_map>
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

    // === State Dict 接口（用于 Checkpoint）===

    /**
     * @brief 获取优化器的状态字典
     * @return 优化器状态字典（包含优化器类型特定的状态）
     * @note 类似 PyTorch 的 optimizer.state_dict()
     */
    virtual std::unordered_map<std::string, std::any> state_dict() const = 0;

    /**
     * @brief 从状态字典加载优化器状态
     * @param state_dict 优化器状态字典
     * @note 类似 PyTorch 的 optimizer.load_state_dict(state_dict)
     */
    virtual void load_state_dict(const std::unordered_map<std::string, std::any> &state_dict) = 0;

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
