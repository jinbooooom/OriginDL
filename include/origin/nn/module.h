#ifndef __ORIGIN_DL_MODULE_H__
#define __ORIGIN_DL_MODULE_H__

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "../core/parameter.h"
#include "../core/tensor.h"
#include "../core/tensor_options.h"
#include "../io/model_io.h"
#include "../utils/exception.h"

namespace origin
{

// 前向声明
class Module;

/**
 * @brief 神经网络模块基类
 * @details 提供参数管理、递归收集、设备管理等功能
 */
class Module
{
protected:
    // 参数注册机制
    std::unordered_map<std::string, Parameter *> parameters_;
    std::unordered_map<std::string, std::unique_ptr<Module>> modules_;

    // 训练状态
    bool training_;

public:
    /**
     * @brief 构造函数
     */
    Module();

    /**
     * @brief 虚析构函数
     */
    virtual ~Module() = default;

    // === 核心接口 ===

    /**
     * @brief 前向传播
     * @param input 输入张量
     * @return 输出张量
     */
    virtual Tensor forward(const Tensor &input) = 0;

    /**
     * @brief 调用操作符
     */
    Tensor operator()(const Tensor &input);

    // === 参数管理 ===

    /**
     * @brief 获取所有参数
     * @return 参数向量
     */
    virtual std::vector<Parameter *> parameters();

    /**
     * @brief 注册参数
     * @param name 参数名称
     * @param param 参数引用
     */
    void register_parameter(const std::string &name, Parameter &param);

    /**
     * @brief 注册子模块
     * @param name 模块名称
     * @param module 子模块
     */
    void register_module(const std::string &name, std::unique_ptr<Module> module);

    // === State Dict 接口（PyTorch 风格）===

    /**
     * @brief State Dict 类型定义
     */
    using StateDict = std::unordered_map<std::string, Tensor>;

    /**
     * @brief 获取模型的状态字典（参数）
     * @return 状态字典，键为参数名称（包含模块路径），值为参数张量
     * @note 类似 PyTorch 的 model.state_dict()
     */
    virtual StateDict state_dict() const;

    /**
     * @brief 从状态字典加载参数
     * @param state_dict 状态字典
     * @param strict 是否严格匹配（默认true），如果为false，允许部分参数缺失
     * @note 类似 PyTorch 的 model.load_state_dict(state_dict, strict=True)
     */
    virtual void load_state_dict(const StateDict &state_dict, bool strict = true);

    /**
     * @brief 从文件加载模型参数（便捷方法）
     * @param filepath 文件路径（.odl 格式）
     * @param strict 是否严格匹配（默认true）
     * @note 内部调用 load_state_dict(load(filepath), strict)
     *       类似 PyTorch 的便捷用法，但 PyTorch 没有这个方法，需要手动两步
     */
    void load(const std::string &filepath, bool strict = true);

    /**
     * @brief 获取命名参数（递归，包含模块路径）
     * @param prefix 参数名称前缀（用于递归调用）
     * @return 参数名称和参数的映射
     */
    std::unordered_map<std::string, Parameter *> named_parameters(const std::string &prefix = "");

    /**
     * @brief 获取命名参数（const 版本，用于 state_dict）
     * @param prefix 参数名称前缀（用于递归调用）
     * @return 参数名称和参数的映射
     */
    std::unordered_map<std::string, const Parameter *> named_parameters(const std::string &prefix) const;

    // === 状态管理===

    /**
     * @brief 设置训练模式
     * @param mode 训练模式标志
     */
    void train(bool mode = true);

    /**
     * @brief 设置评估模式
     */
    void eval();

    /**
     * @brief 判断是否为训练模式
     * @return 是否为训练模式
     */
    bool is_training() const { return training_; }

    // === 设备管理 ===

    /**
     * @brief 迁移到指定设备
     * @param device 目标设备
     */
    virtual void to(Device device);

    /**
     * @brief 迁移到指定选项
     * @param options 张量选项
     */
    void to(const TensorOptions &options);

    // === 梯度管理 ===

    /**
     * @brief 清除所有参数的梯度
     */
    void zero_grad();

    // === 子模块访问===

    /**
     * @brief 获取子模块
     * @tparam T 模块类型
     * @param name 模块名称
     * @return 子模块引用
     */
    template <typename T>
    T &get_module(const std::string &name)
    {
        auto it = modules_.find(name);
        if (it == modules_.end())
        {
            THROW_RUNTIME_ERROR("Module '{}' not found", name);
        }
        auto *ptr = dynamic_cast<T *>(it->second.get());
        if (ptr == nullptr)
        {
            THROW_RUNTIME_ERROR("Module '{}' is not of expected type", name);
        }
        return *ptr;
    }
};

}  // namespace origin

#endif  // __ORIGIN_DL_MODULE_H__
