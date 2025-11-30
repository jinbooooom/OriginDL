#ifndef __ORIGIN_DL_SEQUENTIAL_H__
#define __ORIGIN_DL_SEQUENTIAL_H__

#include <memory>
#include <vector>
#include "../core/tensor.h"
#include "module.h"

namespace origin
{

/**
 * @brief 顺序模型容器
 */
class Sequential : public Module
{
private:
    std::vector<std::unique_ptr<Module>> modules_;

public:
    /**
     * @brief 默认构造函数
     */
    Sequential() = default;

    /**
     * @brief 析构函数
     */
    virtual ~Sequential() = default;

    /**
     * @brief 添加层
     * @param module 模块指针
     */
    void add(std::unique_ptr<Module> module);

    /**
     * @brief 前向传播
     * @param input 输入张量
     * @return 输出张量
     */
    Tensor forward(const Tensor &input) override;

    /**
     * @brief 重写parameters以收集所有子模块参数
     * @return 参数向量
     */
    std::vector<Parameter *> parameters() override;

    /**
     * @brief 重写to以迁移所有子模块到指定设备
     * @param device 目标设备
     */
    void to(Device device) override;

    /**
     * @brief 访问器
     * @param index 索引
     * @return 模块引用
     */
    Module &operator[](size_t index);

    /**
     * @brief 访问器（const版本）
     * @param index 索引
     * @return 模块常量引用
     */
    const Module &operator[](size_t index) const;

    /**
     * @brief 获取模块数量
     * @return 模块数量
     */
    size_t size() const { return modules_.size(); }
};

}  // namespace origin

#endif  // __ORIGIN_DL_SEQUENTIAL_H__
