#ifndef __ORIGIN_DL_ALLOCATOR_H__
#define __ORIGIN_DL_ALLOCATOR_H__

#include <memory>
#include <stdexcept>
#include "../../../common/inner_types.h"
#include "../../basic_types.h"

namespace origin
{

/**
 * @brief 内存分配器抽象基类
 *
 * 提供设备无关的内存分配接口，支持CPU和CUDA内存分配
 */
class Allocator
{
public:
    virtual ~Allocator() = default;

    /**
     * @brief 分配内存
     * @param size 要分配的字节数
     * @return 分配的内存指针
     */
    virtual void *allocate(size_t size) = 0;

    /**
     * @brief 释放内存
     * @param ptr 要释放的内存指针
     */
    virtual void deallocate(void *ptr) = 0;

    /**
     * @brief 获取设备类型
     * @return 设备类型
     */
    virtual DeviceType device_type() const = 0;
};

/**
 * @brief CPU内存分配器
 */
class CPUAllocator : public Allocator
{
public:
    void *allocate(size_t size) override;
    void deallocate(void *ptr) override;
    DeviceType device_type() const override { return DeviceType::kCPU; }
};

/**
 * @brief CUDA内存分配器
 */
class CUDAAllocator : public Allocator
{
private:
    int device_index_;

public:
    explicit CUDAAllocator(int device_index = 0) : device_index_(device_index) {}

    void *allocate(size_t size) override;
    void deallocate(void *ptr) override;
    DeviceType device_type() const override { return DeviceType::kCUDA; }
};

/**
 * @brief 分配器工厂类
 */
class AllocatorFactory
{
public:
    /**
     * @brief 创建分配器
     * @param device_type 设备类型
     * @param device_index 设备索引
     * @return 分配器智能指针
     */
    static std::unique_ptr<Allocator> create_allocator(DeviceType device_type, int device_index = 0);
};

}  // namespace origin

#endif  // __ORIGIN_DL_ALLOCATOR_H__