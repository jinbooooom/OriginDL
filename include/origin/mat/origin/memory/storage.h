#ifndef __ORIGIN_DL_STORAGE_H__
#define __ORIGIN_DL_STORAGE_H__

#include <memory>
#include <stdexcept>
#include "../../basic_types.h"

namespace origin
{

// 前向声明
class MemoryPool;

/**
 * @brief 存储类，管理原始内存
 *
 * 负责管理实际的数据存储，包括内存分配、释放等。
 *
 */
class Storage
{
private:
    void *data_;
    size_t size_;  // Size in bytes
    DeviceType device_type_;
    int device_index_;
    MemoryPool *pool_;

    // Disable copy constructor and copy assignment
    Storage(const Storage &)            = delete;
    Storage &operator=(const Storage &) = delete;

public:
    /**
     * @brief 构造函数
     * @param size 存储大小（字节）
     * @param device_type 设备类型
     * @param device_index 设备索引
     */
    Storage(size_t size, DeviceType device_type, int device_index = 0);

    /**
     * @brief 析构函数
     */
    ~Storage();

    // Move constructor
    Storage(Storage &&other) noexcept;

    // Move assignment operator
    Storage &operator=(Storage &&other) noexcept;

    /**
     * @brief 创建Storage的工厂方法
     * @param size 存储大小
     * @param device_type 设备类型
     * @param device_index 设备索引
     * @return Storage的shared_ptr
     */
    static std::shared_ptr<Storage> create(size_t size, DeviceType device_type, int device_index = 0);

    /**
     * @brief 获取数据指针
     * @return 数据指针
     */
    void *data() { return data_; }
    const void *data() const { return data_; }

    /**
     * @brief 获取存储大小
     * @return 大小（字节）
     */
    size_t size() const { return size_; }

    /**
     * @brief 获取设备类型
     * @return 设备类型
     */
    DeviceType device_type() const { return device_type_; }

    /**
     * @brief 获取设备索引
     * @return 设备索引
     */
    int device_index() const { return device_index_; }

    /**
     * @brief 将数据复制到另一个设备
     * @param target_device_type 目标设备类型
     * @param target_device_index 目标设备索引
     * @return 新的Storage对象
     */
    std::shared_ptr<Storage> to_device(DeviceType target_device_type, int target_device_index = 0) const;
};

}  // namespace origin

#endif  // __ORIGIN_DL_STORAGE_H__