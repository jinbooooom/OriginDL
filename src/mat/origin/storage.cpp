#include "origin/mat/origin/storage.h"
#include <cstring>  // For std::memcpy
#include <stdexcept>
// #include "origin/utils/exception.h" // Temporarily removed

namespace origin
{

Storage::Storage(size_t size, DeviceType device_type, int device_index)
    : data_(nullptr), size_(size), device_type_(device_type), device_index_(device_index), ref_count_(1)
{
    allocator_ = AllocatorFactory::create_allocator(device_type_, device_index_);
    data_      = allocator_->allocate(size_);
}

Storage::~Storage()
{
    if (data_ != nullptr)
    {
        allocator_->deallocate(data_);
        data_ = nullptr;
    }
}

Storage::Storage(Storage &&other) noexcept
    : data_(other.data_),
      size_(other.size_),
      device_type_(other.device_type_),
      device_index_(other.device_index_),
      ref_count_(other.ref_count_.load()),
      allocator_(std::move(other.allocator_))
{
    other.data_      = nullptr;
    other.size_      = 0;
    other.ref_count_ = 0;
}

Storage &Storage::operator=(Storage &&other) noexcept
{
    if (this != &other)
    {
        if (data_ != nullptr)
        {
            allocator_->deallocate(data_);
        }

        data_         = other.data_;
        size_         = other.size_;
        device_type_  = other.device_type_;
        device_index_ = other.device_index_;
        ref_count_    = other.ref_count_.load();
        allocator_    = std::move(other.allocator_);

        other.data_      = nullptr;
        other.size_      = 0;
        other.ref_count_ = 0;
    }
    return *this;
}

std::shared_ptr<Storage> Storage::create(size_t size, DeviceType device_type, int device_index)
{
    return std::shared_ptr<Storage>(new Storage(size, device_type, device_index));
}

void Storage::add_ref()
{
    ref_count_.fetch_add(1);
}

void Storage::release()
{
    int count = ref_count_.fetch_sub(1);
    if (count == 1)
    {
        delete this;
    }
}

std::shared_ptr<Storage> Storage::to_device(DeviceType target_device_type, int target_device_index) const
{
    if (target_device_type == device_type_ && target_device_index == device_index_)
    {
        // Same device, just return a copy
        auto new_storage = create(size_, device_type_, device_index_);
        memcpy(new_storage->data(), data_, size_);
        return new_storage;
    }

    if (target_device_type == DeviceType::kCPU)
    {
        // Copy to CPU
        auto cpu_storage = create(size_, DeviceType::kCPU);
        memcpy(cpu_storage->data(), data_, size_);
        return cpu_storage;
    }
    else
    {
        // CUDA not supported yet
        throw std::runtime_error("CUDA device transfer not supported yet.");
    }
}

}  // namespace origin