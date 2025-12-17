#include "origin/mat/origin/storage.h"
#include <cstring>  // For std::memcpy
#include <stdexcept>
#include "origin/utils/exception.h"

#ifdef WITH_CUDA
#    include <cuda_runtime.h>
#endif

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
        if (device_type_ == DeviceType::kCPU)
        {
            memcpy(new_storage->data(), data_, size_);
        }
        else
        {
#ifdef WITH_CUDA
            // CUDA to CUDA copy
            cudaMemcpy(new_storage->data(), data_, size_, cudaMemcpyDeviceToDevice);
#else
            THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
        }
        return new_storage;
    }

    if (target_device_type == DeviceType::kCPU)
    {
        // Copy to CPU
        auto cpu_storage = create(size_, DeviceType::kCPU);
        if (device_type_ == DeviceType::kCUDA)
        {
#ifdef WITH_CUDA
            cudaError_t err = cudaMemcpy(cpu_storage->data(), data_, size_, cudaMemcpyDeviceToHost);
            if (err != cudaSuccess)
            {
                THROW_RUNTIME_ERROR("CUDA memory copy failed: {}", cudaGetErrorString(err));
            }
            cudaDeviceSynchronize();
#else
            THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
        }
        else
        {
            memcpy(cpu_storage->data(), data_, size_);
        }
        return cpu_storage;
    }
    else if (target_device_type == DeviceType::kCUDA)
    {
        // Copy to CUDA
        auto cuda_storage = create(size_, DeviceType::kCUDA, target_device_index);
        if (device_type_ == DeviceType::kCPU)
        {
#ifdef WITH_CUDA
            cudaMemcpy(cuda_storage->data(), data_, size_, cudaMemcpyHostToDevice);
#else
            THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
        }
        else
        {
#ifdef WITH_CUDA
            cudaMemcpy(cuda_storage->data(), data_, size_, cudaMemcpyDeviceToDevice);
#else
            THROW_RUNTIME_ERROR("CUDA support not compiled in");
#endif
        }
        return cuda_storage;
    }
    else
    {
        Device src_device(device_type_, device_index_);
        Device dst_device(target_device_type, target_device_index);
        THROW_RUNTIME_ERROR("Unsupported device type for transfer: from device {} to device {}", src_device.to_string(),
                            dst_device.to_string());
    }
}

}  // namespace origin