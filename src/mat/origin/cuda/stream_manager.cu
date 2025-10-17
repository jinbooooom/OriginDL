#include <algorithm>
#include <stdexcept>
#include "origin/mat/origin/cuda/cuda_utils.cuh"
#include "origin/mat/origin/cuda/stream_manager.cuh"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cuda
{

// ============================================================================
// StreamManager 实现
// ============================================================================

StreamManager::StreamManager(size_t num_streams) : current_stream_(0)
{
    if (num_streams == 0)
    {
        THROW_INVALID_ARG("Number of streams must be greater than 0");
    }

    streams_.resize(num_streams);
    events_.resize(num_streams);

    // 创建CUDA流
    for (size_t i = 0; i < num_streams; ++i)
    {
        cudaError_t err = cudaStreamCreate(&streams_[i]);
        if (err != cudaSuccess)
        {
            // 清理已创建的流
            for (size_t j = 0; j < i; ++j)
            {
                cudaStreamDestroy(streams_[j]);
            }
            THROW_RUNTIME_ERROR("Failed to create CUDA stream {}: {}", i, cudaGetErrorString(err));
        }
    }

    // 创建CUDA事件
    for (size_t i = 0; i < num_streams; ++i)
    {
        cudaError_t err = cudaEventCreate(&events_[i]);
        if (err != cudaSuccess)
        {
            // 清理已创建的事件和流
            for (size_t j = 0; j < i; ++j)
            {
                cudaEventDestroy(events_[j]);
            }
            for (size_t j = 0; j < num_streams; ++j)
            {
                cudaStreamDestroy(streams_[j]);
            }
            THROW_RUNTIME_ERROR("Failed to create CUDA event {}: {}", i, cudaGetErrorString(err));
        }
    }
}

StreamManager::~StreamManager()
{
    // 销毁所有事件
    for (auto &event : events_)
    {
        if (event != nullptr)
        {
            cudaEventDestroy(event);
        }
    }

    // 销毁所有流
    for (auto &stream : streams_)
    {
        if (stream != nullptr)
        {
            cudaStreamDestroy(stream);
        }
    }
}

StreamManager::StreamManager(StreamManager &&other) noexcept
    : streams_(std::move(other.streams_)), events_(std::move(other.events_)), current_stream_(other.current_stream_)
{
    // 清空源对象
    other.streams_.clear();
    other.events_.clear();
    other.current_stream_ = 0;
}

StreamManager &StreamManager::operator=(StreamManager &&other) noexcept
{
    if (this != &other)
    {
        // 清理当前对象
        for (auto &event : events_)
        {
            if (event != nullptr)
            {
                cudaEventDestroy(event);
            }
        }
        for (auto &stream : streams_)
        {
            if (stream != nullptr)
            {
                cudaStreamDestroy(stream);
            }
        }

        // 移动资源
        streams_        = std::move(other.streams_);
        events_         = std::move(other.events_);
        current_stream_ = other.current_stream_;

        // 清空源对象
        other.streams_.clear();
        other.events_.clear();
        other.current_stream_ = 0;
    }
    return *this;
}

cudaStream_t StreamManager::get_next_stream()
{
    if (streams_.empty())
    {
        THROW_RUNTIME_ERROR("No streams available");
    }

    cudaStream_t stream = streams_[current_stream_];
    current_stream_     = (current_stream_ + 1) % streams_.size();
    return stream;
}

cudaStream_t StreamManager::get_stream(size_t index) const
{
    if (index >= streams_.size())
    {
        THROW_INVALID_ARG("Stream index {} out of range [0, {})", index, streams_.size());
    }
    return streams_[index];
}

void StreamManager::record_event(size_t stream_id)
{
    if (stream_id >= events_.size())
    {
        THROW_INVALID_ARG("Stream index {} out of range [0, {})", stream_id, events_.size());
    }

    cudaError_t err = cudaEventRecord(events_[stream_id], streams_[stream_id]);
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("Failed to record event for stream {}: {}", stream_id, cudaGetErrorString(err));
    }
}

void StreamManager::wait_for_event(size_t event_id)
{
    if (event_id >= events_.size())
    {
        THROW_INVALID_ARG("Event index {} out of range [0, {})", event_id, events_.size());
    }

    cudaError_t err = cudaEventSynchronize(events_[event_id]);
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("Failed to synchronize event {}: {}", event_id, cudaGetErrorString(err));
    }
}

void StreamManager::wait_for_stream(size_t current_stream, size_t wait_for_stream)
{
    if (current_stream >= streams_.size())
    {
        THROW_INVALID_ARG("Current stream index {} out of range [0, {})", current_stream, streams_.size());
    }
    if (wait_for_stream >= events_.size())
    {
        THROW_INVALID_ARG("Wait for stream index {} out of range [0, {})", wait_for_stream, events_.size());
    }

    cudaError_t err = cudaStreamWaitEvent(streams_[current_stream], events_[wait_for_stream], 0);
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("Failed to wait for stream {} in stream {}: {}", wait_for_stream, current_stream,
                            cudaGetErrorString(err));
    }
}

void StreamManager::synchronize_all()
{
    for (size_t i = 0; i < streams_.size(); ++i)
    {
        cudaError_t err = cudaStreamSynchronize(streams_[i]);
        if (err != cudaSuccess)
        {
            THROW_RUNTIME_ERROR("Failed to synchronize stream {}: {}", i, cudaGetErrorString(err));
        }
    }
}

bool StreamManager::is_valid() const
{
    return !streams_.empty() && !events_.empty();
}

// ============================================================================
// AdvancedStreamManager 实现
// ============================================================================

AdvancedStreamManager::AdvancedStreamManager(size_t num_streams, bool enable_profiling)
    : stream_manager_(num_streams), enable_profiling_(enable_profiling), next_event_id_(0)
{
    stream_usage_count_.resize(num_streams, 0);
    events_.resize(num_streams * 2);  // 为每个流分配2个事件

    // 创建事件
    for (size_t i = 0; i < events_.size(); ++i)
    {
        cudaError_t err = cudaEventCreate(&events_[i]);
        if (err != cudaSuccess)
        {
            // 清理已创建的事件
            for (size_t j = 0; j < i; ++j)
            {
                cudaEventDestroy(events_[j]);
            }
            THROW_RUNTIME_ERROR("Failed to create advanced CUDA event {}: {}", i, cudaGetErrorString(err));
        }
    }
}

AdvancedStreamManager::~AdvancedStreamManager()
{
    // 销毁所有事件
    for (auto &event : events_)
    {
        if (event != nullptr)
        {
            cudaEventDestroy(event);
        }
    }
}

AdvancedStreamManager::AdvancedStreamManager(AdvancedStreamManager &&other) noexcept
    : stream_manager_(std::move(other.stream_manager_)),
      stream_usage_count_(std::move(other.stream_usage_count_)),
      events_(std::move(other.events_)),
      enable_profiling_(other.enable_profiling_),
      next_event_id_(other.next_event_id_)
{
    // 清空源对象
    other.stream_usage_count_.clear();
    other.events_.clear();
    other.enable_profiling_ = false;
    other.next_event_id_    = 0;
}

AdvancedStreamManager &AdvancedStreamManager::operator=(AdvancedStreamManager &&other) noexcept
{
    if (this != &other)
    {
        // 清理当前对象
        for (auto &event : events_)
        {
            if (event != nullptr)
            {
                cudaEventDestroy(event);
            }
        }

        // 移动资源
        stream_manager_     = std::move(other.stream_manager_);
        stream_usage_count_ = std::move(other.stream_usage_count_);
        events_             = std::move(other.events_);
        enable_profiling_   = other.enable_profiling_;
        next_event_id_      = other.next_event_id_;

        // 清空源对象
        other.stream_usage_count_.clear();
        other.events_.clear();
        other.enable_profiling_ = false;
        other.next_event_id_    = 0;
    }
    return *this;
}

cudaStream_t AdvancedStreamManager::get_optimal_stream()
{
    // 找到使用次数最少的流
    auto min_it          = std::min_element(stream_usage_count_.begin(), stream_usage_count_.end());
    size_t optimal_index = std::distance(stream_usage_count_.begin(), min_it);

    // 增加使用计数
    stream_usage_count_[optimal_index]++;

    return stream_manager_.get_stream(optimal_index);
}

cudaStream_t AdvancedStreamManager::get_stream(size_t index) const
{
    return stream_manager_.get_stream(index);
}

size_t AdvancedStreamManager::record_event(size_t stream_id)
{
    if (stream_id >= stream_manager_.num_streams())
    {
        THROW_INVALID_ARG("Stream index {} out of range [0, {})", stream_id, stream_manager_.num_streams());
    }

    size_t event_id = next_event_id_ % events_.size();
    next_event_id_++;

    cudaError_t err = cudaEventRecord(events_[event_id], stream_manager_.get_stream(stream_id));
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("Failed to record advanced event {} for stream {}: {}", event_id, stream_id,
                            cudaGetErrorString(err));
    }

    return event_id;
}

void AdvancedStreamManager::wait_for_event(size_t event_id)
{
    if (event_id >= events_.size())
    {
        THROW_INVALID_ARG("Event index {} out of range [0, {})", event_id, events_.size());
    }

    cudaError_t err = cudaEventSynchronize(events_[event_id]);
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("Failed to synchronize advanced event {}: {}", event_id, cudaGetErrorString(err));
    }
}

void AdvancedStreamManager::wait_for_event_in_stream(size_t current_stream, size_t wait_for_event)
{
    if (current_stream >= stream_manager_.num_streams())
    {
        THROW_INVALID_ARG("Current stream index {} out of range [0, {})", current_stream,
                          stream_manager_.num_streams());
    }
    if (wait_for_event >= events_.size())
    {
        THROW_INVALID_ARG("Wait for event index {} out of range [0, {})", wait_for_event, events_.size());
    }

    cudaError_t err = cudaStreamWaitEvent(stream_manager_.get_stream(current_stream), events_[wait_for_event], 0);
    if (err != cudaSuccess)
    {
        THROW_RUNTIME_ERROR("Failed to wait for event {} in stream {}: {}", wait_for_event, current_stream,
                            cudaGetErrorString(err));
    }
}

void AdvancedStreamManager::synchronize_all()
{
    stream_manager_.synchronize_all();
}

std::vector<size_t> AdvancedStreamManager::get_stream_usage_stats() const
{
    return stream_usage_count_;
}

void AdvancedStreamManager::reset_stats()
{
    std::fill(stream_usage_count_.begin(), stream_usage_count_.end(), 0);
    next_event_id_ = 0;
}

// ============================================================================
// StreamParallelExecutor 实现
// ============================================================================

StreamParallelExecutor::StreamParallelExecutor(size_t num_streams) : stream_manager_(num_streams) {}

StreamParallelExecutor::~StreamParallelExecutor() = default;

template <typename TaskType>
void StreamParallelExecutor::execute_parallel(const std::vector<TaskType> &tasks)
{
    // 将任务分配到不同的流中
    for (size_t i = 0; i < tasks.size(); ++i)
    {
        cudaStream_t stream = stream_manager_.get_optimal_stream();
        // 这里需要根据具体的任务类型来执行
        // 由于TaskType是模板参数，具体实现需要在特化版本中提供
    }
}

template <typename StageType>
void StreamParallelExecutor::execute_pipeline(const std::vector<StageType> &stages)
{
    // 流水线执行：将任务分解为多个阶段，在不同流中执行
    for (size_t i = 0; i < stages.size(); ++i)
    {
        cudaStream_t stream = stream_manager_.get_optimal_stream();
        // 这里需要根据具体的阶段类型来执行
        // 由于StageType是模板参数，具体实现需要在特化版本中提供
    }
}

void StreamParallelExecutor::wait_all()
{
    stream_manager_.synchronize_all();
}

}  // namespace cuda
}  // namespace origin
