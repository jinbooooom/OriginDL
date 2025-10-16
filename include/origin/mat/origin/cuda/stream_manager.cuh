#ifndef __ORIGIN_DL_STREAM_MANAGER_H__
#define __ORIGIN_DL_STREAM_MANAGER_H__

#include <cuda_runtime.h>
#include <memory>
#include <vector>

namespace origin
{
namespace cuda
{

/**
 * @brief CUDA流管理器类
 * @details 管理多个CUDA流，实现流并行优化，提高GPU利用率
 */
class StreamManager
{
public:
    /**
     * @brief 构造函数
     * @param num_streams 流数量，默认为4
     * @details 创建指定数量的CUDA流，用于并行执行
     */
    explicit StreamManager(size_t num_streams = 4);

    /**
     * @brief 析构函数
     * @details 销毁所有CUDA流
     */
    ~StreamManager();

    // 禁用拷贝构造和赋值操作
    StreamManager(const StreamManager &)            = delete;
    StreamManager &operator=(const StreamManager &) = delete;

    // 支持移动构造和赋值
    StreamManager(StreamManager &&other) noexcept;
    StreamManager &operator=(StreamManager &&other) noexcept;

    /**
     * @brief 获取下一个可用流
     * @return CUDA流句柄
     * @details 轮询方式返回流，实现负载均衡
     */
    cudaStream_t get_next_stream();

    /**
     * @brief 获取指定索引的流
     * @param index 流索引
     * @return CUDA流句柄
     * @throws std::out_of_range 当索引超出范围时抛出
     */
    cudaStream_t get_stream(size_t index) const;

    /**
     * @brief 记录事件到指定流
     * @param stream_id 流索引
     * @details 在当前流中记录一个事件，用于流间同步
     */
    void record_event(size_t stream_id);

    /**
     * @brief 等待指定事件完成
     * @param event_id 事件索引
     * @details 阻塞等待指定事件完成
     */
    void wait_for_event(size_t event_id);

    /**
     * @brief 在流间建立依赖关系
     * @param current_stream 当前流索引
     * @param wait_for_stream 等待的流索引
     * @details 让当前流等待指定流的事件完成
     */
    void wait_for_stream(size_t current_stream, size_t wait_for_stream);

    /**
     * @brief 同步所有流
     * @details 等待所有流中的操作完成
     */
    void synchronize_all();

    /**
     * @brief 获取流数量
     * @return 流数量
     */
    size_t num_streams() const { return streams_.size(); }

    /**
     * @brief 检查流是否有效
     * @return 如果所有流都有效返回true，否则返回false
     */
    bool is_valid() const;

private:
    std::vector<cudaStream_t> streams_;  ///< CUDA流数组
    std::vector<cudaEvent_t> events_;    ///< CUDA事件数组
    size_t current_stream_;              ///< 当前流索引
};

/**
 * @brief 高级流管理器类
 * @details 提供更高级的流管理功能，包括自动负载均衡和性能监控
 */
class AdvancedStreamManager
{
public:
    /**
     * @brief 构造函数
     * @param num_streams 流数量，默认为4
     * @param enable_profiling 是否启用性能分析，默认为false
     */
    explicit AdvancedStreamManager(size_t num_streams = 4, bool enable_profiling = false);

    /**
     * @brief 析构函数
     */
    ~AdvancedStreamManager();

    // 禁用拷贝构造和赋值操作
    AdvancedStreamManager(const AdvancedStreamManager &)            = delete;
    AdvancedStreamManager &operator=(const AdvancedStreamManager &) = delete;

    // 支持移动构造和赋值
    AdvancedStreamManager(AdvancedStreamManager &&other) noexcept;
    AdvancedStreamManager &operator=(AdvancedStreamManager &&other) noexcept;

    /**
     * @brief 获取最优流
     * @return CUDA流句柄
     * @details 根据负载情况选择最优的流
     */
    cudaStream_t get_optimal_stream();

    /**
     * @brief 获取指定索引的流
     * @param index 流索引
     * @return CUDA流句柄
     */
    cudaStream_t get_stream(size_t index) const;

    /**
     * @brief 记录事件
     * @param stream_id 流索引
     * @return 事件索引
     */
    size_t record_event(size_t stream_id);

    /**
     * @brief 等待事件完成
     * @param event_id 事件索引
     */
    void wait_for_event(size_t event_id);

    /**
     * @brief 建立流依赖
     * @param current_stream 当前流索引
     * @param wait_for_event 等待的事件索引
     */
    void wait_for_event_in_stream(size_t current_stream, size_t wait_for_event);

    /**
     * @brief 同步所有流
     */
    void synchronize_all();

    /**
     * @brief 获取流统计信息
     * @return 包含每个流使用次数的向量
     */
    std::vector<size_t> get_stream_usage_stats() const;

    /**
     * @brief 重置统计信息
     */
    void reset_stats();

    /**
     * @brief 获取流数量
     * @return 流数量
     */
    size_t num_streams() const { return stream_manager_.num_streams(); }

    /**
     * @brief 检查是否启用性能分析
     * @return 如果启用性能分析返回true
     */
    bool is_profiling_enabled() const { return enable_profiling_; }

private:
    StreamManager stream_manager_;            ///< 基础流管理器
    std::vector<size_t> stream_usage_count_;  ///< 流使用计数
    std::vector<cudaEvent_t> events_;         ///< 事件数组
    bool enable_profiling_;                   ///< 是否启用性能分析
    size_t next_event_id_;                    ///< 下一个事件ID
};

/**
 * @brief 流并行执行器
 * @details 提供高级的流并行执行功能，支持流水线处理
 */
class StreamParallelExecutor
{
public:
    /**
     * @brief 构造函数
     * @param num_streams 流数量，默认为4
     */
    explicit StreamParallelExecutor(size_t num_streams = 4);

    /**
     * @brief 析构函数
     */
    ~StreamParallelExecutor();

    /**
     * @brief 并行执行多个任务
     * @tparam TaskType 任务类型
     * @param tasks 任务向量
     * @details 将任务分配到不同的流中并行执行
     */
    template <typename TaskType>
    void execute_parallel(const std::vector<TaskType> &tasks);

    /**
     * @brief 流水线执行
     * @tparam StageType 阶段类型
     * @param stages 阶段向量
     * @details 将任务分解为多个阶段，在不同流中流水线执行
     */
    template <typename StageType>
    void execute_pipeline(const std::vector<StageType> &stages);

    /**
     * @brief 等待所有任务完成
     */
    void wait_all();

    /**
     * @brief 获取流管理器
     * @return 流管理器的引用
     */
    AdvancedStreamManager &get_stream_manager() { return stream_manager_; }

private:
    AdvancedStreamManager stream_manager_;  ///< 高级流管理器
};

}  // namespace cuda
}  // namespace origin

#endif  // __ORIGIN_DL_STREAM_MANAGER_H__
