#ifndef __BENCHMARK_COMMON_TIMER_H__
#define __BENCHMARK_COMMON_TIMER_H__

#include <chrono>

/**
 * @brief 高精度计时器
 */
class Timer
{
public:
    void start() { start_time_ = std::chrono::high_resolution_clock::now(); }
    
    double elapsed_us() const
    {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time_);
        return static_cast<double>(duration.count());  // 返回微秒
    }

private:
    std::chrono::high_resolution_clock::time_point start_time_;
};

#endif  // __BENCHMARK_COMMON_TIMER_H__
