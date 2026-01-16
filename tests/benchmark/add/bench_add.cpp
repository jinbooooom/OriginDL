#include <vector>
#include "origin.h"
#include "origin/operators/math/add.h"
#include "benchmark/common/benchmark_framework.h"
#include "benchmark/common/timer.h"

using namespace origin;
using namespace origin::benchmark;
namespace F = origin::functional;

/**
 * @brief Add算子基准测试类
 */
class AddBenchmark : public BenchmarkFramework {
public:
    double run_benchmark(const BenchmarkConfig& config) override
    {
        // Add算子只需要一个shape
        if (config.shapes.size() != 1)
        {
            THROW_RUNTIME_ERROR("Add benchmark requires exactly 1 shape, got {}", config.shapes.size());
        }
        
        const Shape& shape = config.shapes[0];
        
        // 创建输入张量
        size_t numel = shape.elements();
        std::vector<float> data0(numel, 1.0f);
        std::vector<float> data1(numel, 2.0f);
        
        auto x0 = Tensor(data0, shape, origin::dtype(config.dtype).device(config.device));
        auto x1 = Tensor(data1, shape, origin::dtype(config.dtype).device(config.device));
        
        // 预热
        for (int i = 0; i < config.warmup_cnt; ++i)
        {
            auto result = F::add(x0, x1);
            // 确保计算完成（对于CUDA，需要同步）
            if (config.device.type() == DeviceType::kCUDA)
            {
                cuda::synchronize();
            }
        }
        
        // 正式测试
        Timer timer;
        timer.start();
        
        for (int i = 0; i < config.repeat_cnt; ++i)
        {
            auto result = F::add(x0, x1);
            // 确保计算完成
            if (config.device.type() == DeviceType::kCUDA)
            {
                cuda::synchronize();
            }
        }
        
        double total_time_us = timer.elapsed_us();
        return total_time_us / config.repeat_cnt;
    }


    size_t get_required_shapes_count() const override
    {
        return 1;
    }

    void validate_shapes(const std::vector<Shape>& shapes) const override
    {
        if (shapes.size() != 1)
        {
            THROW_RUNTIME_ERROR("Add requires exactly 1 shape, got {}", shapes.size());
        }
    }

    std::string get_operator_name() const override
    {
        return "Add";
    }
};

int main(int argc, char* argv[])
{
    AddBenchmark benchmark;
    return benchmark.run(argc, argv);
}
