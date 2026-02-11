#include <vector>
#include "benchmark/common/benchmark_framework.h"
#include "benchmark/common/timer.h"
#include "origin.h"
#include "origin/operators/activation/sigmoid.h"

using namespace origin;
using namespace origin::benchmark;
namespace F = origin::functional;

/**
 * @brief Sigmoid算子基准测试类
 */
class SigmoidBenchmark : public BenchmarkFramework
{
public:
    double run_benchmark(const BenchmarkConfig &config) override
    {
        if (config.shapes.size() != 1)
        {
            THROW_RUNTIME_ERROR("Sigmoid benchmark requires exactly 1 shape, got {}", config.shapes.size());
        }

        const Shape &shape = config.shapes[0];
        size_t numel       = shape.elements();
        std::vector<float> data(numel, 0.0f);

        auto x = Tensor(data, shape, origin::dtype(config.dtype).device(config.device));

        // Sigmoid不支持inplace操作
        if (config.inplace)
        {
            THROW_RUNTIME_ERROR("Sigmoid operator does not support inplace operations");
        }

        // 预热
        for (int i = 0; i < config.warmup_cnt; ++i)
        {
            auto result = F::sigmoid(x);
        }
        if (config.device.type() == DeviceType::kCUDA)
        {
            cuda::synchronize();
        }

        // 正式测试
        Timer timer;
        timer.start();

        for (int i = 0; i < config.repeat_cnt; ++i)
        {
            auto result = F::sigmoid(x);
        }
        if (config.device.type() == DeviceType::kCUDA)
        {
            cuda::synchronize();
        }

        return timer.elapsed_us() / config.repeat_cnt;
    }

    size_t get_required_shapes_count() const override { return 1; }

    void validate_shapes(const std::vector<Shape> &shapes) const override
    {
        if (shapes.size() != 1)
        {
            THROW_RUNTIME_ERROR("Sigmoid requires exactly 1 shape, got {}", shapes.size());
        }
    }

    std::string get_operator_name() const override { return "Sigmoid"; }
};

int main(int argc, char *argv[])
{
    SigmoidBenchmark benchmark;
    return benchmark.run(argc, argv);
}
