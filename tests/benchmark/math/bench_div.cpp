#include <vector>
#include "benchmark/common/benchmark_framework.h"
#include "benchmark/common/timer.h"
#include "origin.h"
#include "origin/operators/math/div.h"

using namespace origin;
using namespace origin::benchmark;
namespace F = origin::functional;

/**
 * @brief Div算子基准测试类
 */
class DivBenchmark : public BenchmarkFramework
{
public:
    double run_benchmark(const BenchmarkConfig &config) override
    {
        if (config.shapes.size() != 1)
        {
            THROW_RUNTIME_ERROR("Div benchmark requires exactly 1 shape, got {}", config.shapes.size());
        }

        const Shape &shape = config.shapes[0];
        size_t numel       = shape.elements();
        std::vector<float> data0(numel, 1.0f);
        std::vector<float> data1(numel, 2.0f);

        auto x0 = Tensor(data0, shape, origin::dtype(config.dtype).device(config.device));
        auto x1 = Tensor(data1, shape, origin::dtype(config.dtype).device(config.device));

        // 预热
        if (config.inplace)
        {
            for (int i = 0; i < config.warmup_cnt; ++i)
            {
                F::div_(x0, x1);
            }
        }
        else
        {
            for (int i = 0; i < config.warmup_cnt; ++i)
            {
                auto result = F::div(x0, x1);
            }
        }
        if (config.device.type() == DeviceType::kCUDA)
        {
            cuda::synchronize();
        }

        // 正式测试
        Timer timer;
        timer.start();

        if (config.inplace)
        {
            for (int i = 0; i < config.repeat_cnt; ++i)
            {
                F::div_(x0, x1);
            }
        }
        else
        {
            for (int i = 0; i < config.repeat_cnt; ++i)
            {
                auto result = F::div(x0, x1);
            }
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
            THROW_RUNTIME_ERROR("Div requires exactly 1 shape, got {}", shapes.size());
        }
    }

    std::string get_operator_name() const override { return "Div"; }
};

int main(int argc, char *argv[])
{
    DivBenchmark benchmark;
    return benchmark.run(argc, argv);
}
