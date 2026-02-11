#include <vector>
#include "benchmark/common/benchmark_framework.h"
#include "benchmark/common/timer.h"
#include "origin.h"
#include "origin/mat/scalar.h"
#include "origin/operators/math/pow.h"

using namespace origin;
using namespace origin::benchmark;
namespace F = origin::functional;

/**
 * @brief Pow算子基准测试类
 */
class PowBenchmark : public BenchmarkFramework
{
public:
    double run_benchmark(const BenchmarkConfig &config) override
    {
        if (config.shapes.size() != 1)
        {
            THROW_RUNTIME_ERROR("Pow benchmark requires exactly 1 shape, got {}", config.shapes.size());
        }

        const Shape &shape = config.shapes[0];
        size_t numel       = shape.elements();
        std::vector<float> data(numel, 1.0f);

        auto x = Tensor(data, shape, origin::dtype(config.dtype).device(config.device));
        Scalar exponent(2.0f);

        // 预热
        if (config.inplace)
        {
            for (int i = 0; i < config.warmup_cnt; ++i)
            {
                F::pow_(x, exponent);
            }
        }
        else
        {
            for (int i = 0; i < config.warmup_cnt; ++i)
            {
                auto result = F::pow(x, exponent);
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
                F::pow_(x, exponent);
            }
        }
        else
        {
            for (int i = 0; i < config.repeat_cnt; ++i)
            {
                auto result = F::pow(x, exponent);
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
            THROW_RUNTIME_ERROR("Pow requires exactly 1 shape, got {}", shapes.size());
        }
    }

    std::string get_operator_name() const override { return "Pow"; }
};

int main(int argc, char *argv[])
{
    PowBenchmark benchmark;
    return benchmark.run(argc, argv);
}
