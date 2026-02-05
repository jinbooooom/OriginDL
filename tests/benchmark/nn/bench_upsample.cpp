#include <vector>
#include "benchmark/common/benchmark_framework.h"
#include "benchmark/common/timer.h"
#include "origin.h"
#include "origin/operators/nn/upsample.h"

using namespace origin;
using namespace origin::benchmark;
namespace F = origin::functional;

/**
 * @brief Upsample算子基准测试类
 */
class UpsampleBenchmark : public BenchmarkFramework
{
public:
    double run_benchmark(const BenchmarkConfig &config) override
    {
        if (config.shapes.size() != 1)
        {
            THROW_RUNTIME_ERROR("Upsample benchmark requires exactly 1 shape, got {}", config.shapes.size());
        }

        const Shape &shape = config.shapes[0];
        
        // Upsample需要4D输入 (N, C, H, W)
        if (shape.ndims() != 4)
        {
            THROW_RUNTIME_ERROR("Upsample requires 4D shape (N, C, H, W), got {}", shape.to_string());
        }

        size_t numel = shape.elements();
        std::vector<float> data(numel, 1.0f);

        auto x = Tensor(data, shape, origin::dtype(config.dtype).device(config.device));

        // 使用默认的缩放因子 (2.0, 2.0) 和 nearest模式
        std::pair<float, float> scale_factor(2.0f, 2.0f);
        std::string mode = "nearest";

        // 预热
        for (int i = 0; i < config.warmup_cnt; ++i)
        {
            auto result = F::upsample(x, mode, scale_factor);
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
            auto result = F::upsample(x, mode, scale_factor);
        }
        if (config.device.type() == DeviceType::kCUDA)
        {
            cuda::synchronize();
        }

        return timer.elapsed_us() / config.repeat_cnt;
    }

    std::vector<std::vector<Shape>> get_default_shapes() const override
    {
        return {
            {Shape({1, 1, 3, 3})},
            {Shape({1, 3, 10, 10})},
            {Shape({1, 3, 32, 32})},
            {Shape({1, 64, 64, 64})},
            {Shape({4, 3, 224, 224})},
        };
    }

    size_t get_required_shapes_count() const override { return 1; }

    void validate_shapes(const std::vector<Shape> &shapes) const override
    {
        if (shapes.size() != 1)
        {
            THROW_RUNTIME_ERROR("Upsample requires exactly 1 shape, got {}", shapes.size());
        }
        if (shapes[0].ndims() != 4)
        {
            THROW_RUNTIME_ERROR("Upsample requires 4D shape (N, C, H, W), got {}", shapes[0].to_string());
        }
    }

    std::string get_operator_name() const override { return "Upsample"; }
};

int main(int argc, char *argv[])
{
    UpsampleBenchmark benchmark;
    return benchmark.run(argc, argv);
}
