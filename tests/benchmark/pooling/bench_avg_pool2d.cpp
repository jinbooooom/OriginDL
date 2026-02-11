#include <vector>
#include "benchmark/common/benchmark_framework.h"
#include "benchmark/common/timer.h"
#include "origin.h"
#include "origin/operators/pooling/avg_pool2d.h"

using namespace origin;
using namespace origin::benchmark;
namespace F = origin::functional;

/**
 * @brief AvgPool2d算子基准测试类
 */
class AvgPool2dBenchmark : public BenchmarkFramework
{
public:
    double run_benchmark(const BenchmarkConfig &config) override
    {
        if (config.shapes.size() != 1)
        {
            THROW_RUNTIME_ERROR("AvgPool2d benchmark requires exactly 1 shape, got {}", config.shapes.size());
        }

        const Shape &shape = config.shapes[0];

        // AvgPool2d需要4D输入 (N, C, H, W)
        if (shape.ndims() != 4)
        {
            THROW_RUNTIME_ERROR("AvgPool2d requires 4D shape (N, C, H, W), got {}", shape.to_string());
        }

        size_t numel = shape.elements();
        std::vector<float> data(numel, 1.0f);

        auto x = Tensor(data, shape, origin::dtype(config.dtype).device(config.device));

        // 使用默认的kernel_size, stride, pad
        int kernel_size = 2;
        int stride      = 2;
        int pad         = 0;

        // 预热
        for (int i = 0; i < config.warmup_cnt; ++i)
        {
            auto result = F::avg_pool2d(x, kernel_size, stride, pad);
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
            auto result = F::avg_pool2d(x, kernel_size, stride, pad);
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
            {Shape({1, 1, 3, 3})},    {Shape({1, 3, 10, 10})},   {Shape({1, 3, 32, 32})},
            {Shape({1, 64, 64, 64})}, {Shape({4, 3, 224, 224})},
        };
    }

    size_t get_required_shapes_count() const override { return 1; }

    void validate_shapes(const std::vector<Shape> &shapes) const override
    {
        if (shapes.size() != 1)
        {
            THROW_RUNTIME_ERROR("AvgPool2d requires exactly 1 shape, got {}", shapes.size());
        }
        if (shapes[0].ndims() != 4)
        {
            THROW_RUNTIME_ERROR("AvgPool2d requires 4D shape (N, C, H, W), got {}", shapes[0].to_string());
        }
    }

    std::string get_operator_name() const override { return "AvgPool2d"; }
};

int main(int argc, char *argv[])
{
    AvgPool2dBenchmark benchmark;
    return benchmark.run(argc, argv);
}
