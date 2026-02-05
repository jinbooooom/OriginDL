#include <vector>
#include "benchmark/common/benchmark_framework.h"
#include "benchmark/common/timer.h"
#include "origin.h"
#include "origin/operators/shape/cat.h"

using namespace origin;
using namespace origin::benchmark;
namespace F = origin::functional;

/**
 * @brief Cat算子基准测试类
 */
class CatBenchmark : public BenchmarkFramework
{
public:
    double run_benchmark(const BenchmarkConfig &config) override
    {
        if (config.shapes.size() != 1)
        {
            THROW_RUNTIME_ERROR("Cat benchmark requires exactly 1 shape, got {}", config.shapes.size());
        }

        const Shape &shape = config.shapes[0];
        size_t numel = shape.elements();
        
        // Cat需要至少2个tensor，创建3个相同shape的tensor
        std::vector<float> data0(numel, 1.0f);
        std::vector<float> data1(numel, 2.0f);
        std::vector<float> data2(numel, 3.0f);

        auto x0 = Tensor(data0, shape, origin::dtype(config.dtype).device(config.device));
        auto x1 = Tensor(data1, shape, origin::dtype(config.dtype).device(config.device));
        auto x2 = Tensor(data2, shape, origin::dtype(config.dtype).device(config.device));

        // 在最后一个维度上cat
        int dim = static_cast<int>(shape.ndims()) - 1;

        // 预热
        for (int i = 0; i < config.warmup_cnt; ++i)
        {
            auto result = F::cat({x0, x1, x2}, dim);
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
            auto result = F::cat({x0, x1, x2}, dim);
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
            THROW_RUNTIME_ERROR("Cat requires exactly 1 shape, got {}", shapes.size());
        }
        if (shapes[0].ndims() == 0)
        {
            THROW_RUNTIME_ERROR("Cat requires at least 1D shape");
        }
    }

    std::string get_operator_name() const override { return "Cat"; }
};

int main(int argc, char *argv[])
{
    CatBenchmark benchmark;
    return benchmark.run(argc, argv);
}
