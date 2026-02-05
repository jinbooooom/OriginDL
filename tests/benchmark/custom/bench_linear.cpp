#include <vector>
#include "benchmark/common/benchmark_framework.h"
#include "benchmark/common/timer.h"
#include "origin.h"
#include "origin/operators/custom/linear.h"

using namespace origin;
using namespace origin::benchmark;
namespace F = origin::functional;

/**
 * @brief Linear算子基准测试类
 */
class LinearBenchmark : public BenchmarkFramework
{
public:
    double run_benchmark(const BenchmarkConfig &config) override
    {
        if (config.shapes.size() != 3)
        {
            THROW_RUNTIME_ERROR("Linear benchmark requires exactly 3 shapes, got {}", config.shapes.size());
        }

        const Shape &x_shape = config.shapes[0];
        const Shape &weight_shape = config.shapes[1];
        const Shape &bias_shape = config.shapes[2];
        
        // x: (N, in_features), weight: (out_features, in_features), bias: (out_features,)
        if (x_shape.ndims() != 2)
        {
            THROW_RUNTIME_ERROR("Linear x must be 2D (N, in_features), got {}", x_shape.to_string());
        }
        if (weight_shape.ndims() != 2)
        {
            THROW_RUNTIME_ERROR("Linear weight must be 2D (out_features, in_features), got {}", weight_shape.to_string());
        }
        if (bias_shape.ndims() != 1)
        {
            THROW_RUNTIME_ERROR("Linear bias must be 1D (out_features,), got {}", bias_shape.to_string());
        }
        
        int in_features = x_shape[1];
        int out_features = weight_shape[0];
        if (weight_shape[1] != in_features)
        {
            THROW_RUNTIME_ERROR("Linear weight shape mismatch: weight has in_features={}, but x has in_features={}",
                                weight_shape[1], in_features);
        }
        if (bias_shape[0] != out_features)
        {
            THROW_RUNTIME_ERROR("Linear bias shape mismatch: bias has out_features={}, but weight has out_features={}",
                                bias_shape[0], out_features);
        }

        size_t x_numel = x_shape.elements();
        std::vector<float> x_data(x_numel);
        for (size_t i = 0; i < x_numel; ++i)
        {
            x_data[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        size_t weight_numel = weight_shape.elements();
        std::vector<float> weight_data(weight_numel);
        for (size_t i = 0; i < weight_numel; ++i)
        {
            weight_data[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        size_t bias_numel = bias_shape.elements();
        std::vector<float> bias_data(bias_numel);
        for (size_t i = 0; i < bias_numel; ++i)
        {
            bias_data[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        auto x = Tensor(x_data, x_shape, origin::dtype(config.dtype).device(config.device));
        auto weight = Tensor(weight_data, weight_shape, origin::dtype(config.dtype).device(config.device));
        auto bias = Tensor(bias_data, bias_shape, origin::dtype(config.dtype).device(config.device));

        // 预热
        for (int i = 0; i < config.warmup_cnt; ++i)
        {
            auto result = F::custom_linear(x, weight, bias, in_features, out_features, true);
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
            auto result = F::custom_linear(x, weight, bias, in_features, out_features, true);
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
            {Shape({32, 128}), Shape({256, 128}), Shape({256})},  // x, weight, bias
            {Shape({64, 256}), Shape({512, 256}), Shape({512})},
            {Shape({128, 512}), Shape({1024, 512}), Shape({1024})},
            {Shape({256, 1024}), Shape({2048, 1024}), Shape({2048})},
            {Shape({512, 2048}), Shape({4096, 2048}), Shape({4096})},
        };
    }

    size_t get_required_shapes_count() const override { return 3; }

    void validate_shapes(const std::vector<Shape> &shapes) const override
    {
        if (shapes.size() != 3)
        {
            THROW_RUNTIME_ERROR("Linear requires exactly 3 shapes, got {}", shapes.size());
        }
        const Shape &x_shape = shapes[0];
        const Shape &weight_shape = shapes[1];
        const Shape &bias_shape = shapes[2];
        
        if (x_shape.ndims() != 2)
        {
            THROW_RUNTIME_ERROR("Linear x must be 2D (N, in_features), got {}", x_shape.to_string());
        }
        if (weight_shape.ndims() != 2)
        {
            THROW_RUNTIME_ERROR("Linear weight must be 2D (out_features, in_features), got {}", weight_shape.to_string());
        }
        if (bias_shape.ndims() != 1)
        {
            THROW_RUNTIME_ERROR("Linear bias must be 1D (out_features,), got {}", bias_shape.to_string());
        }
        
        int in_features = x_shape[1];
        int out_features = weight_shape[0];
        if (weight_shape[1] != in_features)
        {
            THROW_RUNTIME_ERROR("Linear weight shape mismatch: weight has in_features={}, but x has in_features={}",
                                weight_shape[1], in_features);
        }
        if (bias_shape[0] != out_features)
        {
            THROW_RUNTIME_ERROR("Linear bias shape mismatch: bias has out_features={}, but weight has out_features={}",
                                bias_shape[0], out_features);
        }
    }

    std::string get_operator_name() const override { return "Linear"; }
};

int main(int argc, char *argv[])
{
    LinearBenchmark benchmark;
    return benchmark.run(argc, argv);
}
