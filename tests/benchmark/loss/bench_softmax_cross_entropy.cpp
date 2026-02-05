#include <vector>
#include "benchmark/common/benchmark_framework.h"
#include "benchmark/common/timer.h"
#include "origin.h"
#include "origin/operators/loss/softmax_cross_entropy.h"

using namespace origin;
using namespace origin::benchmark;
namespace F = origin::functional;

/**
 * @brief SoftmaxCrossEntropy算子基准测试类
 */
class SoftmaxCrossEntropyBenchmark : public BenchmarkFramework
{
public:
    double run_benchmark(const BenchmarkConfig &config) override
    {
        if (config.shapes.size() != 2)
        {
            THROW_RUNTIME_ERROR("SoftmaxCrossEntropy benchmark requires exactly 2 shapes, got {}", config.shapes.size());
        }

        const Shape &x_shape = config.shapes[0];
        const Shape &target_shape = config.shapes[1];
        
        // x: (N, C), target: (N,)
        if (x_shape.ndims() != 2)
        {
            THROW_RUNTIME_ERROR("SoftmaxCrossEntropy x must be 2D (N, C), got {}", x_shape.to_string());
        }
        if (target_shape.ndims() != 1)
        {
            THROW_RUNTIME_ERROR("SoftmaxCrossEntropy target must be 1D (N,), got {}", target_shape.to_string());
        }
        if (x_shape[0] != target_shape[0])
        {
            THROW_RUNTIME_ERROR("SoftmaxCrossEntropy x and target must have same batch size, got {} and {}",
                                x_shape[0], target_shape[0]);
        }

        size_t x_numel = x_shape.elements();
        std::vector<float> x_data(x_numel);
        for (size_t i = 0; i < x_numel; ++i)
        {
            x_data[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        size_t target_numel = target_shape.elements();
        std::vector<int> target_data(target_numel);
        int num_classes = x_shape[1];
        for (size_t i = 0; i < target_numel; ++i)
        {
            target_data[i] = rand() % num_classes;
        }

        auto x = Tensor(x_data, x_shape, origin::dtype(config.dtype).device(config.device));
        auto target = Tensor(target_data, target_shape, origin::dtype(DataType::kInt32).device(config.device));

        // 预热
        for (int i = 0; i < config.warmup_cnt; ++i)
        {
            auto result = F::softmax_cross_entropy(x, target);
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
            auto result = F::softmax_cross_entropy(x, target);
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
            {Shape({32, 10}), Shape({32})},      // (N, C), (N,)
            {Shape({64, 100}), Shape({64})},
            {Shape({128, 1000}), Shape({128})},
            {Shape({256, 10}), Shape({256})},
            {Shape({512, 100}), Shape({512})},
        };
    }

    size_t get_required_shapes_count() const override { return 2; }

    void validate_shapes(const std::vector<Shape> &shapes) const override
    {
        if (shapes.size() != 2)
        {
            THROW_RUNTIME_ERROR("SoftmaxCrossEntropy requires exactly 2 shapes, got {}", shapes.size());
        }
        if (shapes[0].ndims() != 2)
        {
            THROW_RUNTIME_ERROR("SoftmaxCrossEntropy x must be 2D (N, C), got {}", shapes[0].to_string());
        }
        if (shapes[1].ndims() != 1)
        {
            THROW_RUNTIME_ERROR("SoftmaxCrossEntropy target must be 1D (N,), got {}", shapes[1].to_string());
        }
        if (shapes[0][0] != shapes[1][0])
        {
            THROW_RUNTIME_ERROR("SoftmaxCrossEntropy x and target must have same batch size, got {} and {}",
                                shapes[0][0], shapes[1][0]);
        }
    }

    std::string get_operator_name() const override { return "SoftmaxCrossEntropy"; }
};

int main(int argc, char *argv[])
{
    SoftmaxCrossEntropyBenchmark benchmark;
    return benchmark.run(argc, argv);
}
