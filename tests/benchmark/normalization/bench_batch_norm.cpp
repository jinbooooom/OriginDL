#include <vector>
#include "benchmark/common/benchmark_framework.h"
#include "benchmark/common/timer.h"
#include "origin.h"
#include "origin/operators/normalization/batch_norm.h"

using namespace origin;
using namespace origin::benchmark;
namespace F = origin::functional;

/**
 * @brief BatchNorm算子基准测试类
 */
class BatchNormBenchmark : public BenchmarkFramework
{
public:
    double run_benchmark(const BenchmarkConfig &config) override
    {
        if (config.shapes.size() != 5)
        {
            THROW_RUNTIME_ERROR("BatchNorm benchmark requires exactly 5 shapes, got {}", config.shapes.size());
        }

        const Shape &x_shape            = config.shapes[0];
        const Shape &gamma_shape        = config.shapes[1];
        const Shape &beta_shape         = config.shapes[2];
        const Shape &running_mean_shape = config.shapes[3];
        const Shape &running_var_shape  = config.shapes[4];

        // x: (N, C, H, W), gamma/beta/running_mean/running_var: (C,)
        if (x_shape.ndims() != 4)
        {
            THROW_RUNTIME_ERROR("BatchNorm x must be 4D (N, C, H, W), got {}", x_shape.to_string());
        }
        if (gamma_shape.ndims() != 1 || beta_shape.ndims() != 1 || running_mean_shape.ndims() != 1 ||
            running_var_shape.ndims() != 1)
        {
            THROW_RUNTIME_ERROR("BatchNorm gamma/beta/running_mean/running_var must be 1D (C,)");
        }

        int C = x_shape[1];
        if (gamma_shape[0] != C || beta_shape[0] != C || running_mean_shape[0] != C || running_var_shape[0] != C)
        {
            THROW_RUNTIME_ERROR(
                "BatchNorm channel dimension mismatch: x has C={}, but gamma/beta/running_mean/running_var have "
                "different sizes",
                C);
        }

        size_t x_numel = x_shape.elements();
        std::vector<float> x_data(x_numel);
        for (size_t i = 0; i < x_numel; ++i)
        {
            x_data[i] = static_cast<float>(rand()) / RAND_MAX;
        }

        size_t C_size = C;
        std::vector<float> gamma_data(C_size, 1.0f);
        std::vector<float> beta_data(C_size, 0.0f);
        std::vector<float> running_mean_data(C_size, 0.0f);
        std::vector<float> running_var_data(C_size, 1.0f);

        auto x     = Tensor(x_data, x_shape, origin::dtype(config.dtype).device(config.device));
        auto gamma = Tensor(gamma_data, gamma_shape, origin::dtype(config.dtype).device(config.device));
        auto beta  = Tensor(beta_data, beta_shape, origin::dtype(config.dtype).device(config.device));
        auto running_mean =
            Tensor(running_mean_data, running_mean_shape, origin::dtype(config.dtype).device(config.device));
        auto running_var =
            Tensor(running_var_data, running_var_shape, origin::dtype(config.dtype).device(config.device));

        bool training  = false;  // 测试模式
        float eps      = 1e-5f;
        float momentum = 0.1f;
        int num_dims   = 4;

        // 预热
        for (int i = 0; i < config.warmup_cnt; ++i)
        {
            auto result = F::batch_norm(x, gamma, beta, running_mean, running_var, training, eps, momentum, num_dims);
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
            auto result = F::batch_norm(x, gamma, beta, running_mean, running_var, training, eps, momentum, num_dims);
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
            {Shape({1, 3, 32, 32}), Shape({3}), Shape({3}), Shape({3}),
             Shape({3})},  // x, gamma, beta, running_mean, running_var
            {Shape({4, 64, 64, 64}), Shape({64}), Shape({64}), Shape({64}), Shape({64})},
            {Shape({8, 128, 32, 32}), Shape({128}), Shape({128}), Shape({128}), Shape({128})},
            {Shape({16, 256, 16, 16}), Shape({256}), Shape({256}), Shape({256}), Shape({256})},
            {Shape({32, 512, 8, 8}), Shape({512}), Shape({512}), Shape({512}), Shape({512})},
        };
    }

    size_t get_required_shapes_count() const override { return 5; }

    void validate_shapes(const std::vector<Shape> &shapes) const override
    {
        if (shapes.size() != 5)
        {
            THROW_RUNTIME_ERROR("BatchNorm requires exactly 5 shapes, got {}", shapes.size());
        }
        const Shape &x_shape            = shapes[0];
        const Shape &gamma_shape        = shapes[1];
        const Shape &beta_shape         = shapes[2];
        const Shape &running_mean_shape = shapes[3];
        const Shape &running_var_shape  = shapes[4];

        if (x_shape.ndims() != 4)
        {
            THROW_RUNTIME_ERROR("BatchNorm x must be 4D (N, C, H, W), got {}", x_shape.to_string());
        }
        if (gamma_shape.ndims() != 1 || beta_shape.ndims() != 1 || running_mean_shape.ndims() != 1 ||
            running_var_shape.ndims() != 1)
        {
            THROW_RUNTIME_ERROR("BatchNorm gamma/beta/running_mean/running_var must be 1D (C,)");
        }

        int C = x_shape[1];
        if (gamma_shape[0] != C || beta_shape[0] != C || running_mean_shape[0] != C || running_var_shape[0] != C)
        {
            THROW_RUNTIME_ERROR(
                "BatchNorm channel dimension mismatch: x has C={}, but gamma/beta/running_mean/running_var have "
                "different sizes",
                C);
        }
    }

    std::string get_operator_name() const override { return "BatchNorm"; }
};

int main(int argc, char *argv[])
{
    BatchNormBenchmark benchmark;
    return benchmark.run(argc, argv);
}
