#include <vector>
#include "origin.h"
#include "origin/operators/math/mat_mul.h"
#include "benchmark/common/benchmark_framework.h"
#include "benchmark/common/timer.h"

using namespace origin;
using namespace origin::benchmark;
namespace F = origin::functional;

/**
 * @brief MatMul算子基准测试类
 */
class MatMulBenchmark : public BenchmarkFramework {
public:
    double run_benchmark(const BenchmarkConfig& config) override
    {
        // MatMul算子需要两个shape
        if (config.shapes.size() != 2)
        {
            THROW_RUNTIME_ERROR("MatMul benchmark requires exactly 2 shapes, got {}", config.shapes.size());
        }
        
        const Shape& shape_a = config.shapes[0];
        const Shape& shape_b = config.shapes[1];
        
        // 创建输入张量
        size_t numel_a = shape_a.elements();
        size_t numel_b = shape_b.elements();
        std::vector<float> data_a(numel_a, 1.0f);
        std::vector<float> data_b(numel_b, 2.0f);
        
        auto x0 = Tensor(data_a, shape_a, origin::dtype(config.dtype).device(config.device));
        auto x1 = Tensor(data_b, shape_b, origin::dtype(config.dtype).device(config.device));
        
        // 预热
        for (int i = 0; i < config.warmup_cnt; ++i)
        {
            auto result = F::mat_mul(x0, x1);
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
            auto result = F::mat_mul(x0, x1);
            // 确保计算完成
            if (config.device.type() == DeviceType::kCUDA)
            {
                cuda::synchronize();
            }
        }
        
        double total_time_us = timer.elapsed_us();
        return total_time_us / config.repeat_cnt;
    }

    std::vector<std::vector<Shape>> get_default_shapes() const override
    {
        // 对于matmul，每个元素是包含两个Shape的vector
        return {
            {Shape({1, 1}), Shape({1, 1})},              // {1, 1} x {1, 1} -> {1, 1}
            {Shape({10, 10}), Shape({10, 10})},          // {10, 10} x {10, 10} -> {10, 10}
            {Shape({100, 100}), Shape({100, 100})},      // {100, 100} x {100, 100} -> {100, 100}
            {Shape({1000, 1000}), Shape({1000, 1000})},  // {1000, 1000} x {1000, 1000} -> {1000, 1000}
            {Shape({10000, 10000}), Shape({10000, 10000})},  // {10000, 10000} x {10000, 10000} -> {10000, 10000}
        };
    }

    size_t get_required_shapes_count() const override
    {
        return 2;
    }

    void validate_shapes(const std::vector<Shape>& shapes) const override
    {
        if (shapes.size() != 2)
        {
            THROW_RUNTIME_ERROR("MatMul requires exactly 2 shapes, got {}", shapes.size());
        }
        
        const Shape& shape_a = shapes[0];
        const Shape& shape_b = shapes[1];
        
        // 验证至少是2维
        if (shape_a.ndims() < 2 || shape_b.ndims() < 2)
        {
            THROW_RUNTIME_ERROR("MatMul requires at least 2D shapes, got shapes with {} and {} dimensions",
                               shape_a.ndims(), shape_b.ndims());
        }
        
        // 验证矩阵乘法维度兼容性
        if (shape_a.ndims() == 2 && shape_b.ndims() == 2)
        {
            if (shape_a[1] != shape_b[0])
            {
                THROW_RUNTIME_ERROR("MatMul dimension mismatch: {} x {} (shape_a[1]={} != shape_b[0]={})",
                                   shape_a.to_string(), shape_b.to_string(), shape_a[1], shape_b[0]);
            }
        }
        else if (shape_a.ndims() == 3 && shape_b.ndims() == 2)
        {
            // 批量矩阵乘法：{batch, m, k} x {k, n} -> {batch, m, n}
            if (shape_a[2] != shape_b[0])
            {
                THROW_RUNTIME_ERROR("MatMul dimension mismatch: {} x {} (shape_a[2]={} != shape_b[0]={})",
                                   shape_a.to_string(), shape_b.to_string(), shape_a[2], shape_b[0]);
            }
        }
        else if (shape_a.ndims() == 3 && shape_b.ndims() == 3)
        {
            // 批量矩阵乘法：{batch, m, k} x {batch, k, n} -> {batch, m, n}
            if (shape_a[0] != shape_b[0] || shape_a[2] != shape_b[1])
            {
                THROW_RUNTIME_ERROR("MatMul dimension mismatch: {} x {} (batch or k dimension mismatch)",
                                   shape_a.to_string(), shape_b.to_string());
            }
        }
    }

    std::string get_operator_name() const override
    {
        return "MatMul";
    }

    std::string get_additional_help() const override
    {
        return "                            Note: For matmul, use two shapes separated by ':'\n"
               "                            e.g., \"100,200:200,50\" for {100,200} x {200,50} -> {100,50}";
    }
};

int main(int argc, char* argv[])
{
    MatMulBenchmark benchmark;
    return benchmark.run(argc, argv);
}
