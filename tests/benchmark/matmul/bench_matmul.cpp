#include <vector>
#include "benchmark/common/benchmark_framework.h"
#include "benchmark/common/timer.h"
#include "origin.h"
#include "origin/operators/math/mat_mul.h"

using namespace origin;
using namespace origin::benchmark;
namespace F = origin::functional;

/**
 * @brief MatMul算子基准测试类
 */
class MatMulBenchmark : public BenchmarkFramework
{
public:
    double run_benchmark(const BenchmarkConfig &config) override
    {
        // MatMul算子需要两个shape
        if (config.shapes.size() != 2)
        {
            THROW_RUNTIME_ERROR("MatMul benchmark requires exactly 2 shapes, got {}", config.shapes.size());
        }

        const Shape &shape_a = config.shapes[0];
        const Shape &shape_b = config.shapes[1];

        // 创建输入张量
        size_t numel_a = shape_a.elements();
        size_t numel_b = shape_b.elements();
        std::vector<float> data_a(numel_a, 1.0f);
        std::vector<float> data_b(numel_b, 2.0f);

        auto x0 = Tensor(data_a, shape_a, origin::dtype(config.dtype).device(config.device));
        auto x1 = Tensor(data_b, shape_b, origin::dtype(config.dtype).device(config.device));

        // 检查是否启用了就地操作
        if (config.inplace)
        {
            loge("Error: MatMul operator does not support inplace operations in OriginDL");
            THROW_RUNTIME_ERROR("MatMul operator does not support inplace operations in OriginDL");
        }

        // 预热
        for (int i = 0; i < config.warmup_cnt; ++i)
        {
            auto result = F::mat_mul(x0, x1);
        }
        // 预热结束后同步，确保预热完成
        if (config.device.type() == DeviceType::kCUDA)
        {
            cuda::synchronize();
        }

        // 正式测试
        Timer timer;
        timer.start();

        for (int i = 0; i < config.repeat_cnt; ++i)
        {
            auto result = F::mat_mul(x0, x1);
        }
        // 正式测试结束后同步，确保所有计算完成
        if (config.device.type() == DeviceType::kCUDA)
        {
            cuda::synchronize();
        }

        double total_time_us = timer.elapsed_us();
        return total_time_us / config.repeat_cnt;
    }

    std::vector<std::vector<Shape>> get_default_shapes() const override
    {
        // 对于matmul，每个元素是包含两个Shape的vector
        return {
            {Shape({1, 1}), Shape({1, 1})},                  // {1, 1} x {1, 1} -> {1, 1}
            {Shape({10, 10}), Shape({10, 10})},              // {10, 10} x {10, 10} -> {10, 10}
            {Shape({100, 100}), Shape({100, 100})},          // {100, 100} x {100, 100} -> {100, 100}
            {Shape({1000, 1000}), Shape({1000, 1000})},      // {1000, 1000} x {1000, 1000} -> {1000, 1000}
            {Shape({10000, 10000}), Shape({10000, 10000})},  // {10000, 10000} x {10000, 10000} -> {10000, 10000}
        };
    }

    size_t get_required_shapes_count() const override { return 2; }

    void validate_shapes(const std::vector<Shape> &shapes) const override
    {
        if (shapes.size() != 2)
        {
            THROW_RUNTIME_ERROR("MatMul requires exactly 2 shapes, got {}", shapes.size());
        }

        const Shape &shape_a = shapes[0];
        const Shape &shape_b = shapes[1];

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

    std::string get_operator_name() const override { return "MatMul"; }

    std::string get_additional_help() const override
    {
        return "                            Note: For matmul, use two shapes separated by ':'\n"
               "                            e.g., \"100,200:200,50\" for {100,200} x {200,50} -> {100,50}";
    }
};

int main(int argc, char *argv[])
{
    MatMulBenchmark benchmark;
    return benchmark.run(argc, argv);
}

/*
./build/bin/benchmark/bench_matmul -d cuda -r 20

A100，ORIGIN_KERNEL_ALGO=0
shape                                   repeat          device          dtype           origindl_time_us    
{1, 1}:{1, 1}                           20              cuda:0          float32         7.8000              
{10, 10}:{10, 10}                       20              cuda:0          float32         7.8500              
{100, 100}:{100, 100}                   20              cuda:0          float32         8.0000              
{1000, 1000}:{1000, 1000}               20              cuda:0          float32         638.5500            
{10000, 10000}:{10000, 10000}           20              cuda:0          float32         1559682.3000 

A100，ORIGIN_KERNEL_ALGO=1
shape                                   repeat          device          dtype           origindl_time_us    
{1, 1}:{1, 1}                           20              cuda:0          float32         7.6500              
{10, 10}:{10, 10}                       20              cuda:0          float32         7.7000              
{100, 100}:{100, 100}                   20              cuda:0          float32         7.9000              
{1000, 1000}:{1000, 1000}               20              cuda:0          float32         604.7500            
{10000, 10000}:{10000, 10000}           20              cuda:0          float32         983119.1000 

A100，ORIGIN_KERNEL_ALGO=2
shape                                   repeat          device          dtype           origindl_time_us    
{1, 1}:{1, 1}                           20              cuda:0          float32         7.7500              
{10, 10}:{10, 10}                       20              cuda:0          float32         7.6000              
{100, 100}:{100, 100}                   20              cuda:0          float32         7.7000              
{1000, 1000}:{1000, 1000}               20              cuda:0          float32         618.1500            
{10000, 10000}:{10000, 10000}           20              cuda:0          float32         981852.6500

A100，ORIGIN_KERNEL_ALGO=3
shape                                   repeat          device          dtype           origindl_time_us    
{1, 1}:{1, 1}                           20              cuda:0          float32         8.2000              
{10, 10}:{10, 10}                       20              cuda:0          float32         8.0000              
{100, 100}:{100, 100}                   20              cuda:0          float32         14.2000             
{1000, 1000}:{1000, 1000}               20              cuda:0          float32         1206.1500           
{10000, 10000}:{10000, 10000}           20              cuda:0          float32         1304046.5500

A100，ORIGIN_KERNEL_ALGO=4

A100，ORIGIN_KERNEL_ALGO=5
shape                                   repeat          device          dtype           origindl_time_us    
{1, 1}:{1, 1}                           20              cuda:0          float32         7.6000              
{10, 10}:{10, 10}                       20              cuda:0          float32         7.5000              
{100, 100}:{100, 100}                   20              cuda:0          float32         20.8000             
{1000, 1000}:{1000, 1000}               20              cuda:0          float32         1229.9500           
{10000, 10000}:{10000, 10000}           20              cuda:0          float32         1368403.3000 

A100，ORIGIN_KERNEL_ALGO=6
shape                                   repeat          device          dtype           origindl_time_us    
{1, 1}:{1, 1}                           20              cuda:0          float32         7.4000              
{10, 10}:{10, 10}                       20              cuda:0          float32         7.4500              
{100, 100}:{100, 100}                   20              cuda:0          float32         20.6000             
{1000, 1000}:{1000, 1000}               20              cuda:0          float32         1233.5000           
{10000, 10000}:{10000, 10000}           20              cuda:0          float32         1353519.8500 

A100，ORIGIN_KERNEL_ALGO=7
shape                                   repeat          device          dtype           origindl_time_us    
{1, 1}:{1, 1}                           20              cuda:0          float32         9.1500              
{10, 10}:{10, 10}                       20              cuda:0          float32         10.5000             
{100, 100}:{100, 100}                   20              cuda:0          float32         61.0000             
{1000, 1000}:{1000, 1000}               20              cuda:0          float32         480.0000            
{10000, 10000}:{10000, 10000}           20              cuda:0          float32         174940.3500

A100，ORIGIN_KERNEL_ALGO=8
shape                                   repeat          device          dtype           origindl_time_us    
{1, 1}:{1, 1}                           20              cuda:0          float32         7.9000              
{10, 10}:{10, 10}                       20              cuda:0          float32         7.8000              
{100, 100}:{100, 100}                   20              cuda:0          float32         11.4500             
{1000, 1000}:{1000, 1000}               20              cuda:0          float32         296.3500            
{10000, 10000}:{10000, 10000}           20              cuda:0          float32         402449.2500

A100，ORIGIN_KERNEL_ALGO=9
shape                                   repeat          device          dtype           origindl_time_us    
{1, 1}:{1, 1}                           20              cuda:0          float32         8.4000              
{10, 10}:{10, 10}                       20              cuda:0          float32         8.0000              
{100, 100}:{100, 100}                   20              cuda:0          float32         21.4500             
{1000, 1000}:{1000, 1000}               20              cuda:0          float32         128.1500            
{10000, 10000}:{10000, 10000}           20              cuda:0          float32         181679.7000 

A100，ORIGIN_KERNEL_ALGO=6666
shape                                   repeat          device          dtype           origindl_time_us    
{1, 1}:{1, 1}                           20              cuda:0          float32         7.9500              
{10, 10}:{10, 10}                       20              cuda:0          float32         8.2000              
{100, 100}:{100, 100}                   20              cuda:0          float32         8.2500              
{1000, 1000}:{1000, 1000}               20              cuda:0          float32         125.6500            
{10000, 10000}:{10000, 10000}           20              cuda:0          float32         179499.2500


export ORIGIN_KERNEL_ALGO=6666
python3 run_benchmark.py -f matmul -d cuda -r 20

=================================================================================================
Matmul Operator Performance Comparison
=================================================================================================
Shape                         Repeat   Device   Dtype     OriginDL(us)    PyTorch(us)     Speedup
-------------------------------------------------------------------------------------------------
{1,1}:{1,1}                   20       cuda:0   float32   8.2000          23.4699         2.8622 
{10,10}:{10,10}               20       cuda:0   float32   8.2000          22.2084         2.7083 
{100,100}:{100,100}           20       cuda:0   float32   8.4000          21.8670         2.6032 
{1000,1000}:{1000,1000}       20       cuda:0   float32   125.3500        142.9909        1.1407 
{10000,10000}:{10000,10000}   20       cuda:0   float32   189848.7500     181272.7210     0.9548 
*/
