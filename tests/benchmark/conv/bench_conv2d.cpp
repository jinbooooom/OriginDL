#include <vector>
#include "benchmark/common/benchmark_framework.h"
#include "benchmark/common/timer.h"
#include "origin.h"
#include "origin/operators/conv/conv2d.h"

using namespace origin;
using namespace origin::benchmark;
namespace F = origin::functional;

/**
 * @brief Conv2d算子基准测试类
 */
class Conv2dBenchmark : public BenchmarkFramework
{
public:
    double run_benchmark(const BenchmarkConfig &config) override
    {
        // Conv2d算子需要两个shape：输入x和卷积核W
        if (config.shapes.size() != 2)
        {
            THROW_RUNTIME_ERROR("Conv2d benchmark requires exactly 2 shapes (x and W), got {}", config.shapes.size());
        }

        const Shape &x_shape = config.shapes[0];  // (N, C, H, W)
        const Shape &W_shape = config.shapes[1];  // (OC, C, KH, KW)

        // 验证形状维度
        if (x_shape.ndims() != 4)
        {
            THROW_RUNTIME_ERROR("Conv2d input x must be 4D (N, C, H, W), got shape {}", x_shape.to_string());
        }
        if (W_shape.ndims() != 4)
        {
            THROW_RUNTIME_ERROR("Conv2d weight W must be 4D (OC, C, KH, KW), got shape {}", W_shape.to_string());
        }

        // 验证通道数匹配
        if (x_shape[1] != W_shape[1])
        {
            THROW_RUNTIME_ERROR("Conv2d channel mismatch: x has {} channels, but W expects {} channels", x_shape[1],
                                W_shape[1]);
        }

        // 创建输入张量
        size_t numel_x = x_shape.elements();
        size_t numel_W = W_shape.elements();
        std::vector<float> data_x(numel_x, 1.0f);
        std::vector<float> data_W(numel_W, 2.0f);

        auto x = Tensor(data_x, x_shape, origin::dtype(config.dtype).device(config.device));
        auto W = Tensor(data_W, W_shape, origin::dtype(config.dtype).device(config.device));

        // 检查是否启用了就地操作
        if (config.inplace)
        {
            loge("Error: Conv2d operator does not support inplace operations in OriginDL");
            THROW_RUNTIME_ERROR("Conv2d operator does not support inplace operations in OriginDL");
        }

        // 使用默认的stride和pad
        int stride = 1;
        int pad    = 0;

        // 预热
        for (int i = 0; i < config.warmup_cnt; ++i)
        {
            auto result = F::conv2d(x, W, nullptr, stride, pad);
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
            auto result = F::conv2d(x, W, nullptr, stride, pad);
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
        // 对于conv2d，每个元素是包含两个Shape的vector：输入x和卷积核W
        return {
            // 小规模测试
            {Shape({1, 1, 3, 3}), Shape({1, 1, 3, 3})},     // 1x1通道，3x3图像，3x3卷积核
            {Shape({1, 3, 10, 10}), Shape({1, 3, 3, 3})},   // 1x3通道，10x10图像，3x3卷积核
            {Shape({1, 3, 32, 32}), Shape({16, 3, 3, 3})},  // 1x3通道，32x32图像，16输出通道，3x3卷积核
            // 中等规模测试
            {Shape({1, 64, 64, 64}), Shape({64, 64, 3, 3})},  // 1x64通道，64x64图像，64输出通道，3x3卷积核
            {Shape({4, 3, 224, 224}), Shape({64, 3, 7, 7})},  // 4x3通道，224x224图像，64输出通道，7x7卷积核
            // 大规模测试
            {Shape({8, 64, 224, 224}), Shape({128, 64, 3, 3})},  // 8x64通道，224x224图像，128输出通道，3x3卷积核
        };
    }

    size_t get_required_shapes_count() const override { return 2; }

    void validate_shapes(const std::vector<Shape> &shapes) const override
    {
        if (shapes.size() != 2)
        {
            THROW_RUNTIME_ERROR("Conv2d requires exactly 2 shapes (x and W), got {}", shapes.size());
        }

        const Shape &x_shape = shapes[0];
        const Shape &W_shape = shapes[1];

        // 验证输入x必须是4D
        if (x_shape.ndims() != 4)
        {
            THROW_RUNTIME_ERROR("Conv2d input x must be 4D (N, C, H, W), got shape with {} dimensions",
                                x_shape.ndims());
        }

        // 验证卷积核W必须是4D
        if (W_shape.ndims() != 4)
        {
            THROW_RUNTIME_ERROR("Conv2d weight W must be 4D (OC, C, KH, KW), got shape with {} dimensions",
                                W_shape.ndims());
        }

        // 验证通道数匹配
        if (x_shape[1] != W_shape[1])
        {
            THROW_RUNTIME_ERROR("Conv2d channel mismatch: x has {} channels, but W expects {} channels", x_shape[1],
                                W_shape[1]);
        }
    }

    std::string get_operator_name() const override { return "Conv2d"; }

    std::string get_additional_help() const override
    {
        return "                            Note: For conv2d, use two shapes separated by ':'\n"
               "                            e.g., \"1,3,224,224:64,3,7,7\" for input (1,3,224,224) and weight "
               "(64,3,7,7)\n"
               "                            Format: x_shape:W_shape where x is (N,C,H,W) and W is (OC,C,KH,KW)";
    }
};

int main(int argc, char *argv[])
{
    Conv2dBenchmark benchmark;
    return benchmark.run(argc, argv);
}
