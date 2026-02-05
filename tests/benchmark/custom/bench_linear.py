"""
Linear算子性能测试
测试PyTorch的linear操作性能
"""

import sys
from pathlib import Path

# 添加common目录到路径
common_dir = Path(__file__).parent.parent / 'common'
if str(common_dir) not in sys.path:
    sys.path.insert(0, str(common_dir))

from typing import List, Tuple
import torch
import torch.nn.functional as F
from benchmark_framework import BenchmarkFramework, BenchmarkConfig
from timer import Timer


class LinearBenchmark(BenchmarkFramework):
    """Linear算子基准测试类"""
    
    def get_required_shapes_count(self) -> int:
        """Linear算子需要3个shape：x, weight, bias"""
        return 3
    
    def run_benchmark(self, config: BenchmarkConfig) -> float:
        """执行PyTorch的linear操作性能测试"""
        if len(config.shapes) != 3:
            raise ValueError(f"Linear benchmark requires exactly 3 shapes, got {len(config.shapes)}")
        
        x_shape, weight_shape, bias_shape = config.shapes
        device = self._parse_device(config.device)
        dtype = torch.float32 if config.dtype == 'float32' else torch.float64
        
        # x: (N, in_features), weight: (out_features, in_features), bias: (out_features,)
        x = torch.randn(x_shape, dtype=dtype, device=device)
        weight = torch.randn(weight_shape, dtype=dtype, device=device)
        bias = torch.randn(bias_shape, dtype=dtype, device=device)
        
        # 预热
        for _ in range(config.warmup_cnt):
            result = F.linear(x, weight, bias)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 正式测试
        timer = Timer()
        timer.start()
        
        for _ in range(config.repeat_cnt):
            result = F.linear(x, weight, bias)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        return timer.elapsed_us() / config.repeat_cnt
    
    def get_default_shapes(self):
        """获取默认测试形状"""
        return [
            [(32, 128), (256, 128), (256,)],  # x, weight, bias
            [(64, 256), (512, 256), (512,)],
            [(128, 512), (1024, 512), (1024,)],
            [(256, 1024), (2048, 1024), (2048,)],
            [(512, 2048), (4096, 2048), (4096,)],
        ]
    
    def validate_shapes(self, shapes: List[Tuple[int, ...]]) -> None:
        """验证形状是否有效"""
        if len(shapes) != 3:
            raise ValueError(f"Linear requires exactly 3 shapes, got {len(shapes)}")
        x_shape, weight_shape, bias_shape = shapes
        
        if len(x_shape) != 2:
            raise ValueError(f"Linear x must be 2D (N, in_features), got {x_shape}")
        if len(weight_shape) != 2:
            raise ValueError(f"Linear weight must be 2D (out_features, in_features), got {weight_shape}")
        if len(bias_shape) != 1:
            raise ValueError(f"Linear bias must be 1D (out_features,), got {bias_shape}")
        
        in_features = x_shape[1]
        out_features = weight_shape[0]
        if weight_shape[1] != in_features:
            raise ValueError(f"Linear weight shape mismatch: weight has in_features={weight_shape[1]}, but x has in_features={in_features}")
        if bias_shape[0] != out_features:
            raise ValueError(f"Linear bias shape mismatch: bias has out_features={bias_shape[0]}, but weight has out_features={out_features}")


def benchmark_linear_comparison(device_filter=None,
                             shape_filter=None,
                             warmup_cnt=5,
                             repeat_cnt=100,
                             inplace=False,
                             verbose=False):
    """
    运行linear算子的PyTorch性能测试
    
    Args:
        device_filter: 设备过滤，'cpu'或'cuda'
        shape_filter: shape过滤，例如 '32,128:256,128:256' (x:weight:bias)
        warmup_cnt: 预热次数
        repeat_cnt: 重复次数
        inplace: 是否使用就地操作（linear不支持inplace）
        verbose: 是否输出详细信息
    
    Returns:
        PyTorch测试结果列表
    """
    benchmark = LinearBenchmark()
    return benchmark.run(
        device_filter=device_filter,
        shape_filter=shape_filter,
        warmup_cnt=warmup_cnt,
        repeat_cnt=repeat_cnt,
        inplace=False,  # linear不支持inplace
        verbose=verbose
    )


if __name__ == "__main__":
    benchmark = LinearBenchmark()
    sys.exit(benchmark.run_from_command_line())
