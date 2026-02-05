"""
AdaptiveAvgPool2d算子性能测试
测试PyTorch的adaptive_avg_pool2d操作性能
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


class AdaptiveAvgPool2dBenchmark(BenchmarkFramework):
    """AdaptiveAvgPool2d算子基准测试类"""
    
    def get_required_shapes_count(self) -> int:
        """AdaptiveAvgPool2d算子只需要一个shape（4D输入）"""
        return 1
    
    def run_benchmark(self, config: BenchmarkConfig) -> float:
        """执行PyTorch的adaptive_avg_pool2d操作性能测试"""
        if len(config.shapes) != 1:
            raise ValueError(f"AdaptiveAvgPool2d benchmark requires exactly 1 shape, got {len(config.shapes)}")
        
        shape = config.shapes[0]
        device = self._parse_device(config.device)
        dtype = torch.float32 if config.dtype == 'float32' else torch.float64
        
        # AdaptiveAvgPool2d需要4D输入 (N, C, H, W)
        if len(shape) != 4:
            raise ValueError(f"AdaptiveAvgPool2d requires 4D shape (N, C, H, W), got {shape}")
        
        x = torch.ones(shape, dtype=dtype, device=device)
        
        # 使用默认的输出大小 (1, 1)
        output_size = (1, 1)
        
        # 预热
        for _ in range(config.warmup_cnt):
            result = F.adaptive_avg_pool2d(x, output_size)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 正式测试
        timer = Timer()
        timer.start()
        
        for _ in range(config.repeat_cnt):
            result = F.adaptive_avg_pool2d(x, output_size)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        return timer.elapsed_us() / config.repeat_cnt
    
    def get_default_shapes(self):
        """获取默认测试形状"""
        return [
            [(1, 1, 3, 3)],
            [(1, 3, 10, 10)],
            [(1, 3, 32, 32)],
            [(1, 64, 64, 64)],
            [(4, 3, 224, 224)],
        ]
    
    def validate_shapes(self, shapes: List[Tuple[int, ...]]) -> None:
        """验证形状是否有效"""
        if len(shapes) != 1:
            raise ValueError(f"AdaptiveAvgPool2d requires exactly 1 shape, got {len(shapes)}")
        if len(shapes[0]) != 4:
            raise ValueError(f"AdaptiveAvgPool2d requires 4D shape (N, C, H, W), got {shapes[0]}")


def benchmark_adaptive_avg_pool2d_comparison(device_filter=None,
                             shape_filter=None,
                             warmup_cnt=5,
                             repeat_cnt=100,
                             inplace=False,
                             verbose=False):
    """
    运行adaptive_avg_pool2d算子的PyTorch性能测试
    
    Args:
        device_filter: 设备过滤，'cpu'或'cuda'
        shape_filter: shape过滤，例如 '1,3,224,224'
        warmup_cnt: 预热次数
        repeat_cnt: 重复次数
        inplace: 是否使用就地操作（adaptive_avg_pool2d不支持inplace）
        verbose: 是否输出详细信息
    
    Returns:
        PyTorch测试结果列表
    """
    benchmark = AdaptiveAvgPool2dBenchmark()
    return benchmark.run(
        device_filter=device_filter,
        shape_filter=shape_filter,
        warmup_cnt=warmup_cnt,
        repeat_cnt=repeat_cnt,
        inplace=False,  # adaptive_avg_pool2d不支持inplace
        verbose=verbose
    )


if __name__ == "__main__":
    benchmark = AdaptiveAvgPool2dBenchmark()
    sys.exit(benchmark.run_from_command_line())
