"""
Add算子性能测试
测试PyTorch的add操作性能
"""

import sys
import subprocess
from pathlib import Path

# 添加common目录到路径
common_dir = Path(__file__).parent.parent / 'common'
if str(common_dir) not in sys.path:
    sys.path.insert(0, str(common_dir))

from typing import List, Tuple, Dict, Optional
import torch
from benchmark_framework import BenchmarkFramework, BenchmarkConfig
from timer import Timer


class AddBenchmark(BenchmarkFramework):
    """Add算子基准测试类"""
    
    def get_required_shapes_count(self) -> int:
        """Add算子只需要一个shape"""
        return 1
    
    def run_benchmark(self, config: BenchmarkConfig) -> float:
        """执行PyTorch的add操作性能测试"""
        # Add算子只需要一个shape
        if len(config.shapes) != 1:
            raise ValueError(f"Add benchmark requires exactly 1 shape, got {len(config.shapes)}")
        
        shape = config.shapes[0]
        
        # 解析设备
        device = self._parse_device(config.device)
        dtype = torch.float32 if config.dtype == 'float32' else torch.float64
        
        # 创建输入张量
        x0 = torch.ones(shape, dtype=dtype, device=device)
        x1 = torch.full(shape, 2.0, dtype=dtype, device=device)
        
        # 预热
        if config.inplace:
            for _ in range(config.warmup_cnt):
                x0.add_(x1)
        else:
            for _ in range(config.warmup_cnt):
                result = x0 + x1
        # 预热结束后同步，确保预热完成
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 正式测试
        timer = Timer()
        timer.start()
        
        if config.inplace:
            for _ in range(config.repeat_cnt):
                x0.add_(x1)
        else:
            for _ in range(config.repeat_cnt):
                result = x0 + x1
        # 正式测试结束后同步，确保所有计算完成
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        total_time_us = timer.elapsed_us()
        return total_time_us / config.repeat_cnt
    
    def validate_shapes(self, shapes: List[Tuple[int, ...]]) -> None:
        """验证形状是否有效"""
        if len(shapes) != 1:
            raise ValueError(f"Add requires exactly 1 shape, got {len(shapes)}")


def benchmark_add_comparison(device_filter=None,
                             shape_filter=None,
                             warmup_cnt=5,
                             repeat_cnt=100,
                             inplace=False,
                             verbose=False):
    """
    运行add算子的PyTorch性能测试
    
    Args:
        device_filter: 设备过滤，'cpu'或'cuda'
        shape_filter: shape过滤，例如 '1000,1000'
        warmup_cnt: 预热次数
        repeat_cnt: 重复次数
        inplace: 是否使用就地操作
        verbose: 是否输出详细信息
    
    Returns:
        PyTorch测试结果列表，每个元素为:
        {'shape': str, 'device': str, 'dtype': str, 'time_us': float}
    """
    benchmark = AddBenchmark()
    pytorch_results = benchmark.run(
        device_filter=device_filter,
        shape_filter=shape_filter,
        warmup_cnt=warmup_cnt,
        repeat_cnt=repeat_cnt,
        inplace=inplace,
        verbose=verbose
    )
    
    return pytorch_results


if __name__ == "__main__":
    """命令行入口，与C++版本保持一致，可以直接运行此文件进行测试"""
    benchmark = AddBenchmark()
    sys.exit(benchmark.run_from_command_line())
