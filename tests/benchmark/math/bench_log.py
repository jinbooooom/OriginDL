"""
Log算子性能测试
测试PyTorch的log操作性能
"""

import sys
from pathlib import Path

# 添加common目录到路径
common_dir = Path(__file__).parent.parent / 'common'
if str(common_dir) not in sys.path:
    sys.path.insert(0, str(common_dir))

from typing import List, Tuple
import torch
from benchmark_framework import BenchmarkFramework, BenchmarkConfig
from timer import Timer


class LogBenchmark(BenchmarkFramework):
    """Log算子基准测试类"""
    
    def get_required_shapes_count(self) -> int:
        """Log算子只需要一个shape"""
        return 1
    
    def run_benchmark(self, config: BenchmarkConfig) -> float:
        """执行PyTorch的log操作性能测试"""
        if len(config.shapes) != 1:
            raise ValueError(f"Log benchmark requires exactly 1 shape, got {len(config.shapes)}")
        
        shape = config.shapes[0]
        device = self._parse_device(config.device)
        dtype = torch.float32 if config.dtype == 'float32' else torch.float64
        
        # log需要正数，使用ones确保输入为正
        x = torch.ones(shape, dtype=dtype, device=device) * 2.0
        
        # 预热
        if config.inplace:
            for _ in range(config.warmup_cnt):
                x.log_()
        else:
            for _ in range(config.warmup_cnt):
                result = torch.log(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 正式测试
        timer = Timer()
        timer.start()
        
        if config.inplace:
            for _ in range(config.repeat_cnt):
                x.log_()
        else:
            for _ in range(config.repeat_cnt):
                result = torch.log(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        return timer.elapsed_us() / config.repeat_cnt
    
    def validate_shapes(self, shapes: List[Tuple[int, ...]]) -> None:
        """验证形状是否有效"""
        if len(shapes) != 1:
            raise ValueError(f"Log requires exactly 1 shape, got {len(shapes)}")


def benchmark_log_comparison(device_filter=None,
                             shape_filter=None,
                             warmup_cnt=5,
                             repeat_cnt=100,
                             inplace=False,
                             verbose=False):
    """
    运行log算子的PyTorch性能测试
    
    Args:
        device_filter: 设备过滤，'cpu'或'cuda'
        shape_filter: shape过滤，例如 '1000,1000'
        warmup_cnt: 预热次数
        repeat_cnt: 重复次数
        inplace: 是否使用就地操作
        verbose: 是否输出详细信息
    
    Returns:
        PyTorch测试结果列表
    """
    benchmark = LogBenchmark()
    return benchmark.run(
        device_filter=device_filter,
        shape_filter=shape_filter,
        warmup_cnt=warmup_cnt,
        repeat_cnt=repeat_cnt,
        inplace=inplace,
        verbose=verbose
    )


if __name__ == "__main__":
    benchmark = LogBenchmark()
    sys.exit(benchmark.run_from_command_line())
