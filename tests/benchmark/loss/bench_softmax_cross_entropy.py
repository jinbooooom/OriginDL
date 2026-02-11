"""
SoftmaxCrossEntropy算子性能测试
测试PyTorch的softmax_cross_entropy操作性能
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


class SoftmaxCrossEntropyBenchmark(BenchmarkFramework):
    """SoftmaxCrossEntropy算子基准测试类"""
    
    def get_required_shapes_count(self) -> int:
        """SoftmaxCrossEntropy算子需要2个shape：x和target"""
        return 2
    
    def run_benchmark(self, config: BenchmarkConfig) -> float:
        """执行PyTorch的softmax_cross_entropy操作性能测试"""
        if len(config.shapes) != 2:
            raise ValueError(f"SoftmaxCrossEntropy benchmark requires exactly 2 shapes, got {len(config.shapes)}")
        
        x_shape, target_shape = config.shapes[0], config.shapes[1]
        device = self._parse_device(config.device)
        dtype = torch.float32 if config.dtype == 'float32' else torch.float64
        
        # x: (N, C), target: (N,)
        x = torch.randn(x_shape, dtype=dtype, device=device)
        target = torch.randint(0, x_shape[1], target_shape, dtype=torch.long, device=device)
        
        # 预热
        for _ in range(config.warmup_cnt):
            loss = F.cross_entropy(x, target)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 正式测试
        timer = Timer()
        timer.start()
        
        for _ in range(config.repeat_cnt):
            loss = F.cross_entropy(x, target)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        return timer.elapsed_us() / config.repeat_cnt
    
    def get_default_shapes(self):
        """获取默认测试形状"""
        return [
            [(32, 10), (32,)],  # (N, C), (N,)
            [(64, 100), (64,)],
            [(128, 1000), (128,)],
            [(256, 10), (256,)],
            [(512, 100), (512,)],
        ]
    
    def validate_shapes(self, shapes: List[Tuple[int, ...]]) -> None:
        """验证形状是否有效"""
        if len(shapes) != 2:
            raise ValueError(f"SoftmaxCrossEntropy requires exactly 2 shapes, got {len(shapes)}")
        if len(shapes[0]) != 2:
            raise ValueError(f"SoftmaxCrossEntropy x must be 2D (N, C), got {shapes[0]}")
        if len(shapes[1]) != 1:
            raise ValueError(f"SoftmaxCrossEntropy target must be 1D (N,), got {shapes[1]}")
        if shapes[0][0] != shapes[1][0]:
            raise ValueError(f"SoftmaxCrossEntropy x and target must have same batch size, got {shapes[0][0]} and {shapes[1][0]}")


def benchmark_softmax_cross_entropy_comparison(device_filter=None,
                             shape_filter=None,
                             warmup_cnt=5,
                             repeat_cnt=100,
                             inplace=False,
                             verbose=False):
    """
    运行softmax_cross_entropy算子的PyTorch性能测试
    
    Args:
        device_filter: 设备过滤，'cpu'或'cuda'
        shape_filter: shape过滤，例如 '32,10:32' (x_shape:target_shape)
        warmup_cnt: 预热次数
        repeat_cnt: 重复次数
        inplace: 是否使用就地操作（softmax_cross_entropy不支持inplace）
        verbose: 是否输出详细信息
    
    Returns:
        PyTorch测试结果列表
    """
    benchmark = SoftmaxCrossEntropyBenchmark()
    return benchmark.run(
        device_filter=device_filter,
        shape_filter=shape_filter,
        warmup_cnt=warmup_cnt,
        repeat_cnt=repeat_cnt,
        inplace=False,  # softmax_cross_entropy不支持inplace
        verbose=verbose
    )


if __name__ == "__main__":
    benchmark = SoftmaxCrossEntropyBenchmark()
    sys.exit(benchmark.run_from_command_line())
