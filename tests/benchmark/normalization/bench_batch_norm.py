"""
BatchNorm算子性能测试
测试PyTorch的batch_norm操作性能
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


class BatchNormBenchmark(BenchmarkFramework):
    """BatchNorm算子基准测试类"""
    
    def get_required_shapes_count(self) -> int:
        """BatchNorm算子需要5个shape：x, gamma, beta, running_mean, running_var"""
        return 5
    
    def run_benchmark(self, config: BenchmarkConfig) -> float:
        """执行PyTorch的batch_norm操作性能测试"""
        if len(config.shapes) != 5:
            raise ValueError(f"BatchNorm benchmark requires exactly 5 shapes, got {len(config.shapes)}")
        
        x_shape, gamma_shape, beta_shape, running_mean_shape, running_var_shape = config.shapes
        device = self._parse_device(config.device)
        dtype = torch.float32 if config.dtype == 'float32' else torch.float64
        
        # x: (N, C, H, W), gamma/beta/running_mean/running_var: (C,)
        x = torch.randn(x_shape, dtype=dtype, device=device)
        gamma = torch.ones(gamma_shape, dtype=dtype, device=device)
        beta = torch.zeros(beta_shape, dtype=dtype, device=device)
        running_mean = torch.zeros(running_mean_shape, dtype=dtype, device=device)
        running_var = torch.ones(running_var_shape, dtype=dtype, device=device)
        
        training = False  # 测试模式
        eps = 1e-5
        momentum = 0.1
        
        # 预热
        for _ in range(config.warmup_cnt):
            result = F.batch_norm(x, running_mean, running_var, gamma, beta, training=training, eps=eps, momentum=momentum)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 正式测试
        timer = Timer()
        timer.start()
        
        for _ in range(config.repeat_cnt):
            result = F.batch_norm(x, running_mean, running_var, gamma, beta, training=training, eps=eps, momentum=momentum)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        return timer.elapsed_us() / config.repeat_cnt
    
    def get_default_shapes(self):
        """获取默认测试形状"""
        return [
            [(1, 3, 32, 32), (3,), (3,), (3,), (3,)],  # x, gamma, beta, running_mean, running_var
            [(4, 64, 64, 64), (64,), (64,), (64,), (64,)],
            [(8, 128, 32, 32), (128,), (128,), (128,), (128,)],
            [(16, 256, 16, 16), (256,), (256,), (256,), (256,)],
            [(32, 512, 8, 8), (512,), (512,), (512,), (512,)],
        ]
    
    def validate_shapes(self, shapes: List[Tuple[int, ...]]) -> None:
        """验证形状是否有效"""
        if len(shapes) != 5:
            raise ValueError(f"BatchNorm requires exactly 5 shapes, got {len(shapes)}")
        x_shape, gamma_shape, beta_shape, running_mean_shape, running_var_shape = shapes
        
        if len(x_shape) != 4:
            raise ValueError(f"BatchNorm x must be 4D (N, C, H, W), got {x_shape}")
        if len(gamma_shape) != 1 or len(beta_shape) != 1 or len(running_mean_shape) != 1 or len(running_var_shape) != 1:
            raise ValueError(f"BatchNorm gamma/beta/running_mean/running_var must be 1D (C,), got {gamma_shape}, {beta_shape}, {running_mean_shape}, {running_var_shape}")
        
        C = x_shape[1]
        if gamma_shape[0] != C or beta_shape[0] != C or running_mean_shape[0] != C or running_var_shape[0] != C:
            raise ValueError(f"BatchNorm channel dimension mismatch: x has C={C}, but gamma/beta/running_mean/running_var have different sizes")


def benchmark_batch_norm_comparison(device_filter=None,
                             shape_filter=None,
                             warmup_cnt=5,
                             repeat_cnt=100,
                             inplace=False,
                             verbose=False):
    """
    运行batch_norm算子的PyTorch性能测试
    
    Args:
        device_filter: 设备过滤，'cpu'或'cuda'
        shape_filter: shape过滤，例如 '1,3,32,32:3:3:3:3' (x:gamma:beta:running_mean:running_var)
        warmup_cnt: 预热次数
        repeat_cnt: 重复次数
        inplace: 是否使用就地操作（batch_norm不支持inplace）
        verbose: 是否输出详细信息
    
    Returns:
        PyTorch测试结果列表
    """
    benchmark = BatchNormBenchmark()
    return benchmark.run(
        device_filter=device_filter,
        shape_filter=shape_filter,
        warmup_cnt=warmup_cnt,
        repeat_cnt=repeat_cnt,
        inplace=False,  # batch_norm不支持inplace
        verbose=verbose
    )


if __name__ == "__main__":
    benchmark = BatchNormBenchmark()
    sys.exit(benchmark.run_from_command_line())
