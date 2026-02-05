"""
Conv2d算子性能测试
测试PyTorch的conv2d操作性能
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


class Conv2dBenchmark(BenchmarkFramework):
    """Conv2d算子基准测试类"""
    
    def get_required_shapes_count(self) -> int:
        """Conv2d算子需要两个shape：输入x和卷积核W"""
        return 2
    
    def run_benchmark(self, config: BenchmarkConfig) -> float:
        """执行PyTorch的conv2d操作性能测试"""
        # Conv2d算子需要两个shape
        if len(config.shapes) != 2:
            raise ValueError(f"Conv2d benchmark requires exactly 2 shapes (x and W), got {len(config.shapes)}")
        
        x_shape = config.shapes[0]  # (N, C, H, W)
        W_shape = config.shapes[1]  # (OC, C, KH, KW)
        
        # 验证形状维度
        if len(x_shape) != 4:
            raise ValueError(f"Conv2d input x must be 4D (N, C, H, W), got shape {x_shape}")
        if len(W_shape) != 4:
            raise ValueError(f"Conv2d weight W must be 4D (OC, C, KH, KW), got shape {W_shape}")
        
        # 验证通道数匹配
        if x_shape[1] != W_shape[1]:
            raise ValueError(
                f"Conv2d channel mismatch: x has {x_shape[1]} channels, but W expects {W_shape[1]} channels"
            )
        
        # 解析设备
        device = self._parse_device(config.device)
        dtype = torch.float32 if config.dtype == 'float32' else torch.float64
        
        # 创建输入张量
        x = torch.ones(x_shape, dtype=dtype, device=device)
        W = torch.full(W_shape, 2.0, dtype=dtype, device=device)
        
        # 使用默认的stride和pad
        stride = 1
        pad = 0
        
        # 对于就地操作，需要预先创建输出张量
        out = None
        if config.inplace:
            # 计算输出形状
            # OH = (H + 2*pad - KH) // stride + 1
            # OW = (W + 2*pad - KW) // stride + 1
            H, W_in = x_shape[2], x_shape[3]
            KH, KW = W_shape[2], W_shape[3]
            OH = (H + 2 * pad - KH) // stride + 1
            OW = (W_in + 2 * pad - KW) // stride + 1
            OC = W_shape[0]
            output_shape = (x_shape[0], OC, OH, OW)
            out = torch.empty(output_shape, dtype=dtype, device=device)
        
        # 预热
        if config.inplace:
            for _ in range(config.warmup_cnt):
                F.conv2d(x, W, None, stride, pad, out=out)
        else:
            for _ in range(config.warmup_cnt):
                result = F.conv2d(x, W, None, stride, pad)
        # 预热结束后同步，确保预热完成
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 正式测试
        timer = Timer()
        timer.start()
        
        if config.inplace:
            # 对于就地操作，使用 out 参数
            for _ in range(config.repeat_cnt):
                F.conv2d(x, W, None, stride, pad, out=out)
        else:
            for _ in range(config.repeat_cnt):
                result = F.conv2d(x, W, None, stride, pad)
        # 正式测试结束后同步，确保所有计算完成
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        total_time_us = timer.elapsed_us()
        return total_time_us / config.repeat_cnt
    
    def get_default_shapes(self):
        """获取默认测试形状"""
        # 对于conv2d，每个元素是包含两个shape的列表：输入x和卷积核W
        return [
            # 小规模测试
            [(1, 1, 3, 3), (1, 1, 3, 3)],              # 1x1通道，3x3图像，3x3卷积核
            [(1, 3, 10, 10), (1, 3, 3, 3)],            # 1x3通道，10x10图像，3x3卷积核
            [(1, 3, 32, 32), (16, 3, 3, 3)],           # 1x3通道，32x32图像，16输出通道，3x3卷积核
            # 中等规模测试
            [(1, 64, 64, 64), (64, 64, 3, 3)],          # 1x64通道，64x64图像，64输出通道，3x3卷积核
            [(4, 3, 224, 224), (64, 3, 7, 7)],         # 4x3通道，224x224图像，64输出通道，7x7卷积核
            # 大规模测试
            [(8, 64, 224, 224), (128, 64, 3, 3)],       # 8x64通道，224x224图像，128输出通道，3x3卷积核
        ]
    
    def validate_shapes(self, shapes: List[Tuple[int, ...]]) -> None:
        """验证形状是否有效"""
        if len(shapes) != 2:
            raise ValueError(f"Conv2d requires exactly 2 shapes (x and W), got {len(shapes)}")
        
        x_shape = shapes[0]
        W_shape = shapes[1]
        
        # 验证输入x必须是4D
        if len(x_shape) != 4:
            raise ValueError(
                f"Conv2d input x must be 4D (N, C, H, W), got shape with {len(x_shape)} dimensions"
            )
        
        # 验证卷积核W必须是4D
        if len(W_shape) != 4:
            raise ValueError(
                f"Conv2d weight W must be 4D (OC, C, KH, KW), got shape with {len(W_shape)} dimensions"
            )
        
        # 验证通道数匹配
        if x_shape[1] != W_shape[1]:
            raise ValueError(
                f"Conv2d channel mismatch: x has {x_shape[1]} channels, but W expects {W_shape[1]} channels"
            )


def benchmark_conv2d_comparison(device_filter=None,
                                shape_filter=None,
                                warmup_cnt=5,
                                repeat_cnt=100,
                                inplace=False,
                                verbose=False):
    """
    运行conv2d算子的PyTorch性能测试
    
    Args:
        device_filter: 设备过滤，'cpu'或'cuda'
        shape_filter: shape过滤，例如 '1,3,224,224:64,3,7,7'
        warmup_cnt: 预热次数
        repeat_cnt: 重复次数
        inplace: 是否使用就地操作
        verbose: 是否输出详细信息
    
    Returns:
        PyTorch测试结果列表，每个元素为:
        {'shape': str, 'device': str, 'dtype': str, 'time_us': float}
    """
    benchmark = Conv2dBenchmark()
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
    benchmark = Conv2dBenchmark()
    sys.exit(benchmark.run_from_command_line())
