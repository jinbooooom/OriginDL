"""
MatMul算子性能测试
测试PyTorch的matmul操作性能
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


class MatMulBenchmark(BenchmarkFramework):
    """MatMul算子基准测试类（注意：算子名称为 mat_mul）"""
    
    def get_required_shapes_count(self) -> int:
        """MatMul算子需要两个shape"""
        return 2
    
    def run_benchmark(self, config: BenchmarkConfig) -> float:
        """执行PyTorch的matmul操作性能测试"""
        # MatMul算子需要两个shape
        if len(config.shapes) != 2:
            raise ValueError(f"MatMul benchmark requires exactly 2 shapes, got {len(config.shapes)}")
        
        shape_a = config.shapes[0]
        shape_b = config.shapes[1]
        
        # 解析设备
        device = self._parse_device(config.device)
        dtype = torch.float32 if config.dtype == 'float32' else torch.float64
        
        # 创建输入张量
        x0 = torch.ones(shape_a, dtype=dtype, device=device)
        x1 = torch.full(shape_b, 2.0, dtype=dtype, device=device)
        
        # 对于就地操作，需要预先创建输出张量
        out = None
        if config.inplace:
            # 计算输出形状
            if len(shape_a) == 2 and len(shape_b) == 2:
                # 2D x 2D: {m, k} x {k, n} -> {m, n}
                output_shape = (shape_a[0], shape_b[1])
            elif len(shape_a) == 3 and len(shape_b) == 2:
                # 3D x 2D: {batch, m, k} x {k, n} -> {batch, m, n}
                output_shape = (shape_a[0], shape_a[1], shape_b[1])
            elif len(shape_a) == 3 and len(shape_b) == 3:
                # 3D x 3D: {batch, m, k} x {batch, k, n} -> {batch, m, n}
                output_shape = (shape_a[0], shape_a[1], shape_b[2])
            else:
                raise ValueError(f"MatMul inplace operation only supports 2D x 2D, 3D x 2D, or 3D x 3D matrices, got {shape_a} x {shape_b}")
            out = torch.empty(output_shape, dtype=dtype, device=device)
        
        # 预热
        if config.inplace:
            for _ in range(config.warmup_cnt):
                torch.matmul(x0, x1, out=out)
        else:
            for _ in range(config.warmup_cnt):
                result = torch.matmul(x0, x1)
        # 预热结束后同步，确保预热完成
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # 正式测试
        timer = Timer()
        timer.start()
        
        if config.inplace:
            # 对于就地操作，使用 out 参数
            for _ in range(config.repeat_cnt):
                torch.matmul(x0, x1, out=out)
        else:
            for _ in range(config.repeat_cnt):
                result = torch.matmul(x0, x1)
        # 正式测试结束后同步，确保所有计算完成
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        total_time_us = timer.elapsed_us()
        return total_time_us / config.repeat_cnt
    
    def get_default_shapes(self):
        """获取默认测试形状"""
        # 对于matmul，每个元素是包含两个shape的列表
        return [
            [(1, 1), (1, 1)],
            [(10, 10), (10, 10)],
            [(100, 100), (100, 100)],
            [(1000, 1000), (1000, 1000)],
            [(10000, 10000), (10000, 10000)],
        ]
    
    def validate_shapes(self, shapes: List[Tuple[int, ...]]) -> None:
        """验证形状是否有效"""
        if len(shapes) != 2:
            raise ValueError(f"MatMul requires exactly 2 shapes, got {len(shapes)}")
        
        shape_a = shapes[0]
        shape_b = shapes[1]
        
        # 验证至少是2维
        if len(shape_a) < 2 or len(shape_b) < 2:
            raise ValueError(
                f"MatMul requires at least 2D shapes, got shapes with {len(shape_a)} and {len(shape_b)} dimensions"
            )
        
        # 验证矩阵乘法维度兼容性
        if len(shape_a) == 2 and len(shape_b) == 2:
            # 标准矩阵乘法：{m, k} x {k, n} -> {m, n}
            if shape_a[1] != shape_b[0]:
                raise ValueError(
                    f"MatMul dimension mismatch: {shape_a} x {shape_b} "
                    f"(shape_a[1]={shape_a[1]} != shape_b[0]={shape_b[0]})"
                )
        elif len(shape_a) == 3 and len(shape_b) == 2:
            # 批量矩阵乘法：{batch, m, k} x {k, n} -> {batch, m, n}
            if shape_a[2] != shape_b[0]:
                raise ValueError(
                    f"MatMul dimension mismatch: {shape_a} x {shape_b} "
                    f"(shape_a[2]={shape_a[2]} != shape_b[0]={shape_b[0]})"
                )
        elif len(shape_a) == 3 and len(shape_b) == 3:
            # 批量矩阵乘法：{batch, m, k} x {batch, k, n} -> {batch, m, n}
            if shape_a[0] != shape_b[0] or shape_a[2] != shape_b[1]:
                raise ValueError(
                    f"MatMul dimension mismatch: {shape_a} x {shape_b} "
                    f"(batch or k dimension mismatch)"
                )


def benchmark_mat_mul_comparison(device_filter=None,
                                shape_filter=None,
                                warmup_cnt=5,
                                repeat_cnt=100,
                                inplace=False,
                                verbose=False):
    """
    运行mat_mul算子的PyTorch性能测试
    
    Args:
        device_filter: 设备过滤，'cpu'或'cuda'
        shape_filter: shape过滤，例如 '100,200:200,50'
        warmup_cnt: 预热次数
        repeat_cnt: 重复次数
        inplace: 是否使用就地操作
        verbose: 是否输出详细信息
    
    Returns:
        PyTorch测试结果列表，每个元素为:
        {'shape': str, 'device': str, 'dtype': str, 'time_us': float}
    """
    benchmark = MatMulBenchmark()
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
    benchmark = MatMulBenchmark()
    sys.exit(benchmark.run_from_command_line())
