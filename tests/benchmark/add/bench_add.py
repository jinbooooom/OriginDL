#!/usr/bin/env python3
"""
PyTorch Add算子性能测试
测试PyTorch的add算子性能，输出格式与C++版本一致
"""

import sys
import os
import torch
import argparse
import subprocess
from typing import List, Dict, Optional, Tuple
from pathlib import Path

# 导入common工具
sys.path.insert(0, str(Path(__file__).parent.parent))
from common.parser_utils import parse_shape_string
from common.timer import Timer


def device_type_to_string(device_str: str) -> str:
    """
    将设备字符串转换为简单的设备字符串 "cpu" 或 "cuda"
    @note torch.device("cuda") 会返回 "cuda:0"，我们需要 "cuda"
    """
    if device_str.startswith("cuda"):
        return "cuda"
    return "cpu"


def shape_to_string(shape: List[int]) -> str:
    """
    将shape列表转换为字符串格式 {dim1, dim2, ...}
    与C++版本的Shape::to_string()格式一致
    """
    if not shape:
        return "{}"
    return "{" + ", ".join(str(d) for d in shape) + "}"


def parse_shape(shape_str: str) -> List[int]:
    """解析形状字符串，例如 '{100,100}' -> [100, 100]"""
    # 移除花括号并分割
    shape_str = shape_str.strip('{}')
    if not shape_str:
        return []
    return [int(x.strip()) for x in shape_str.split(',')]


def benchmark_pytorch_add(shape: List[int], dtype_str: str, device_str: str,
                          warmup_iterations: int = 10,
                          benchmark_iterations: int = 100) -> float:
    """
    运行性能测试
    
    Args:
        shape: 输入张量形状
        dtype_str: 数据类型字符串
        device_str: 设备类型字符串
        warmup_iterations: 预热次数
        benchmark_iterations: 正式测试次数
    
    Returns:
        平均执行时间（微秒）
    """
    # 转换数据类型
    dtype_map = {
        'float32': torch.float32,
        'float64': torch.float64,
        'int32': torch.int32,
        'int64': torch.int64,
    }
    dtype = dtype_map.get(dtype_str, torch.float32)
    
    # 转换设备
    device = torch.device(device_str)
    
    # 创建输入张量
    numel = 1
    for dim in shape:
        numel *= dim
    
    x0 = torch.ones(shape, dtype=dtype, device=device)
    x1 = torch.ones(shape, dtype=dtype, device=device) * 2.0
    
    # 预热
    for _ in range(warmup_iterations):
        result = x0 + x1
        # 确保计算完成（对于CUDA，需要同步）
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    # 正式测试
    timer = Timer()
    timer.start()
    
    for _ in range(benchmark_iterations):
        result = x0 + x1
        # 确保计算完成
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    total_time_us = timer.elapsed_us()
    return total_time_us / benchmark_iterations


def usage(program_name: str):
    """打印使用说明"""
    print(f"Usage: {program_name} [OPTIONS]")
    print("Options:")
    print("  -d, --device DEVICE       Device type: cpu or cuda (can be specified multiple times)")
    print("                            If not specified, tests all available devices")
    print("  -s, --shape SHAPE         Tensor shape, e.g., \"100,100\" or \"1000,1000\"")
    print("                            (can be specified multiple times)")
    print("                            If not specified, uses default shapes")
    print("  -w, --warmup ITERATIONS   Number of warmup iterations (default: 10)")
    print("  -r, --repeat ITERATIONS   Number of repeat iterations (default: 100)")
    print("  -h, --help                Show this help message")


def main():
    # 默认测试配置
    default_shapes = [
        [100, 100],
        [1000, 1000],
        [100, 100, 100],
        [1000],
        [10, 10, 10, 10],
    ]
    
    shapes = []
    devices = []
    dtypes = ['float32']
    
    use_default_shapes = True
    use_default_devices = True
    warmup_iterations = 5
    benchmark_iterations = 100
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='PyTorch Add operator performance benchmark',
        add_help=False
    )
    parser.add_argument('-d', '--device', action='append', dest='devices',
                       help='Device type: cpu or cuda (can be specified multiple times)')
    parser.add_argument('-s', '--shape', action='append', dest='shapes',
                       help='Tensor shape, e.g., "100,100" or "1000,1000" (can be specified multiple times)')
    parser.add_argument('-w', '--warmup', type=int, dest='warmup',
                       help='Number of warmup iterations (default: 5)')
    parser.add_argument('-r', '--repeat', type=int, dest='repeat',
                       help='Number of repeat iterations (default: 100)')
    parser.add_argument('-h', '--help', action='store_true',
                       help='Show this help message')
    
    args = parser.parse_args()
    
    # 处理help
    if args.help:
        usage(sys.argv[0])
        return 0
    
    # 处理devices
    if args.devices:
        use_default_devices = False
        for device_str in args.devices:
            if device_str == "cpu":
                devices.append("cpu")
            elif device_str == "cuda":
                devices.append("cuda")
            else:
                print(f"Error: Invalid device '{device_str}'. Use 'cpu' or 'cuda'.", file=sys.stderr)
                return 1
    
    # 处理shapes
    if args.shapes:
        use_default_shapes = False
        for shape_str in args.shapes:
            shape = parse_shape_string(shape_str)
            if not shape:
                print(f"Error: Invalid shape '{shape_str}'. Expected format: 'dim1,dim2,...'", file=sys.stderr)
                return 1
            shapes.append(shape)
    
    # 处理warmup
    if args.warmup is not None:
        if args.warmup < 0:
            print("Error: Warmup iterations must be non-negative", file=sys.stderr)
            return 1
        warmup_iterations = args.warmup
    
    # 处理repeat
    if args.repeat is not None:
        if args.repeat <= 0:
            print("Error: Repeat iterations must be positive", file=sys.stderr)
            return 1
        benchmark_iterations = args.repeat
    
    # 如果使用默认shapes，则使用默认列表
    if use_default_shapes:
        shapes = default_shapes
    
    # 如果使用默认devices，则根据CUDA可用性决定
    if use_default_devices:
        devices.append("cpu")
        if torch.cuda.is_available():
            devices.append("cuda")
    else:
        # 检查指定的CUDA设备是否可用
        has_cuda = "cuda" in devices
        if has_cuda:
            if not torch.cuda.is_available():
                print("Warning: CUDA is not available, skipping CUDA tests", file=sys.stderr)
                devices = [d for d in devices if d != "cuda"]
    
    if not shapes:
        print("Error: No shapes specified", file=sys.stderr)
        return 1
    
    if not devices:
        print("Error: No devices specified", file=sys.stderr)
        return 1
    
    # 输出表头（制表符分隔，与C++版本一致）
    # 注意：输出单位为微秒（us）
    print("shape\tdevice\tdtype\tpytorch_time_us")
    
    # 运行测试
    for shape in shapes:
        for dtype in dtypes:
            for device_str in devices:
                try:
                    avg_time_us = benchmark_pytorch_add(
                        shape, dtype, device_str,
                        warmup_iterations, benchmark_iterations
                    )
                    
                    # 输出结果（制表符分隔）
                    print(f"{shape_to_string(shape)}\t"
                          f"{device_type_to_string(device_str)}\t"
                          f"{dtype}\t"
                          f"{avg_time_us:.4f}")
                except Exception as e:
                    print(f"Error testing {shape_to_string(shape)} {device_type_to_string(device_str)} {dtype}: {e}",
                          file=sys.stderr)
    
    return 0


def run_origindl_benchmark(executable_path: str, 
                          device_filter: Optional[str] = None,
                          shape_filter: Optional[str] = None,
                          warmup_cnt: Optional[int] = None,
                          repeat_cnt: Optional[int] = None) -> List[Dict]:
    """运行OriginDL性能测试并解析结果
    
    Args:
        executable_path: benchmark可执行文件路径
        device_filter: 设备过滤，'cpu' 或 'cuda'，None表示不过滤
        shape_filter: shape过滤，例如 '1000,1000'，None表示不过滤
        warmup_cnt: 预热次数，None表示使用默认值
        repeat_cnt: 重复次数，None表示使用默认值
    
    Returns:
        结果列表，每个元素包含 'shape', 'device', 'dtype', 'time_us'（微秒）
    """
    # 构建命令行参数
    cmd = [executable_path]
    if device_filter:
        cmd.extend(['-d', device_filter])
    if shape_filter:
        cmd.extend(['-s', shape_filter])
    if warmup_cnt is not None:
        cmd.extend(['-w', str(warmup_cnt)])
    if repeat_cnt is not None:
        cmd.extend(['-r', str(repeat_cnt)])
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True
        )
        
        # 解析输出（制表符分隔）
        lines = result.stdout.strip().split('\n')
        if len(lines) < 2:
            print("Error: No benchmark results from OriginDL", file=sys.stderr)
            return []
        
        # 跳过表头
        results = []
        for line in lines[1:]:
            if not line.strip():
                continue
            # 使用制表符分割，但去除格式化空格
            parts = line.split('\t')
            # 过滤空字符串
            parts = [p.strip() for p in parts if p.strip()]
            if len(parts) >= 5:
                shape_str = parts[0]
                repeat_cnt = int(parts[1])  # 重复测试次数
                device = parts[2]
                dtype = parts[3]
                time_us = float(parts[4])  # C++程序输出的是微秒（us）
                results.append({
                    'shape': shape_str,
                    'device': device,
                    'dtype': dtype,
                    'repeat_cnt': repeat_cnt,
                    'time_us': time_us
                })
        
        return results
    except subprocess.CalledProcessError as e:
        print(f"Error running OriginDL benchmark: {e}", file=sys.stderr)
        print(f"stdout: {e.stdout}", file=sys.stderr)
        print(f"stderr: {e.stderr}", file=sys.stderr)
        return []
    except FileNotFoundError:
        print(f"Error: Executable not found: {executable_path}", file=sys.stderr)
        return []


def benchmark_add_comparison(executable_path: Optional[str] = None,
                             device_filter: Optional[str] = None,
                             shape_filter: Optional[str] = None,
                             warmup_cnt: Optional[int] = None,
                             repeat_cnt: Optional[int] = None,
                             verbose: bool = True) -> Tuple[List[Dict], List[Dict]]:
    """运行add算子的性能对比测试
    
    Args:
        executable_path: benchmark可执行文件路径，None表示自动查找
        device_filter: 设备过滤，'cpu' 或 'cuda'，None表示不过滤
        shape_filter: shape过滤，例如 '1000,1000'，None表示不过滤
        verbose: 是否输出详细信息到stderr
    
    Returns:
        (origindl_results, pytorch_results) 元组，每个是结果字典列表
    """
    if executable_path is None:
        # 自动查找可执行文件
        script_dir = Path(__file__).parent.absolute()
        project_root = script_dir.parent.parent.parent
        
        possible_paths = [
            project_root / "build" / "bin" / "benchmark" / "bench_add",
            project_root / "torch_build" / "bin" / "benchmark" / "bench_add",
            project_root / "build" / "bin" / "bench_add",
            project_root / "torch_build" / "bin" / "bench_add",
        ]
        
        for path in possible_paths:
            if path.exists() and os.access(path, os.X_OK):
                executable_path = str(path)
                break
        
        if not executable_path:
            raise FileNotFoundError(
                "Cannot find bench_add executable. "
                "Please specify the path manually or ensure the executable is in one of: "
                "build/bin/benchmark/bench_add, torch_build/bin/benchmark/bench_add"
            )
        elif verbose:
            print(f"Auto-detected executable: {executable_path}", file=sys.stderr)
    
    # 运行OriginDL性能测试
    if verbose:
        print("Running OriginDL benchmark...", file=sys.stderr)
    origindl_results = run_origindl_benchmark(
        executable_path, device_filter, shape_filter, warmup_cnt, repeat_cnt
    )
    
    if not origindl_results:
        raise RuntimeError("Failed to get OriginDL benchmark results")
    
    # 使用默认值或传入的参数
    warmup_iterations = warmup_cnt if warmup_cnt is not None else 10
    repeat_iterations = repeat_cnt if repeat_cnt is not None else 100
    
    # 测试PyTorch性能
    if verbose:
        print("Running PyTorch benchmark...", file=sys.stderr)
    pytorch_results = []
    
    for result in origindl_results:
        shape = parse_shape(result['shape'])
        dtype_str = result['dtype']
        device_str = result['device']
        
        # 检查设备是否可用
        if device_str == 'cuda' and not torch.cuda.is_available():
            if verbose:
                print(f"Warning: CUDA not available, skipping {result['shape']} on {device_str}", 
                      file=sys.stderr)
            continue
        
        try:
            pytorch_time_us = benchmark_pytorch_add(
                shape, dtype_str, device_str, warmup_iterations, repeat_iterations
            )
            pytorch_results.append({
                'shape': result['shape'],
                'device': device_str,
                'dtype': dtype_str,
                'time_us': pytorch_time_us
            })
        except Exception as e:
            if verbose:
                print(f"Error benchmarking PyTorch for {result['shape']} on {device_str}: {e}", 
                      file=sys.stderr)
    
    return origindl_results, pytorch_results


if __name__ == "__main__":
    sys.exit(main())
