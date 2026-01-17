#!/usr/bin/env python3
"""
OriginDL算子性能对比测试脚本
调用C++性能测试程序和PyTorch测试程序，输出OriginDL与PyTorch的性能对比数据
"""

import subprocess
import sys
import argparse
import os
from pathlib import Path
from typing import List, Dict, Optional, Tuple


def discover_benchmark_operators(project_root: Path) -> List[str]:
    """自动发现可用的benchmark算子
    
    Args:
        project_root: 项目根目录路径
    
    Returns:
        算子名称列表，例如 ['add']
    """
    operators = []
    benchmark_dir = project_root / "tests" / "benchmark"
    
    if not benchmark_dir.exists():
        return operators
    
    # 遍历 benchmark 目录下的子目录
    for item in benchmark_dir.iterdir():
        if item.is_dir() and item.name != "common":
            # 检查是否有对应的 Python 测试脚本
            pytorch_script = item / f"bench_{item.name}.py"
            if pytorch_script.exists():
                operators.append(item.name)
    
    return sorted(operators)


def find_benchmark_executable(operator: str, project_root: Path) -> Optional[str]:
    """查找benchmark可执行文件
    
    Args:
        operator: 算子名称，例如 'add'
        project_root: 项目根目录路径
    
    Returns:
        可执行文件路径，如果未找到返回None
    """
    possible_paths = [
        project_root / "build" / "bin" / "benchmark" / f"bench_{operator}",
        project_root / "torch_build" / "bin" / "benchmark" / f"bench_{operator}",
        project_root / "build" / "bin" / f"bench_{operator}",
        project_root / "torch_build" / "bin" / f"bench_{operator}",
    ]
    
    for path in possible_paths:
        if path.exists() and os.access(path, os.X_OK):
            return str(path)
    
    return None


def import_pytorch_benchmark_module(operator: str, project_root: Path):
    """动态导入PyTorch benchmark模块
    
    Args:
        operator: 算子名称，例如 'add'
        project_root: 项目根目录路径
    
    Returns:
        导入的模块对象
    """
    module_path = project_root / "tests" / "benchmark" / operator / f"bench_{operator}.py"
    
    if not module_path.exists():
        raise FileNotFoundError(f"PyTorch benchmark script not found: {module_path}")
    
    # 添加模块路径到sys.path
    module_dir = str(module_path.parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    
    # 导入模块
    module_name = f"bench_{operator}"
    if module_name in sys.modules:
        del sys.modules[module_name]
    
    import importlib.util
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module


def run_cpp_benchmark(executable_path: str,
                      device_filter: Optional[str] = None,
                      shape_filter: Optional[str] = None,
                      warmup_cnt: Optional[int] = None,
                      repeat_cnt: Optional[int] = None) -> List[Dict]:
    """运行C++ benchmark可执行文件并解析输出
    
    Args:
        executable_path: C++可执行文件路径
        device_filter: 设备过滤，'cpu' 或 'cuda'
        shape_filter: shape过滤，例如 '1000,1000'
        warmup_cnt: 预热次数
        repeat_cnt: 重复次数
    
    Returns:
        OriginDL测试结果列表，每个元素为:
        {'shape': str, 'device': str, 'dtype': str, 'time_us': float, 'repeat_cnt': int}
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
        # 运行C++程序并捕获输出
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output_lines = result.stdout.strip().split('\n')
        
        # 跳过表头行（第一行）
        if len(output_lines) < 2:
            return []
        
        results = []
        for line in output_lines[1:]:  # 跳过表头
            if not line.strip():
                continue
            
            # 解析制表符分隔的行: shape \t repeat \t device \t dtype \t origindl_time_us
            parts = line.split('\t')
            if len(parts) < 5:
                continue
            
            shape_str = parts[0].strip()
            repeat_str = parts[1].strip()
            device_str = parts[2].strip()
            dtype_str = parts[3].strip()
            time_str = parts[4].strip()
            
            # 统一shape格式：移除空格，例如 {1, 1} -> {1,1}
            # 保持与Python输出格式一致
            shape_str = shape_str.replace(' ', '')
            
            try:
                time_us = float(time_str)
                repeat_cnt_value = int(repeat_str) if repeat_str.isdigit() else None
                
                results.append({
                    'shape': shape_str,
                    'device': device_str,
                    'dtype': dtype_str,
                    'time_us': time_us,
                    'repeat_cnt': repeat_cnt_value
                })
            except ValueError:
                continue
        
        return results
        
    except subprocess.CalledProcessError as e:
        # C++程序执行失败
        return []
    except Exception as e:
        return []


def run_operator_benchmark(operator: str, 
                          project_root: Path,
                          device_filter: Optional[str] = None,
                          shape_filter: Optional[str] = None,
                          warmup_cnt: Optional[int] = None,
                          repeat_cnt: Optional[int] = None,
                          verbose: bool = False) -> Optional[Tuple[List[Dict], List[Dict]]]:
    """运行单个算子的性能对比测试
    
    Args:
        operator: 算子名称
        project_root: 项目根目录路径
        device_filter: 设备过滤，'cpu' 或 'cuda'
        shape_filter: shape过滤，例如 '1000,1000'
        verbose: 是否输出详细信息
    
    Returns:
        (origindl_results, pytorch_results) 元组，如果失败返回None
    """
    try:
        # 1. 调用C++可执行文件获取OriginDL结果
        executable_path = find_benchmark_executable(operator, project_root)
        origindl_results = []
        
        if executable_path:
            origindl_results = run_cpp_benchmark(
                executable_path=executable_path,
                device_filter=device_filter,
                shape_filter=shape_filter,
                warmup_cnt=warmup_cnt,
                repeat_cnt=repeat_cnt
            )
        elif verbose:
            print(f"Warning: C++ executable not found for operator '{operator}'", file=sys.stderr)
        
        # 2. 调用Python函数获取PyTorch结果
        module = import_pytorch_benchmark_module(operator, project_root)
        
        # 查找benchmark函数（例如 benchmark_add_comparison）
        benchmark_func_name = f"benchmark_{operator}_comparison"
        if not hasattr(module, benchmark_func_name):
            if verbose:
                print(f"Warning: Module {module.__name__} does not have function {benchmark_func_name}", 
                      file=sys.stderr)
            return None
        
        benchmark_func = getattr(module, benchmark_func_name)
        
        # 运行PyTorch benchmark测试
        # 检查函数是否支持参数
        import inspect
        func_sig = inspect.signature(benchmark_func)
        func_params = {}
        
        if 'device_filter' in func_sig.parameters:
            func_params['device_filter'] = device_filter
        if 'shape_filter' in func_sig.parameters:
            func_params['shape_filter'] = shape_filter
        if 'warmup_cnt' in func_sig.parameters and warmup_cnt is not None:
            func_params['warmup_cnt'] = warmup_cnt
        if 'repeat_cnt' in func_sig.parameters and repeat_cnt is not None:
            func_params['repeat_cnt'] = repeat_cnt
        if 'verbose' in func_sig.parameters:
            func_params['verbose'] = verbose
        
        # 调用Python函数获取PyTorch结果
        pytorch_results = benchmark_func(**func_params)
        
        return origindl_results, pytorch_results
        
    except Exception as e:
        if verbose:
            print(f"Error running benchmark for operator '{operator}': {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
        return None


def print_comparison_table(operator: str,
                          origindl_results: List[Dict], 
                          pytorch_results: List[Dict]):
    """打印性能对比表格
    
    Args:
        operator: 算子名称
        origindl_results: OriginDL测试结果列表
        pytorch_results: PyTorch测试结果列表
    """

    origindl_map = {(r['shape'], r['device'], r['dtype']): {'time_us': r['time_us'], 'repeat_cnt': r.get('repeat_cnt', 100)} 
                    for r in origindl_results}
    pytorch_map = {(r['shape'], r['device'], r['dtype']): r['time_us'] 
                   for r in pytorch_results}
    
    # 算子名称（首字母大写）
    operator_name = operator.capitalize()
    
    # 收集所有需要输出的行，用于计算列宽
    all_keys = sorted(set(origindl_map.keys()) | set(pytorch_map.keys()))
    rows = []
    for key in all_keys:
        shape, device, dtype = key
        origindl_data = origindl_map.get(key, None)
        pytorch_time = pytorch_map.get(key, None)
        
        if origindl_data is not None:
            origindl_time = origindl_data['time_us']
            repeat_cnt = origindl_data['repeat_cnt']
        else:
            origindl_time = None
            repeat_cnt = None
        
        repeat_str = str(repeat_cnt) if repeat_cnt is not None else 'N/A'
        origindl_str = f"{origindl_time:.4f}" if origindl_time is not None else 'N/A'
        pytorch_str = f"{pytorch_time:.4f}" if pytorch_time is not None else 'N/A'
        
        if origindl_time is not None and pytorch_time is not None:
            speedup = pytorch_time / origindl_time if origindl_time > 0 else 0.0
            speedup_str = f"{speedup:.4f}"
        else:
            speedup_str = 'N/A'
        
        rows.append({
            'shape': shape,
            'repeat': repeat_str,
            'device': device,
            'dtype': dtype,
            'origindl': origindl_str,
            'pytorch': pytorch_str,
            'speedup': speedup_str
        })
    
    # 计算每列的最大宽度
    max_shape_width = max(5, max(len(row['shape']) for row in rows))  # "Shape" 长度
    max_repeat_width = max(6, max(len(row['repeat']) for row in rows))  # "Repeat" 长度
    max_device_width = max(6, max(len(row['device']) for row in rows))  # "Device" 长度
    max_dtype_width = max(5, max(len(row['dtype']) for row in rows))  # "Dtype" 长度
    max_origindl_width = max(13, max(len(row['origindl']) for row in rows))  # "OriginDL(us)" 长度
    max_pytorch_width = max(13, max(len(row['pytorch']) for row in rows))  # "PyTorch(us)" 长度
    max_speedup_width = max(7, max(len(row['speedup']) for row in rows))  # "Speedup" 长度
    
    # 输出对比表格（每列之间至少3个字符间距）
    column_spacing = 3
    
    # 计算表头行宽度
    header_line = (f"{'Shape':<{max_shape_width}}{' ' * column_spacing}"
                   f"{'Repeat':<{max_repeat_width}}{' ' * column_spacing}"
                   f"{'Device':<{max_device_width}}{' ' * column_spacing}"
                   f"{'Dtype':<{max_dtype_width}}{' ' * column_spacing}"
                   f"{'OriginDL(us)':<{max_origindl_width}}{' ' * column_spacing}"
                   f"{'PyTorch(us)':<{max_pytorch_width}}{' ' * column_spacing}"
                   f"{'Speedup':<{max_speedup_width}}")
    
    # 计算最长数据行宽度
    max_line_width = len(header_line)
    for row in rows:
        data_line = (f"{row['shape']:<{max_shape_width}}{' ' * column_spacing}"
                     f"{row['repeat']:<{max_repeat_width}}{' ' * column_spacing}"
                     f"{row['device']:<{max_device_width}}{' ' * column_spacing}"
                     f"{row['dtype']:<{max_dtype_width}}{' ' * column_spacing}"
                     f"{row['origindl']:<{max_origindl_width}}{' ' * column_spacing}"
                     f"{row['pytorch']:<{max_pytorch_width}}{' ' * column_spacing}"
                     f"{row['speedup']:<{max_speedup_width}}")
        max_line_width = max(max_line_width, len(data_line))
    
    print("\n" + "="*max_line_width)
    print(f"{operator_name} Operator Performance Comparison")
    print("="*max_line_width)
    print(header_line)
    print("-"*max_line_width)
    
    # 输出所有结果
    for row in rows:
        data_line = (f"{row['shape']:<{max_shape_width}}{' ' * column_spacing}"
                     f"{row['repeat']:<{max_repeat_width}}{' ' * column_spacing}"
                     f"{row['device']:<{max_device_width}}{' ' * column_spacing}"
                     f"{row['dtype']:<{max_dtype_width}}{' ' * column_spacing}"
                     f"{row['origindl']:<{max_origindl_width}}{' ' * column_spacing}"
                     f"{row['pytorch']:<{max_pytorch_width}}{' ' * column_spacing}"
                     f"{row['speedup']:<{max_speedup_width}}")
        print(data_line)
    
    print("="*max_line_width)


def main():
    """命令行入口"""
    # 获取项目根目录
    script_path = Path(__file__).resolve()
    project_root = script_path.parent
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(
        description='OriginDL operator performance comparison with PyTorch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 benchmark.py                    # Test all operators
  python3 benchmark.py -f add             # Test add operator only
  python3 benchmark.py -f add -d cpu      # Test add operator on CPU only
  python3 benchmark.py -f add -d cpu -s 1000,1000  # Test add with specific shape
  python3 benchmark.py -f add -w 5 -r 50  # Test with custom warmup and repeat counts
        """
    )
    
    parser.add_argument(
        '-f', '--function',
        dest='operator',
        help='Operator name to benchmark (e.g., add). If not specified, all operators will be tested.'
    )
    parser.add_argument(
        '-d', '--device',
        help='Device type filter (cpu or cuda). If not specified, all devices will be tested.'
    )
    parser.add_argument(
        '-s', '--shape',
        help='Tensor shape filter (e.g., "1000,1000"). If not specified, default shapes will be used.'
    )
    parser.add_argument(
        '-w', '--warmup',
        type=int,
        help='Number of warmup iterations (default: 10)'
    )
    parser.add_argument(
        '-r', '--repeat',
        type=int,
        help='Number of repeat iterations (default: 100)'
    )
    
    args = parser.parse_args()
    
    # 发现可用的算子
    available_operators = discover_benchmark_operators(project_root)
    
    if not available_operators:
        print("Error: No benchmark operators found.", file=sys.stderr)
        print("Please ensure benchmark operators are built and available.", file=sys.stderr)
        sys.exit(1)
    
    # 确定要测试的算子列表
    if args.operator:
        if args.operator not in available_operators:
            print(f"Error: Operator '{args.operator}' not found.", file=sys.stderr)
            print(f"Available operators: {', '.join(available_operators)}", file=sys.stderr)
            sys.exit(1)
        operators_to_test = [args.operator]
    else:
        operators_to_test = available_operators
    
    # 运行测试
    success_count = 0
    fail_count = 0
    
    for operator in operators_to_test:
        if len(operators_to_test) > 1:
            print(f"\n{'='*80}", file=sys.stderr)
            print(f"Testing operator: {operator}", file=sys.stderr)
            print('='*80, file=sys.stderr)
        
        result = run_operator_benchmark(
            operator=operator,
            project_root=project_root,
            device_filter=args.device,
            shape_filter=args.shape,
            warmup_cnt=args.warmup,
            repeat_cnt=args.repeat,
            verbose=(len(operators_to_test) == 1)  # 只在测试单个算子时输出详细信息
        )
        
        if result is not None:
            origindl_results, pytorch_results = result
            print_comparison_table(operator, origindl_results, pytorch_results)
            success_count += 1
        else:
            print(f"\nError: Failed to run benchmark for operator '{operator}'", file=sys.stderr)
            fail_count += 1
    
    # 输出总结
    if len(operators_to_test) > 1:
        print(f"\n{'='*80}", file=sys.stderr)
        print(f"Summary: {success_count} succeeded, {fail_count} failed", file=sys.stderr)
    
    # 如果所有测试都失败，返回错误码
    if fail_count > 0 and success_count == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
