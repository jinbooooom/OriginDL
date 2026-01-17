"""
Benchmark framework for PyTorch performance testing
学习C++ benchmark框架的设计，提供统一的Python性能测试接口
"""

import sys
import torch
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from parser_utils import parse_shape_string, parse_multiple_shapes_string
from timer import Timer


@dataclass
class BenchmarkConfig:
    """基准测试配置"""
    shapes: List[Tuple[int, ...]]  # 输入的shape列表（单shape或多shape）
    dtype: str = 'float32'          # 数据类型
    device: str = 'cpu'             # 设备（'cpu'或'cuda:0'等）
    warmup_cnt: int = 5             # 预热次数
    repeat_cnt: int = 100           # 重复次数
    
    def __post_init__(self):
        """确保warmup_cnt和repeat_cnt有默认值"""
        if self.warmup_cnt is None:
            self.warmup_cnt = 5
        if self.repeat_cnt is None:
            self.repeat_cnt = 100


class BenchmarkFramework(ABC):
    """基准测试框架基类
    
    提供命令行参数解析、测试循环等通用功能
    子类需要实现 run_benchmark() 方法来执行具体的PyTorch基准测试
    """
    
    @abstractmethod
    def get_required_shapes_count(self) -> int:
        """获取所需的shape数量（纯虚函数，由子类实现）
        
        Returns:
            所需的shape数量，例如add算子返回1，matmul算子返回2
        """
        pass
    
    @abstractmethod
    def run_benchmark(self, config: BenchmarkConfig) -> float:
        """运行基准测试（纯虚函数，由子类实现）
        
        Args:
            config: 基准测试配置
            
        Returns:
            平均执行时间（微秒）
        """
        pass
    
    def get_default_shapes(self) -> List[List[Tuple[int, ...]]]:
        """获取默认测试形状
        
        Returns:
            默认形状列表，每个元素是一个shape组合
            对于单shape算子，每个元素是单个shape的列表
            默认实现：对于单shape算子，返回 [(1,1)], [(10,10)], [(100,100)], [(1000,1000)], [(10000,10000)]
            多shape算子需要重写此方法
        """
        if self.get_required_shapes_count() == 1:
            return [
                [(1, 1)],
                [(10, 10)],
                [(100, 100)],
                [(1000, 1000)],
                [(10000, 10000)],
            ]
        else:
            raise NotImplementedError(
                f"get_default_shapes() must be overridden for operators requiring "
                f"{self.get_required_shapes_count()} shapes"
            )
    
    def validate_shapes(self, shapes: List[Tuple[int, ...]]) -> None:
        """验证形状是否有效（可选，子类可重写）
        
        Args:
            shapes: 要验证的形状列表
            
        Raises:
            ValueError: 如果形状无效
        """
        pass
    
    def get_operator_name(self) -> str:
        """获取算子名称（用于帮助信息）
        
        Returns:
            算子名称字符串
        """
        return self.__class__.__name__.replace('Benchmark', '')
    
    def _parse_device(self, device_str: str) -> torch.device:
        """解析设备字符串为torch.device对象
        
        Args:
            device_str: 设备字符串，例如 'cpu' 或 'cuda:0'
            
        Returns:
            torch.device对象
        """
        if device_str == 'cpu':
            return torch.device('cpu')
        elif device_str.startswith('cuda:'):
            index = int(device_str.split(':')[1])
            return torch.device(f'cuda:{index}')
        else:
            raise ValueError(f"Invalid device string: {device_str}")
    
    def _get_available_devices(self, device_filter: Optional[str] = None) -> List[str]:
        """获取可用设备列表
        
        Args:
            device_filter: 设备过滤，'cpu'或'cuda'，None表示所有设备
            
        Returns:
            可用设备列表，例如 ['cpu', 'cuda:0']
        """
        devices = ['cpu']
        
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            for i in range(device_count):
                devices.append(f'cuda:{i}')
        
        # 应用过滤
        if device_filter == 'cpu':
            return ['cpu']
        elif device_filter == 'cuda':
            return [d for d in devices if d.startswith('cuda')]
        elif device_filter:
            # 如果指定了具体设备，检查是否可用
            if device_filter in devices:
                return [device_filter]
            else:
                raise ValueError(f"Device '{device_filter}' is not available")
        
        return devices
    
    def _format_shape_string(self, shapes: List[Tuple[int, ...]]) -> str:
        """格式化shape为字符串，例如 [(100, 100)] -> "{100,100}" 或 [(100, 200), (200, 50)] -> "{100,200}:{200,50}"
        
        Args:
            shapes: shape列表
            
        Returns:
            格式化的shape字符串
        """
        shape_strs = []
        for shape in shapes:
            shape_str = '{' + ','.join(str(d) for d in shape) + '}'
            shape_strs.append(shape_str)
        return ':'.join(shape_strs)
    
    def _parse_shapes_from_string(self, shape_filter: str) -> List[Tuple[int, ...]]:
        """从字符串解析shapes
        
        Args:
            shape_filter: shape字符串，例如 '1000,1000' 或 '100,200:200,50'
            
        Returns:
            解析后的shape列表
        """
        required_count = self.get_required_shapes_count()
        
        # 检查是否包含冒号（多个shape）
        if ':' in shape_filter:
            shapes = parse_multiple_shapes_string(shape_filter)
        else:
            # 单个shape
            dims = parse_shape_string(shape_filter)
            if not dims:
                raise ValueError(f"Invalid shape string: {shape_filter}")
            shapes = [tuple(dims)]
        
        # 验证shape数量
        if len(shapes) != required_count:
            raise ValueError(
                f"Operator requires {required_count} shape(s), but got {len(shapes)} shape(s) in '{shape_filter}'"
            )
        
        # 验证形状（子类可以重写此方法）
        self.validate_shapes(shapes)
        
        return shapes
    
    def run(self,
            device_filter: Optional[str] = None,
            shape_filter: Optional[str] = None,
            warmup_cnt: int = 5,
            repeat_cnt: int = 100,
            verbose: bool = False) -> List[Dict]:
        """运行基准测试主函数
        
        解析参数，运行测试循环，返回结果列表
        与C++框架的run()方法功能类似
        
        Args:
            device_filter: 设备过滤（'cpu'或'cuda'），None表示所有设备
            shape_filter: shape过滤（例如'1000,1000'），None表示使用默认shapes
            warmup_cnt: 预热次数
            repeat_cnt: 重复次数
            verbose: 是否输出详细信息
            
        Returns:
            PyTorch测试结果列表，每个元素为:
            {'shape': str, 'device': str, 'dtype': str, 'time_us': float}
        """
        # 获取可用设备
        devices = self._get_available_devices(device_filter)
        
        # 获取测试shapes
        if shape_filter:
            shapes_list = [self._parse_shapes_from_string(shape_filter)]
        else:
            shapes_list = self.get_default_shapes()
        
        # 测试数据类型
        dtypes = ['float32']
        
        # 运行测试并收集结果
        results = []
        
        for shapes in shapes_list:
            for dtype in dtypes:
                for device_str in devices:
                    try:
                        config = BenchmarkConfig(
                            shapes=shapes,
                            dtype=dtype,
                            device=device_str,
                            warmup_cnt=warmup_cnt,
                            repeat_cnt=repeat_cnt
                        )
                        
                        avg_time_us = self.run_benchmark(config)
                        
                        shape_str = self._format_shape_string(shapes)
                        
                        results.append({
                            'shape': shape_str,
                            'device': device_str,
                            'dtype': dtype,
                            'time_us': avg_time_us
                        })
                        
                    except Exception as e:
                        if verbose:
                            shape_str = self._format_shape_string(shapes)
                            print(f"Error testing {shape_str} {device_str} {dtype}: {e}", file=sys.stderr)
                        # 跳过失败的测试
        
        return results
    
    def run_from_command_line(self, argv=None) -> int:
        """从命令行运行基准测试（类似C++的run(int argc, char *argv[])）
        
        解析命令行参数，运行测试，输出结果
        
        Args:
            argv: 命令行参数列表，如果为None则使用sys.argv
            
        Returns:
            程序退出码（0表示成功）
        """
        import argparse
        
        parser = argparse.ArgumentParser(description=f'{self.get_operator_name()} operator PyTorch benchmark')
        parser.add_argument('-d', '--device', help='Device filter (cpu or cuda)')
        parser.add_argument('-s', '--shape', help='Tensor shape filter')
        parser.add_argument('-w', '--warmup', type=int, help='Number of warmup iterations (default: 5)')
        parser.add_argument('-r', '--repeat', type=int, help='Number of repeat iterations (default: 100)')
        parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')
        
        args = parser.parse_args(argv)
        
        # 运行测试
        results = self.run(
            device_filter=args.device,
            shape_filter=args.shape,
            warmup_cnt=args.warmup,
            repeat_cnt=args.repeat,
            verbose=args.verbose
        )
        
        # 输出结果（制表符分隔，与C++格式一致）
        if results:
            # 计算列宽
            max_shape_width = max(len('shape'), max(len(r['shape']) for r in results))
            repeat_str = str(args.repeat or 100)
            max_repeat_width = max(len('repeat'), len(repeat_str))
            max_device_width = max(len('device'), max(len(r['device']) for r in results))
            max_dtype_width = max(len('dtype'), max(len(r['dtype']) for r in results))
            max_time_width = max(len('pytorch_time_us'), max(len(f"{r['time_us']:.4f}") for r in results))
            
            # 输出表头
            print(f"{'shape':<{max_shape_width}}\t{'repeat':<{max_repeat_width}}\t{'device':<{max_device_width}}\t"
                  f"{'dtype':<{max_dtype_width}}\t{'pytorch_time_us':<{max_time_width}}")
            
            # 输出结果
            for r in results:
                print(f"{r['shape']:<{max_shape_width}}\t{repeat_str:<{max_repeat_width}}\t"
                      f"{r['device']:<{max_device_width}}\t{r['dtype']:<{max_dtype_width}}\t"
                      f"{r['time_us']:.4f}")
        else:
            print("No results generated", file=sys.stderr)
            return 1
        
        return 0
