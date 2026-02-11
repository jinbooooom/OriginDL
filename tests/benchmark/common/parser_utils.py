"""
Benchmark common utilities for parsing
"""

from typing import List, Tuple


def parse_shape_string(shape_str: str) -> List[int]:
    """
    解析shape字符串，例如 "100,100" -> [100, 100]
    
    Args:
        shape_str: shape字符串，例如 "100,100" 或 "1000,1000"
    
    Returns:
        解析后的shape列表，如果解析失败返回空列表
    """
    if not shape_str:
        return []
    
    dims = []
    for item in shape_str.split(','):
        item = item.strip()
        if not item:
            continue
        try:
            dim = int(item)
            if dim < 0:
                return []  # 返回空列表表示解析失败
            dims.append(dim)
        except ValueError:
            return []  # 返回空列表表示解析失败
    
    return dims


def parse_multiple_shapes_string(shapes_str: str) -> List[Tuple[int, ...]]:
    """
    解析多个shape字符串，例如 "100,200:200,50" -> [(100, 200), (200, 50)]
    
    Args:
        shapes_str: shape字符串，用冒号分隔多个shape，例如 "100,200:200,50"
    
    Returns:
        解析后的shape元组列表，如果解析失败返回空列表
    """
    if not shapes_str:
        return []
    
    shapes = []
    for shape_item in shapes_str.split(':'):
        shape_item = shape_item.strip()
        if not shape_item:
            continue
        
        dims = parse_shape_string(shape_item)
        if not dims:
            return []  # 如果任何一个shape解析失败，返回空列表
        
        shapes.append(tuple(dims))
    
    return shapes
