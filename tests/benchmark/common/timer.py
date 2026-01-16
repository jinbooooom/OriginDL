"""
Benchmark common timer utilities
"""

import time


class Timer:
    """
    高精度计时器
    """
    
    def __init__(self):
        self.start_time = None
    
    def start(self):
        """开始计时"""
        self.start_time = time.perf_counter()
    
    def elapsed_us(self) -> float:
        """
        返回经过的时间（微秒）
        
        Returns:
            经过的时间（微秒）
        """
        if self.start_time is None:
            return 0.0
        
        end_time = time.perf_counter()
        # time.perf_counter() 返回秒，转换为微秒
        elapsed_seconds = end_time - self.start_time
        return elapsed_seconds * 1000000.0  # 转换为微秒
