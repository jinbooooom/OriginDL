#!/bin/bash

# 一键测量所有矩阵乘法算法版本 (ORIGIN_KERNEL_ALGO=0~8) 的性能
# 用法: bench_matmul_algos.sh [-w WARMUP] [-r REPEAT] [-s SHAPE]
# 示例: bench_matmul_algos.sh -w 2 -r 10
# 示例: bench_matmul_algos.sh -w 2 -r 10 -s "100,100:100,100"
# 默认: -w 2 -r 10

set -e

WARMUP=2
REPEAT=10
SHAPE_ARG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -w)
            if [[ -z "${2:-}" ]]; then
                echo "Error: -w requires a number." >&2
                exit 1
            fi
            WARMUP="$2"
            shift 2
            ;;
        -r)
            if [[ -z "${2:-}" ]]; then
                echo "Error: -r requires a number." >&2
                exit 1
            fi
            REPEAT="$2"
            shift 2
            ;;
        -s)
            if [[ -z "${2:-}" ]]; then
                echo "Error: -s requires a value." >&2
                exit 1
            fi
            SHAPE_ARG="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [-w WARMUP] [-r REPEAT] [-s SHAPE]"
            echo "Run matmul benchmark for ORIGIN_KERNEL_ALGO=0..9 with python3 run_benchmark.py."
            echo ""
            echo "Options:"
            echo "  -w WARMUP   Warmup rounds (default: 2)"
            echo "  -r REPEAT  Repeat rounds (default: 10)"
            echo "  -s SHAPE   Shape argument to pass to run_benchmark.py (optional)"
            echo "  -h, --help Show this help"
            echo ""
            echo "Example: $0 -w 2 -r 10"
            echo "Example: $0 -w 2 -r 10 -s \"100,100:100,100\""
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1. Use -h or --help." >&2
            exit 1
            ;;
    esac
done

# 项目根目录：脚本在 scripts/ 下，根目录为上级
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
BENCHMARK_PY="$PROJECT_ROOT/run_benchmark.py"

if [[ ! -f "$BENCHMARK_PY" ]]; then
    echo "Error: run_benchmark.py not found at $BENCHMARK_PY" >&2
    exit 1
fi

cd "$PROJECT_ROOT"

echo "========================================================================"
echo "Matmul all algos benchmark (ORIGIN_KERNEL_ALGO=0..9)"
echo "  -w $WARMUP -r $REPEAT"
if [[ -n "$SHAPE_ARG" ]]; then
    echo "  -s $SHAPE_ARG"
    echo "  run_benchmark.py -d cuda -f matmul -w $WARMUP -r $REPEAT -s $SHAPE_ARG"
else
    echo "  run_benchmark.py -d cuda -f matmul -w $WARMUP -r $REPEAT"
fi
echo "========================================================================"

for algo in 0 1 2 3 4 5 6 7 8; do
    echo ""
    echo ">>>>>>>>>>>>>>>>>>>>>>>>>> V${algo} <<<<<<<<<<<<<<<<<<<<<<<<<<"
    export ORIGIN_KERNEL_ALGO=$algo
    if [[ -n "$SHAPE_ARG" ]]; then
        python3 run_benchmark.py -d cuda -f matmul -w "$WARMUP" -r "$REPEAT" -s $SHAPE_ARG
    else
        python3 run_benchmark.py -d cuda -f matmul -w "$WARMUP" -r "$REPEAT"
    fi
done

echo ""
echo "========================================================================"
echo "Done. All algos 0..8 benchmarked."
echo "========================================================================"
