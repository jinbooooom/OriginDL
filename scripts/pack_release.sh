#!/bin/bash

# 通用发布资源打包脚本（模型、数据等）
# 用法: pack_release.sh -d DIR -o OUTPUT.tar.gz [--exclude PATTERN]... [-n]
# 示例: pack_release.sh -d model -o release/origindl-model-v1.0.0.tar.gz --exclude '.gitkeep'

set -e

INPUT_DIR=""
OUTPUT_PATH=""
DRY_RUN=false
EXCLUDE_PATTERNS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dir)
            if [[ -z "${2:-}" ]]; then
                echo "Error: -d/--dir requires a directory argument." >&2
                exit 1
            fi
            INPUT_DIR="$2"
            shift 2
            ;;
        -o|--output)
            if [[ -z "${2:-}" ]]; then
                echo "Error: -o/--output requires a path argument." >&2
                exit 1
            fi
            OUTPUT_PATH="$2"
            shift 2
            ;;
        --exclude)
            if [[ -z "${2:-}" ]]; then
                echo "Error: --exclude requires a pattern argument." >&2
                exit 1
            fi
            EXCLUDE_PATTERNS+=("$2")
            shift 2
            ;;
        -n|--dry-run)
            DRY_RUN=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 -d DIR -o OUTPUT.tar.gz [OPTIONS]"
            echo "Pack a directory into a .tar.gz archive for release."
            echo ""
            echo "Required:"
            echo "  -d, --dir DIR       Directory to pack (e.g. model, data)"
            echo "  -o, --output PATH   Output archive path (e.g. release/origindl-model-v1.0.0.tar.gz)"
            echo ""
            echo "Options:"
            echo "  --exclude PATTERN   Exclude pattern for tar (repeatable, e.g. '.gitkeep' or 'outputs/*')"
            echo "  -n, --dry-run       Print tar command only, do not run"
            echo "  -h, --help          Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 -d model -o release/origindl-model-v1.0.0.tar.gz --exclude '.gitkeep'"
            echo "  $0 -d data -o release/origindl-data-v1.0.0.tar.gz --exclude 'outputs/*' --exclude '.gitkeep' -n"
            exit 0
            ;;
        *)
            echo "Error: Unknown option: $1. Use -h or --help for usage." >&2
            exit 1
            ;;
    esac
done

if [[ -z "$INPUT_DIR" ]]; then
    echo "Error: -d/--dir is required." >&2
    exit 1
fi
if [[ -z "$OUTPUT_PATH" ]]; then
    echo "Error: -o/--output is required." >&2
    exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
    echo "Error: Input directory does not exist: $INPUT_DIR" >&2
    exit 1
fi

# 输出路径：父目录存在或创建
OUTPUT_PARENT=$(dirname "$OUTPUT_PATH")
if [[ -n "$OUTPUT_PARENT" ]] && [[ "$OUTPUT_PARENT" != "." ]]; then
    mkdir -p "$OUTPUT_PARENT"
fi

# 构建 tar 命令：-C 父目录 + 目录名，保证解压后得到单层目录
INPUT_PARENT=$(cd -P "$(dirname "$INPUT_DIR")" && pwd)
INPUT_BASE=$(basename "$INPUT_DIR")
TAR_EXCLUDE=()
for p in "${EXCLUDE_PATTERNS[@]}"; do
    TAR_EXCLUDE+=(--exclude="$p")
done

if $DRY_RUN; then
    echo "Would run: tar -czf $OUTPUT_PATH ${TAR_EXCLUDE[*]} -C $INPUT_PARENT $INPUT_BASE"
    exit 0
fi

tar -czf "$OUTPUT_PATH" "${TAR_EXCLUDE[@]}" -C "$INPUT_PARENT" "$INPUT_BASE"
echo "Packed: $INPUT_DIR -> $OUTPUT_PATH"
ls -lh "$OUTPUT_PATH"
