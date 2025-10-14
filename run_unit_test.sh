#!/bin/bash

# 单元测试运行脚本
# 使用ctest --verbose执行所有单元测试
# 支持不同后端：ORIGIN(默认)使用build目录，TORCH使用torch_build目录

set -e  # 遇到错误时退出

# 解析命令行参数
BACKEND="ORIGIN"  # 默认后端
QUIET_MODE=false

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            echo "Usage: $0 [BACKEND] [options]"
            echo "Backend options (case insensitive):"
            echo "  ORIGIN/origin  Use OriginMat backend (default, uses build directory)"
            echo "  TORCH/torch    Use LibTorch backend (uses torch_build directory)"
            echo "Other options:"
            echo "  -h, --help     Show this help message"
            echo "  -q, --quiet    Quiet mode (no verbose output)"
            echo ""
            echo "Examples:"
            echo "  $0             # Run tests with ORIGIN backend (default)"
            echo "  $0 ORIGIN      # Run tests with ORIGIN backend"
            echo "  $0 origin      # Run tests with ORIGIN backend (case insensitive)"
            echo "  $0 TORCH       # Run tests with TORCH backend"
            echo "  $0 torch       # Run tests with TORCH backend (case insensitive)"
            echo "  $0 TORCH -q    # Run tests with TORCH backend in quiet mode"
            echo "  $0 torch -q    # Run tests with TORCH backend in quiet mode"
            exit 0
            ;;
        -q|--quiet)
            QUIET_MODE=true
            shift
            ;;
        ORIGIN|origin|TORCH|torch)
            # 转换为大写
            BACKEND=$(echo $1 | tr '[:lower:]' '[:upper:]')
            shift
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use '$0 --help' to see help information"
            exit 1
            ;;
    esac
done

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的消息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查是否在正确的目录
check_directory() {
    if [ ! -f "CMakeLists.txt" ]; then
        print_error "Please run this script from the project root directory"
        exit 1
    fi
}

# 根据后端选择构建目录
get_build_directory() {
    if [ "$BACKEND" = "ORIGIN" ]; then
        echo "build"
    elif [ "$BACKEND" = "TORCH" ]; then
        echo "torch_build"
    else
        print_error "Invalid backend: $BACKEND. Supported backends: ORIGIN, TORCH"
        exit 1
    fi
}

# 检查build目录是否存在
check_build_directory() {
    BUILD_DIR=$(get_build_directory)
    if [ ! -d "$BUILD_DIR" ]; then
        print_error "$BUILD_DIR directory does not exist, please run 'bash build.sh $BACKEND' to build the project first"
        exit 1
    fi
}

# 检查ctest是否可用
check_ctest() {
    if ! command -v ctest &> /dev/null; then
        print_error "ctest command is not available, please ensure CMake is installed"
        exit 1
    fi
}

# 运行单元测试
run_tests() {
    BUILD_DIR=$(get_build_directory)
    
    if [ "$QUIET_MODE" = true ]; then
        # 静默模式
        cd $BUILD_DIR/tests/unit_test
        ctest --quiet
        exit $?
    else
        # 详细模式
        print_info "Starting unit tests for $BACKEND backend..."
        print_info "Using build directory: $BUILD_DIR"
        print_info "Running all tests with ctest --verbose"
        echo "----------------------------------------"
        
        # 进入单元测试目录
        cd $BUILD_DIR/tests/unit_test
        
        # 运行ctest
        if ctest --verbose; then
            print_success "All tests completed successfully"
        else
            print_warning "Some tests failed, please check the output above"
            exit 1
        fi
    fi
}

# 主函数
main() {
    print_info "OriginDL Unit Test Runner Script"
    print_info "Backend: $BACKEND"
    echo "========================================"
    
    # 检查环境
    check_directory
    check_build_directory
    check_ctest
    
    # 运行测试
    run_tests
}

# 运行主程序
main
