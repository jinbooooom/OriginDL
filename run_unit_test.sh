#!/bin/bash

# 单元测试运行脚本
# 使用ctest --verbose执行所有单元测试
# 支持不同后端：ORIGIN(默认)使用build目录，TORCH使用torch_build目录

set -e  # 遇到错误时退出

# 解析命令行参数
BACKEND="ORIGIN"  # 默认后端
QUIET_MODE=false
CUDA_TESTS=false

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
            echo "  --cuda         Run CUDA unit tests (requires CUDA support)"
            echo ""
            echo "Examples:"
            echo "  $0             # Run tests with ORIGIN backend (default)"
            echo "  $0 ORIGIN      # Run tests with ORIGIN backend"
            echo "  $0 origin      # Run tests with ORIGIN backend (case insensitive)"
            echo "  $0 TORCH       # Run tests with TORCH backend"
            echo "  $0 torch       # Run tests with TORCH backend (case insensitive)"
            echo "  $0 TORCH -q    # Run tests with TORCH backend in quiet mode"
            echo "  $0 torch -q    # Run tests with TORCH backend in quiet mode"
            echo "  $0 --cuda      # Run CUDA unit tests"
            echo "  $0 ORIGIN --cuda # Run tests with ORIGIN backend and CUDA tests"
            exit 0
            ;;
        -q|--quiet)
            QUIET_MODE=true
            shift
            ;;
        --cuda)
            CUDA_TESTS=true
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

# 检查CUDA测试目录是否存在
check_cuda_tests() {
    BUILD_DIR=$(get_build_directory)
    CUDA_TEST_DIR="$BUILD_DIR/tests/unit_test_cuda"
    
    if [ ! -d "$CUDA_TEST_DIR" ]; then
        print_error "CUDA test directory does not exist: $CUDA_TEST_DIR"
        print_error "Please ensure the project was built with CUDA support: bash build.sh $BACKEND --cuda"
        exit 1
    fi
    
    # 检查是否有CUDA测试可执行文件
    if [ ! -d "$BUILD_DIR/bin/unit_test_cuda" ]; then
        print_error "CUDA test executables not found in $BUILD_DIR/bin/unit_test_cuda"
        print_error "Please ensure CUDA tests were compiled successfully"
        exit 1
    fi
}

# 检查CUDA是否可用（用于判断测试是否包含CUDA版本）
check_cuda_available() {
    # 检查是否有CUDA设备可用
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            return 0  # CUDA可用
        fi
    fi
    return 1  # CUDA不可用
}

# 运行单元测试
run_tests() {
    BUILD_DIR=$(get_build_directory)
    TEST_FAILED=false
    HAS_CUDA_TESTS=false
    
    # 检查CUDA是否可用（如果可用，参数化测试会包含CUDA版本）
    if check_cuda_available; then
        HAS_CUDA_TESTS=true
    fi
    
        # 如果请求CUDA测试，创建统一的测试环境
    if [ "$CUDA_TESTS" = true ]; then
        print_info "Starting unified CPU and CUDA unit tests for $BACKEND backend..."
        print_info "Using build directory: $BUILD_DIR"
        print_info "Running all tests with ctest --verbose"
        echo "----------------------------------------"

        # 创建临时统一测试目录
        UNIFIED_TEST_DIR="$BUILD_DIR/tests/unified_test"
        mkdir -p "$UNIFIED_TEST_DIR"
        
        # 复制CPU测试配置
        cp "$BUILD_DIR/tests/unit_test/CTestTestfile.cmake" "$UNIFIED_TEST_DIR/"
        
        # 如果CUDA测试存在，合并CUDA测试配置
        if [ -f "$BUILD_DIR/tests/unit_test_cuda/CTestTestfile.cmake" ]; then
            # 读取CUDA测试配置并追加到统一配置中
            grep -v "^# This file was generated by CTest" "$BUILD_DIR/tests/unit_test_cuda/CTestTestfile.cmake" >> "$UNIFIED_TEST_DIR/CTestTestfile.cmake"
        fi
        
        # 进入统一测试目录
        cd "$UNIFIED_TEST_DIR"

        # 运行所有测试（CPU + CUDA）
        if [ "$QUIET_MODE" = true ]; then
            if ! ctest --quiet; then
                TEST_FAILED=true
            fi
        else
            if ! ctest --verbose; then
                TEST_FAILED=true
            fi
        fi
        cd - > /dev/null  # 返回原目录
        
        # 清理临时目录
        rm -rf "$UNIFIED_TEST_DIR"
    else
        # 运行单元测试（可能包含CPU和CUDA参数化测试）
        if [ "$QUIET_MODE" = true ]; then
            # 静默模式
            print_info "Running unit tests..."
            cd $BUILD_DIR/tests/unit_test
            if ! ctest --quiet; then
                TEST_FAILED=true
            fi
            cd - > /dev/null  # 返回原目录
        else
            # 详细模式
            print_info "Starting unit tests for $BACKEND backend..."
            print_info "Using build directory: $BUILD_DIR"
            print_info "Running tests with ctest --verbose"
            echo "----------------------------------------"

            # 进入单元测试目录
            cd $BUILD_DIR/tests/unit_test

            # 运行ctest
            if ! ctest --verbose; then
                print_warning "Some tests failed"
                TEST_FAILED=true
            fi
            cd - > /dev/null  # 返回原目录
        fi
    fi
    
    # 检查总体结果
    if [ "$TEST_FAILED" = true ]; then
        print_warning "Some tests failed, please check the output above"
        exit 1
    else
        if [ "$HAS_CUDA_TESTS" = true ] || [ "$CUDA_TESTS" = true ]; then
            print_success "All tests completed successfully (CPU and CUDA)"
        else
            print_success "All tests completed successfully"
        fi
    fi
}

# 主函数
main() {
    print_info "OriginDL Unit Test Runner Script"
    print_info "Backend: $BACKEND"
    if [ "$CUDA_TESTS" = true ]; then
        print_info "CUDA Tests: Enabled"
    else
        print_info "CUDA Tests: Disabled"
    fi
    echo "========================================"
    
    # 检查环境
    check_directory
    check_build_directory
    check_ctest
    
    # 如果请求CUDA测试，检查CUDA测试环境
    if [ "$CUDA_TESTS" = true ]; then
        check_cuda_tests
    fi
    
    # 运行测试
    run_tests
}

# 运行主程序
main
