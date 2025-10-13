#!/bin/bash

# 单元测试运行脚本
# 使用ctest --verbose执行所有单元测试

set -e  # 遇到错误时退出

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

# 检查build目录是否存在
check_build_directory() {
    if [ ! -d "build" ]; then
        print_error "build directory does not exist, please run 'bash build.sh' to build the project first"
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
    print_info "Starting unit tests..."
    print_info "Running all tests with ctest --verbose"
    echo "----------------------------------------"
    
    # 进入单元测试目录
    cd build/tests/unit_test
    
    # 运行ctest
    if ctest --verbose; then
        print_success "All tests completed successfully"
    else
        print_warning "Some tests failed, please check the output above"
        exit 1
    fi
}

# 主函数
main() {
    print_info "OriginDL Unit Test Runner Script"
    echo "========================================"
    
    # 检查环境
    check_directory
    check_build_directory
    check_ctest
    
    # 运行测试
    run_tests
}

# 处理命令行参数
case "${1:-}" in
    -h|--help)
        echo "Usage: $0 [options]"
        echo "Options:"
        echo "  -h, --help     Show this help message"
        echo "  -q, --quiet    Quiet mode (no verbose output)"
        exit 0
        ;;
    -q|--quiet)
        check_directory
        check_build_directory
        cd build/tests/unit_test
        ctest --quiet
        exit $?
        ;;
    "")
        main
        ;;
    *)
        print_error "Unknown option: $1"
        echo "Use '$0 --help' to see help information"
        exit 1
        ;;
esac
