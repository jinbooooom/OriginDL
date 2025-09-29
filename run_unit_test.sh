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
        print_error "请在项目根目录下运行此脚本"
        exit 1
    fi
}

# 检查build目录是否存在
check_build_directory() {
    if [ ! -d "build" ]; then
        print_error "build目录不存在，请先运行 'bash build.sh' 编译项目"
        exit 1
    fi
}

# 检查ctest是否可用
check_ctest() {
    if ! command -v ctest &> /dev/null; then
        print_error "ctest命令不可用，请确保已安装CMake"
        exit 1
    fi
}

# 运行单元测试
run_tests() {
    print_info "开始运行单元测试..."
    print_info "使用ctest --verbose执行所有测试"
    echo "----------------------------------------"
    
    # 进入单元测试目录
    cd build/tests/unit_test
    
    # 运行ctest
    if ctest --verbose; then
        print_success "所有测试执行完成"
    else
        print_warning "部分测试失败，请查看上面的输出"
        exit 1
    fi
}

# 显示测试统计
show_statistics() {
    print_info "测试统计信息："
    echo "----------------------------------------"
    
    # 统计通过的测试
    local passed=$(ctest --verbose 2>&1 | grep -c "Passed" || echo "0")
    # 统计失败的测试
    local failed=$(ctest --verbose 2>&1 | grep -c "Failed" || echo "0")
    
    echo "通过的测试: $passed"
    echo "失败的测试: $failed"
    
    if [ "$failed" -eq 0 ]; then
        print_success "所有测试都通过了！"
    else
        print_warning "有 $failed 个测试失败"
    fi
}

# 主函数
main() {
    print_info "OriginDL 单元测试运行脚本"
    echo "========================================"
    
    # 检查环境
    check_directory
    check_build_directory
    check_ctest
    
    # 运行测试
    run_tests
    
    # 显示统计
    show_statistics
    
    print_info "脚本执行完成"
}

# 处理命令行参数
case "${1:-}" in
    -h|--help)
        echo "用法: $0 [选项]"
        echo "选项:"
        echo "  -h, --help     显示此帮助信息"
        echo "  -s, --stats    只显示测试统计信息"
        echo "  -q, --quiet    静默模式（不显示详细输出）"
        exit 0
        ;;
    -s|--stats)
        check_directory
        check_build_directory
        cd build/tests/unit_test
        show_statistics
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
        print_error "未知选项: $1"
        echo "使用 '$0 --help' 查看帮助信息"
        exit 1
        ;;
esac
