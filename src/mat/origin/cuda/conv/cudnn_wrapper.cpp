#ifdef HAVE_CUDNN

#include "origin/mat/origin/cuda/cudnn_wrapper.h"
#include "origin/utils/exception.h"
#include "origin/utils/log.h"
#include <cudnn.h>
#include <cuda_runtime.h>
#include <chrono>

// cuDNN 错误检查宏
#define CUDNN_CHECK(expr) \
    do { \
        cudnnStatus_t status = (expr); \
        if (status != CUDNN_STATUS_SUCCESS) { \
            THROW_RUNTIME_ERROR("cuDNN error at {}:{}: {}", __FILE__, __LINE__, static_cast<int>(status)); \
        } \
    } while (0)

// CUDA 错误检查宏
#define CUDA_CHECK(expr) \
    do { \
        cudaError_t err = (expr); \
        if (err != cudaSuccess) { \
            THROW_RUNTIME_ERROR("CUDA error at {}:{}: {}", __FILE__, __LINE__, cudaGetErrorString(err)); \
        } \
    } while (0)

namespace origin
{
namespace cuda
{

struct CudnnWrapper::ConvCache
{
    cudnnTensorDescriptor_t x_desc = nullptr;
    cudnnFilterDescriptor_t w_desc = nullptr;
    cudnnTensorDescriptor_t y_desc = nullptr;
    cudnnTensorDescriptor_t b_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    cudnnConvolutionFwdAlgo_t algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;  // 默认算法
    bool algo_initialized = false;  // 标记算法是否已初始化
    size_t workspace_size = 0;
    void *workspace = nullptr;
    
    // 缓存的参数（用于判断是否需要重新查找算法）
    int cached_N = 0, cached_C = 0, cached_H = 0, cached_W = 0;
    int cached_OC = 0, cached_KH = 0, cached_KW = 0;
    int cached_SH = 0, cached_SW = 0, cached_PH = 0, cached_PW = 0;
    
    ~ConvCache()
    {
        if (x_desc) cudnnDestroyTensorDescriptor(x_desc);
        if (w_desc) cudnnDestroyFilterDescriptor(w_desc);
        if (y_desc) cudnnDestroyTensorDescriptor(y_desc);
        if (b_desc) cudnnDestroyTensorDescriptor(b_desc);
        if (conv_desc) cudnnDestroyConvolutionDescriptor(conv_desc);
        if (workspace) cudaFree(workspace);
    }
};

CudnnWrapper::CudnnWrapper() : cache_(std::make_unique<ConvCache>())
{
    cudnnStatus_t status = cudnnCreate(&handle_);
    if (status != CUDNN_STATUS_SUCCESS)
    {
        THROW_RUNTIME_ERROR("Failed to create cuDNN handle: {}", static_cast<int>(status));
    }
    // 使用 NULL stream 以获得最佳性能（cuDNN 默认就是 NULL stream，但显式设置更安全）
    CUDNN_CHECK(cudnnSetStream(handle_, nullptr));
}

CudnnWrapper::~CudnnWrapper()
{
    if (handle_ != nullptr)
    {
        cudnnDestroy(handle_);
    }
}

CudnnWrapper &CudnnWrapper::get_instance()
{
    static CudnnWrapper instance;
    return instance;
}

void CudnnWrapper::conv2d_forward(const void *x, const void *W, const void *b,
                                  int N, int C, int H, int W_in,
                                  int OC, int KH, int KW,
                                  int SH, int SW, int PH, int PW,
                                  void *result, int OH, int OW,
                                  cudnnDataType_t data_type)
{
    auto &cache = *cache_;
    
    // 创建或更新张量描述符（只在第一次创建）
    if (cache.x_desc == nullptr)
    {
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&cache.x_desc));
        CUDNN_CHECK(cudnnCreateFilterDescriptor(&cache.w_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&cache.y_desc));
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&cache.b_desc));
        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&cache.conv_desc));
    }
    
    // 检查参数是否变化（用于决定是否需要重新设置描述符）
    bool params_changed = !cache.algo_initialized ||
                          cache.cached_N != N || cache.cached_C != C || 
                          cache.cached_H != H || cache.cached_W != W_in ||
                          cache.cached_OC != OC || cache.cached_KH != KH || cache.cached_KW != KW ||
                          cache.cached_SH != SH || cache.cached_SW != SW ||
                          cache.cached_PH != PH || cache.cached_PW != PW;
    
    // 只在参数变化时重新设置描述符（避免每次都设置的开销）
    if (params_changed)
    {
        // 设置输入张量描述符 (N, C, H, W)
        // 注意：origindl 使用行主序存储，strides = {C*H*W, H*W, W, 1}
        // cuDNN 默认使用列主序，但我们可以使用 cudnnSetTensor4dDescriptorEx 手动指定 strides
        // 对于行主序数据，strides = {C*H*W, H*W, W, 1}
        int x_strides[4] = {static_cast<int>(C * H * W_in), static_cast<int>(H * W_in), static_cast<int>(W_in), 1};
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(cache.x_desc,
                                                  data_type,
                                                  N, C, H, W_in,
                                                  x_strides[0], x_strides[1], x_strides[2], x_strides[3]));
        
        // 设置卷积核描述符 (OC, C, KH, KW)
        // 权重也是行主序，strides = {C*KH*KW, KH*KW, KW, 1}
        int w_strides[4] = {static_cast<int>(C * KH * KW), static_cast<int>(KH * KW), static_cast<int>(KW), 1};
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(cache.w_desc,
                                               data_type,
                                               CUDNN_TENSOR_NCHW,
                                               OC, C, KH, KW));
        
        // 设置卷积描述符
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(cache.conv_desc,
                                                     PH, PW,  // pad_h, pad_w
                                                     SH, SW,  // stride_h, stride_w
                                                     1, 1,    // dilation_h, dilation_w
                                                     CUDNN_CONVOLUTION,
                                                     data_type));
        
        // 设置输出张量描述符 (N, OC, OH, OW)
        // 输出也是行主序，strides = {OC*OH*OW, OH*OW, OW, 1}
        int y_strides[4] = {static_cast<int>(OC * OH * OW), static_cast<int>(OH * OW), static_cast<int>(OW), 1};
        CUDNN_CHECK(cudnnSetTensor4dDescriptorEx(cache.y_desc,
                                                  data_type,
                                                  N, OC, OH, OW,
                                                  y_strides[0], y_strides[1], y_strides[2], y_strides[3]));
    }
    
    // 查找最佳算法（使用快速方法，避免每次都测试所有算法）
    // 注意：cudnnGetConvolutionForwardAlgorithm_v7 会测试所有算法，非常耗时（可能需要几秒到几十秒）
    // 我们只在第一次调用或参数变化时查找算法，其他时候直接使用缓存的算法
    // params_changed 已经在上面定义过了
    
    if (params_changed)
    {
        // 参数变化，直接使用快速算法，避免算法查找的耗时
        // 对于大多数卷积，IMPLICIT_GEMM 已经足够快
        // 注意：如果使用 cudnnGetConvolutionForwardAlgorithm_v7 会测试所有算法，非常耗时（可能几十秒）
        cache.algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
        
        // 只在第一次或参数变化时记录日志
        static int call_count = 0;
        call_count++;
        if (call_count <= 5) {  // 只记录前5次，避免日志过多
            logi("cuDNN: Using IMPLICIT_GEMM algorithm for conv (N={}, C={}, H={}, W={}, OC={}, K={}x{}, S={}x{}, P={}x{})",
                 N, C, H, W_in, OC, KH, KW, SH, SW, PH, PW);
        }
        
        // 缓存参数
        cache.cached_N = N;
        cache.cached_C = C;
        cache.cached_H = H;
        cache.cached_W = W_in;
        cache.cached_OC = OC;
        cache.cached_KH = KH;
        cache.cached_KW = KW;
        cache.cached_SH = SH;
        cache.cached_SW = SW;
        cache.cached_PH = PH;
        cache.cached_PW = PW;
        cache.algo_initialized = true;
    }
    // 如果参数没变化，直接使用缓存的算法，避免重复查找
    
    // 获取工作空间大小（只在参数变化时重新获取，避免重复调用导致的性能问题）
    size_t workspace_size_bytes = 0;
    if (params_changed || cache.workspace_size == 0)
    {
        // 只在参数变化时重新获取 workspace size
        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(handle_,
                                                             cache.x_desc,
                                                             cache.w_desc,
                                                             cache.conv_desc,
                                                             cache.y_desc,
                                                             cache.algo,
                                                             &workspace_size_bytes));
        
        // 分配工作空间（如果需要）
        if (workspace_size_bytes > cache.workspace_size)
        {
            if (cache.workspace)
            {
                cudaFree(cache.workspace);
            }
            CUDA_CHECK(cudaMalloc(&cache.workspace, workspace_size_bytes));
            cache.workspace_size = workspace_size_bytes;
        }
    }
    else
    {
        // 参数没变化，使用缓存的 workspace_size
        workspace_size_bytes = cache.workspace_size;
    }
    
    // 执行卷积
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // 记录卷积开始时间（用于调试）
    auto conv_start = std::chrono::high_resolution_clock::now();
    
    // 确保使用正确的 CUDA stream（使用 NULL stream 以获得最佳性能）
    // 注意：cuDNN 默认使用 NULL stream，这通常是最快的
    // 如果需要异步执行，可以设置 stream，但这里我们使用默认的 NULL stream
    CUDNN_CHECK(cudnnConvolutionForward(handle_,
                                        &alpha,
                                        cache.x_desc, x,
                                        cache.w_desc, W,
                                        cache.conv_desc,
                                        cache.algo,
                                        cache.workspace, workspace_size_bytes,
                                        &beta,
                                        cache.y_desc, result));
    
    // 记录卷积结束时间
    auto conv_end = std::chrono::high_resolution_clock::now();
    auto conv_duration = std::chrono::duration_cast<std::chrono::milliseconds>(conv_end - conv_start);
    
    // 对于慢的卷积，记录详细信息
    if (conv_duration.count() > 50) {
        logw("cuDNN conv2d_forward took {}ms (N={}, C={}, H={}, W={}, OC={}, K={}x{}, S={}x{}, P={}x{}, algo={})",
             conv_duration.count(), N, C, H, W_in, OC, KH, KW, SH, SW, PH, PW, static_cast<int>(cache.algo));
    }
    
    // 注意：cuDNN 操作默认是异步的，但某些操作可能会隐式同步
    // 这里我们不显式同步，让调用者决定何时同步
    
    // 添加偏置（如果存在）
    if (b != nullptr)
    {
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(cache.b_desc,
                                              CUDNN_TENSOR_NCHW,
                                              data_type,
                                              1, OC, 1, 1));
        
        CUDNN_CHECK(cudnnAddTensor(handle_,
                                   &alpha,
                                   cache.b_desc, b,
                                   &alpha,
                                   cache.y_desc, result));
    }
}

}  // namespace cuda
}  // namespace origin

#else  // HAVE_CUDNN not defined

// cuDNN 不可用时的空实现
#include "origin/mat/origin/cuda/cudnn_wrapper.h"

namespace origin
{
namespace cuda
{

CudnnWrapper &CudnnWrapper::get_instance()
{
    static CudnnWrapper instance;
    return instance;
}

}  // namespace cuda
}  // namespace origin

#endif  // HAVE_CUDNN
