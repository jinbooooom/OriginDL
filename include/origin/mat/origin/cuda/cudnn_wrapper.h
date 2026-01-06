#ifndef ORIGIN_CUDNN_WRAPPER_H
#define ORIGIN_CUDNN_WRAPPER_H

#ifdef HAVE_CUDNN
#include <cudnn.h>
#include <memory>

namespace origin
{
namespace cuda
{

/**
 * @brief cuDNN 包装器类，管理 cuDNN handle 和卷积操作
 */
class CudnnWrapper
{
public:
    static CudnnWrapper &get_instance();
    
    cudnnHandle_t get_handle() const { return handle_; }
    
    /**
     * @brief 执行前向卷积
     * @param x 输入张量 (N, C, H, W)
     * @param W 卷积核 (OC, C, KH, KW)
     * @param b 偏置 (OC,)，可为 nullptr
     * @param stride 步长 (SH, SW)
     * @param pad 填充 (PH, PW)
     * @param result 输出张量 (N, OC, OH, OW)
     */
    void conv2d_forward(const void *x, const void *W, const void *b,
                       int N, int C, int H, int W_in,
                       int OC, int KH, int KW,
                       int SH, int SW, int PH, int PW,
                       void *result, int OH, int OW,
                       cudnnDataType_t data_type);

private:
    CudnnWrapper();
    ~CudnnWrapper();
    
    CudnnWrapper(const CudnnWrapper &) = delete;
    CudnnWrapper &operator=(const CudnnWrapper &) = delete;
    
    cudnnHandle_t handle_;
    
    // 缓存卷积描述符和算法，避免重复创建
    struct ConvCache;
    std::unique_ptr<ConvCache> cache_;
};

}  // namespace cuda
}  // namespace origin

#else  // HAVE_CUDNN not defined

// cuDNN 不可用时的空实现
namespace origin
{
namespace cuda
{

class CudnnWrapper
{
public:
    static CudnnWrapper &get_instance();
    void *get_handle() const { return nullptr; }
    void conv2d_forward(...) {}
};

}  // namespace cuda
}  // namespace origin

#endif  // HAVE_CUDNN

#endif  // ORIGIN_CUDNN_WRAPPER_H

