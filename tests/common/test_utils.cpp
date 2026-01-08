#include "test_utils.h"

#ifdef WITH_CUDA
#    include <cuda_runtime.h>
#    include "origin/mat/origin/cuda/cuda_utils.cuh"
#endif

namespace origin
{
namespace test
{

bool TestUtils::isEqual(double a, double b, double tolerance)
{
    return std::abs(a - b) < tolerance;
}

bool TestUtils::tensorsEqual(const Tensor &a, const Tensor &b, double tolerance)
{
    if (a.shape() != b.shape())
    {
        return false;
    }

    auto data_a = a.to_vector<float>();
    auto data_b = b.to_vector<float>();

    if (data_a.size() != data_b.size())
    {
        return false;
    }

    for (size_t i = 0; i < data_a.size(); ++i)
    {
        if (!isEqual(data_a[i], data_b[i], tolerance))
        {
            return false;
        }
    }
    return true;
}

bool TestUtils::tensorsNear(const Tensor &a, const Tensor &b, double rtol, double atol)
{
    if (a.shape() != b.shape())
    {
        return false;
    }

    auto data_a = a.to_vector<float>();
    auto data_b = b.to_vector<float>();

    if (data_a.size() != data_b.size())
    {
        return false;
    }

    for (size_t i = 0; i < data_a.size(); ++i)
    {
        double diff    = std::abs(data_a[i] - data_b[i]);
        double max_val = std::max(std::abs(data_a[i]), std::abs(data_b[i]));
        if (diff > atol + rtol * max_val)
        {
            return false;
        }
    }
    return true;
}

bool TestUtils::isCudaAvailable()
{
#ifdef WITH_CUDA
    return cuda::is_cuda_available();
#else
    return false;
#endif
}

}  // namespace test
}  // namespace origin
