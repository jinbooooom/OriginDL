#include <cmath>
#include <memory>
#include "origin/mat/basic_types.h"
#include "origin/mat/origin/device_common/type_dispatcher.h"
#include "origin/mat/origin/origin_mat.h"
#include "origin/mat/scalar.h"
#include "origin/utils/exception.h"

namespace origin
{
namespace cpu
{

std::unique_ptr<OriginMat> pow(const OriginMat &mat, const Scalar &exponent)
{
    auto result = std::make_unique<OriginMat>(mat.shape(), mat.dtype());

    const void *a_data = mat.storage()->data();
    void *c_data       = result->storage()->data();

    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        const T *input_data = static_cast<const T *>(a_data);
        T *output_data      = static_cast<T *>(c_data);

        if (exponent.dtype() == DataType::kFloat64)
        {
            double exp_value = exponent.to_float64();
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                output_data[i] = static_cast<T>(std::pow(static_cast<double>(input_data[i]), exp_value));
            }
        }
        else
        {
            float exp_value = exponent.to_float32();
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                output_data[i] = static_cast<T>(powf(static_cast<float>(input_data[i]), exp_value));
            }
        }
    });

    return result;
}

void pow_inplace(OriginMat &mat, const Scalar &exponent)
{
    void *data = mat.storage()->data();

    device_common::TypeDispatcher::dispatch_void(mat.dtype(), [&]<typename T>() {
        T *data_ptr = static_cast<T *>(data);

        if (exponent.dtype() == DataType::kFloat64)
        {
            double exp_value = exponent.to_float64();
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                data_ptr[i] = static_cast<T>(std::pow(static_cast<double>(data_ptr[i]), exp_value));
            }
        }
        else
        {
            float exp_value = exponent.to_float32();
            for (size_t i = 0; i < mat.elements(); ++i)
            {
                data_ptr[i] = static_cast<T>(powf(static_cast<float>(data_ptr[i]), exp_value));
            }
        }
    });
}

}  // namespace cpu
}  // namespace origin
