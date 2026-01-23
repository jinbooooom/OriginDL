#include "origin/mat/mat.h"
#include "origin/mat/backend.h"
#include "origin/mat/basic_types.h"
#include "origin/utils/exception.h"

#ifdef MAT_BACKEND
#    if MAT_BACKEND == 0  // ORIGIN
#        include "origin/mat/origin/origin_mat.h"
#        ifdef WITH_CUDA
#            include "origin/mat/origin/cuda/cuda_ops.cuh"
#        endif
#        include "origin/mat/origin/cpu/cpu_ops.h"
#    endif
#endif

namespace origin
{

std::unique_ptr<Mat> Mat::cat(const std::vector<const Mat *> &inputs, int dim)
{
    if (inputs.empty())
    {
        THROW_RUNTIME_ERROR("Mat::cat: requires at least 1 input");
    }

    if (inputs.size() == 1)
    {
        // 只有一个输入，直接复制
        return inputs[0]->clone();
    }

    // 检查所有输入的后端类型是否相同
    int backend_type = inputs[0]->backend_type();
    for (size_t i = 1; i < inputs.size(); ++i)
    {
        if (inputs[i]->backend_type() != backend_type)
        {
            THROW_RUNTIME_ERROR("Mat::cat: all inputs must have same backend type, got {} and {}",
                               backend_type, inputs[i]->backend_type());
        }
    }

    // 根据后端类型分发
#ifdef MAT_BACKEND
#    if MAT_BACKEND == 0  // ORIGIN
    if (backend_type == ORIGIN_BACKEND_TYPE)  // ORIGIN (0)
    {
        // 转换为 OriginMat 指针
        std::vector<const OriginMat *> origin_inputs;
        origin_inputs.reserve(inputs.size());
        for (const auto *input : inputs)
        {
            const OriginMat *origin_mat = dynamic_cast<const OriginMat *>(input);
            if (!origin_mat)
            {
                THROW_RUNTIME_ERROR("Mat::cat: failed to cast to OriginMat");
            }
            origin_inputs.push_back(origin_mat);
        }

        // 根据设备类型调用对应的实现
        DeviceType device_type = inputs[0]->device().type();
        if (device_type == DeviceType::kCUDA)
        {
#            ifdef WITH_CUDA
            return cuda::cat(origin_inputs, dim);
#            else
            THROW_RUNTIME_ERROR("CUDA support not compiled in");
#            endif
        }
        else
        {
            return cpu::cat(origin_inputs, dim);
        }
    }
#    endif
#endif

    if (backend_type == TORCH_BACKEND_TYPE)  // TORCH (1)
    {
        THROW_RUNTIME_ERROR("Mat::cat: TorchMat backend not yet implemented");
    }
    else
    {
        THROW_RUNTIME_ERROR("Mat::cat: unsupported backend type: {}", backend_type);
    }
}

std::vector<std::unique_ptr<Mat>> Mat::split(const Mat &input, const std::vector<Shape> &output_shapes, int dim)
{
    int backend_type = input.backend_type();

    // 根据后端类型分发
#ifdef MAT_BACKEND
#    if MAT_BACKEND == 0  // ORIGIN
    if (backend_type == ORIGIN_BACKEND_TYPE)  // ORIGIN (0)
    {
        const OriginMat *origin_mat = dynamic_cast<const OriginMat *>(&input);
        if (!origin_mat)
        {
            THROW_RUNTIME_ERROR("Mat::split: failed to cast to OriginMat");
        }

        // 根据设备类型调用对应的实现
        DeviceType device_type = input.device().type();
        if (device_type == DeviceType::kCUDA)
        {
#            ifdef WITH_CUDA
            return cuda::split(*origin_mat, output_shapes, dim);
#            else
            THROW_RUNTIME_ERROR("CUDA support not compiled in");
#            endif
        }
        else
        {
            return cpu::split(*origin_mat, output_shapes, dim);
        }
    }
#    endif
#endif

    if (backend_type == TORCH_BACKEND_TYPE)  // TORCH (1)
    {
        THROW_RUNTIME_ERROR("Mat::split: TorchMat backend not yet implemented");
    }
    else
    {
        THROW_RUNTIME_ERROR("Mat::split: unsupported backend type: {}", backend_type);
    }
}

}  // namespace origin
