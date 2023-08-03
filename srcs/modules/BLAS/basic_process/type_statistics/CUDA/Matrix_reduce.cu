/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "Matrix_reduce_sum.cuh"
#include "Matrix_reduce_cmp.cuh"



_DECX_API_ de::DH de::cuda::Sum(de::Matrix& src, double* res)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);

    if (_src->Type() == decx::_DATA_TYPES_FLAGS_::_FP32_) {
        float res_fp32 = 0;
        decx::reduce::matrix_reduce2D_full_sum_fp32(_src, &res_fp32);
        *res = res_fp32;
    }

    return handle;
}




_DECX_API_ de::DH de::cuda::Max(de::Matrix& src, double* res)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);

    if (_src->Type() == decx::_DATA_TYPES_FLAGS_::_FP32_) {
        float res_fp32 = 0;
        decx::reduce::matrix_reduce2D_full_cmp_fp32<true>(_src, &res_fp32);
        *res = res_fp32;
    }

    return handle;
}




_DECX_API_ de::DH de::cuda::Min(de::Matrix& src, double* res)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);

    if (_src->Type() == decx::_DATA_TYPES_FLAGS_::_FP32_) {
        float res_fp32 = 0;
        decx::reduce::matrix_reduce2D_full_cmp_fp32<false>(_src, &res_fp32);
        *res = res_fp32;
    }

    return handle;
}



_DECX_API_ de::DH de::cuda::Sum(de::GPU_Matrix& src, double* res)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);

    if (_src->Type() == decx::_DATA_TYPES_FLAGS_::_FP32_) {
        float res_fp32 = 0;
        decx::reduce::dev_matrix_reduce2D_full_sum_fp32(_src, &res_fp32);
        *res = res_fp32;
    }

    return handle;
}




_DECX_API_ de::DH de::cuda::Max(de::GPU_Matrix& src, double* res)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);

    if (_src->Type() == decx::_DATA_TYPES_FLAGS_::_FP32_) {
        float res_fp32 = 0;
        decx::reduce::dev_matrix_reduce2D_full_cmp_fp32<true>(_src, &res_fp32);
        *res = res_fp32;
    }

    return handle;
}


_DECX_API_ de::DH de::cuda::Min(de::GPU_Matrix& src, double* res)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);

    if (_src->Type() == decx::_DATA_TYPES_FLAGS_::_FP32_) {
        float res_fp32 = 0;
        decx::reduce::dev_matrix_reduce2D_full_cmp_fp32<false>(_src, &res_fp32);
        *res = res_fp32;
    }

    return handle;
}



_DECX_API_ de::DH de::cuda::Sum(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (_reduce2D_mode == de::REDUCE_METHOD::_REDUCE2D_H_) {
        switch (_src->Type())
        {
        case decx::_DATA_TYPES_FLAGS_::_FP32_:
            decx::reduce::matrix_reduce2D_1way_sum_fp32<true>(_src, _dst);
            break;
        default:
            break;
        }
    }
    else if (_reduce2D_mode == de::REDUCE_METHOD::_REDUCE2D_V_) {
        switch (_src->Type())
        {
        case decx::_DATA_TYPES_FLAGS_::_FP32_:
            decx::reduce::matrix_reduce2D_1way_sum_fp32<false>(_src, _dst);
            break;
        default:
            break;
        }
    }
    else {
        decx::err::MeaningLessFlag<true>(&handle);
    }

    return handle;
}



_DECX_API_ de::DH de::cuda::Max(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (_reduce2D_mode == de::REDUCE_METHOD::_REDUCE2D_H_) {
        switch (_src->Type())
        {
        case decx::_DATA_TYPES_FLAGS_::_FP32_:
            decx::reduce::matrix_reduce2D_1way_cmp_fp32<true, true>(_src, _dst);
            break;
        default:
            break;
        }
    }
    else if (_reduce2D_mode == de::REDUCE_METHOD::_REDUCE2D_V_) {
        switch (_src->Type())
        {
        case decx::_DATA_TYPES_FLAGS_::_FP32_:
            decx::reduce::matrix_reduce2D_1way_cmp_fp32<true, false>(_src, _dst);
            break;
        default:
            break;
        }
    }
    else {
        decx::err::MeaningLessFlag<true>(&handle);
    }

    return handle;
}



_DECX_API_ de::DH de::cuda::Min(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (_reduce2D_mode == de::REDUCE_METHOD::_REDUCE2D_H_) {
        switch (_src->Type())
        {
        case decx::_DATA_TYPES_FLAGS_::_FP32_:
            decx::reduce::matrix_reduce2D_1way_cmp_fp32<false, true>(_src, _dst);
            break;
        default:
            break;
        }
    }
    else if (_reduce2D_mode == de::REDUCE_METHOD::_REDUCE2D_V_) {
        switch (_src->Type())
        {
        case decx::_DATA_TYPES_FLAGS_::_FP32_:
            decx::reduce::matrix_reduce2D_1way_cmp_fp32<false, false>(_src, _dst);
            break;
        default:
            break;
        }
    }
    else {
        decx::err::MeaningLessFlag<true>(&handle);
    }

    return handle;
}
