/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "Vector_reduce_sum.h"
#include "Vector_reduce_cmp.h"
#include "../../float_half_convert.h"


_DECX_API_ de::DH de::cuda::Sum(de::Vector& src, double* res)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);

    if (_src->Type() == decx::_DATA_TYPES_FLAGS_::_FP32_) {
        float _res_fp32;
        decx::reduce::vector_reduce_sum_fp32<true>(_src, &_res_fp32, &handle);
        *res = _res_fp32;
    }
    else if (_src->Type() == decx::_DATA_TYPES_FLAGS_::_UINT8_) {
        int32_t _res_i32;
        decx::reduce::vector_reduce_sum_u8_i32<true>(_src, &_res_i32, &handle);
        *res = _res_i32;
    }
    else if (_src->Type() == decx::_DATA_TYPES_FLAGS_::_FP16_) {
        float _res_fp32;
        decx::reduce::vector_reduce_sum_fp16_fp32<true>(_src, &_res_fp32, &handle);
        *res = _res_fp32;
    }

    return handle;
}



_DECX_API_ de::DH de::cuda::Max(de::Vector& src, double* res)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);

    if (_src->Type() == decx::_DATA_TYPES_FLAGS_::_FP32_) {
        float _res_fp32;
        decx::reduce::vector_reduce_max_fp32<true, true>(_src, &_res_fp32, &handle);
        *res = _res_fp32;
    }
    else if (_src->Type() == decx::_DATA_TYPES_FLAGS_::_UINT8_) {
        uint8_t _res_i32;
        decx::reduce::vector_reduce_max_u8<true, true>(_src, &_res_i32, &handle);
        *res = _res_i32;
    }
    else if (_src->Type() == decx::_DATA_TYPES_FLAGS_::_FP16_) {
        de::Half _res_fp16;
        decx::reduce::vector_reduce_max_fp16<true, true>(_src, &_res_fp16, &handle);
        *res = de::Half2Float(_res_fp16);
    }

    return handle;
}


_DECX_API_ de::DH de::cuda::Min(de::Vector& src, double* res)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);

    if (_src->Type() == decx::_DATA_TYPES_FLAGS_::_FP32_) {
        float _res_fp32;
        decx::reduce::vector_reduce_max_fp32<true, false>(_src, &_res_fp32, &handle);
        *res = _res_fp32;
    }
    else if (_src->Type() == decx::_DATA_TYPES_FLAGS_::_UINT8_) {
        uint8_t _res_i32;
        decx::reduce::vector_reduce_max_u8<true, false>(_src, &_res_i32, &handle);
        *res = _res_i32;
    }
    else if (_src->Type() == decx::_DATA_TYPES_FLAGS_::_FP16_) {
        de::Half _res_fp16;
        decx::reduce::vector_reduce_max_fp16<true, false>(_src, &_res_fp16, &handle);
        *res = de::Half2Float(_res_fp16);
    }

    return handle;
}



_DECX_API_ de::DH de::cuda::Sum(de::GPU_Vector& src, double* res)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);

    if (_src->Type() == decx::_DATA_TYPES_FLAGS_::_FP32_) {
        float _res_fp32;
        decx::reduce::dev_vector_reduce_sum_fp32<true>(_src, &_res_fp32, &handle);
        *res = _res_fp32;
    }
    else if (_src->Type() == decx::_DATA_TYPES_FLAGS_::_UINT8_) {
        int32_t _res_i32;
        decx::reduce::dev_vector_reduce_sum_u8_i32<true>(_src, &_res_i32, &handle);
        *res = _res_i32;
    }
    else if (_src->Type() == decx::_DATA_TYPES_FLAGS_::_FP16_) {
        float _res_fp32;
        decx::reduce::dev_vector_reduce_sum_fp16_fp32<true>(_src, &_res_fp32, &handle);
        *res = _res_fp32;
    }

    return handle;
}