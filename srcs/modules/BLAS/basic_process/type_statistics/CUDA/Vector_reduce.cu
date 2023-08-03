/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "Vector_reduce_sum.cuh"
#include "Vector_reduce_cmp.cuh"
#include "../../float_half_convert.h"
#include "../Vector_reduce.h"


#define _SELECTED_CALL_(__func_name, _param1, _param2) { \
    if (_async_call) {      decx::async::register_async_task(_stream_id, __func_name, _param1, _param2); }       \
    else {  __func_name(_param1, _param2); }   \
}


namespace decx
{
    namespace reduce
    {
        template <bool _async_call>
        static void _vector_sum_caller(decx::_Vector* src, double* res, de::DH *handle, const uint32_t _stream_id = 0);
        template <bool _async_call>
        static void _dev_vector_sum_caller(decx::_GPU_Vector* src, double* res, de::DH *handle, const uint32_t _stream_id = 0);

        template <bool _async_call, bool _is_max>
        static void _vector_cmp_caller(decx::_Vector* src, double* res, de::DH* handle, const uint32_t _stream_id = 0);
        template <bool _async_call, bool _is_max>
        static void _dev_vector_cmp_caller(decx::_GPU_Vector* src, double* res, de::DH* handle, const uint32_t _stream_id = 0);
    }
}


template <bool _async_call>
static void decx::reduce::_vector_sum_caller(decx::_Vector* src, double* res, de::DH *handle, const uint32_t _stream_id)
{
    if (src->Type() == decx::_DATA_TYPES_FLAGS_::_FP32_) {
        float _res_fp32;
        _SELECTED_CALL_(decx::reduce::vector_reduce_sum_fp32, src, &_res_fp32);
        *res = _res_fp32;
    }
    else if (src->Type() == decx::_DATA_TYPES_FLAGS_::_UINT8_) {
        int32_t _res_i32;
        _SELECTED_CALL_(decx::reduce::vector_reduce_sum_u8_i32, src, &_res_i32);
        *res = _res_i32;
    }
    else if (src->Type() == decx::_DATA_TYPES_FLAGS_::_FP16_) {
        float _res_fp32;
        _SELECTED_CALL_(decx::reduce::vector_reduce_sum_fp16_fp32, src, &_res_fp32);
        *res = _res_fp32;
    }
    else {
        decx::err::Unsupported_Type<true>(handle);
    }
}


template <bool _async_call>
static void decx::reduce::_dev_vector_sum_caller(decx::_GPU_Vector* src, double* res, de::DH* handle, const uint32_t _stream_id)
{
    if (src->Type() == decx::_DATA_TYPES_FLAGS_::_FP32_) {
        float _res_fp32;
        _SELECTED_CALL_(decx::reduce::dev_vector_reduce_sum_fp32, src, &_res_fp32);
        *res = _res_fp32;
    }
    else if (src->Type() == decx::_DATA_TYPES_FLAGS_::_UINT8_) {
        int32_t _res_i32;
        _SELECTED_CALL_(decx::reduce::dev_vector_reduce_sum_u8_i32, src, &_res_i32);
        *res = _res_i32;
    }
    else if (src->Type() == decx::_DATA_TYPES_FLAGS_::_FP16_) {
        float _res_fp32;
        _SELECTED_CALL_(decx::reduce::dev_vector_reduce_sum_fp16_fp32, src, &_res_fp32);
        *res = _res_fp32;
    }
    else {
        decx::err::Unsupported_Type<true>(handle);
    }
}


template <bool _async_call, bool _is_max>
static void decx::reduce::_vector_cmp_caller(decx::_Vector* src, double* res, de::DH* handle, const uint32_t _stream_id)
{
    if (src->Type() == decx::_DATA_TYPES_FLAGS_::_FP32_) {
        float _res_fp32;
        _SELECTED_CALL_(decx::reduce::vector_reduce_cmp_fp32<_is_max>, src, &_res_fp32);
        *res = _res_fp32;
    }
    else if (src->Type() == decx::_DATA_TYPES_FLAGS_::_UINT8_) {
        uint8_t _res_u8;
        _SELECTED_CALL_(decx::reduce::vector_reduce_cmp_u8<_is_max>, src, &_res_u8);
        *res = _res_u8;
    }
    else if (src->Type() == decx::_DATA_TYPES_FLAGS_::_FP16_) {
        de::Half _res_fp16;
        _SELECTED_CALL_(decx::reduce::vector_reduce_cmp_fp16<_is_max>, src, &_res_fp16);
        *res = de::Half2Float(_res_fp16);
    }
    else if (src->Type() == decx::_DATA_TYPES_FLAGS_::_FP64_) {
        double _res_fp64;
        _SELECTED_CALL_(decx::reduce::vector_reduce_cmp_fp64<_is_max>, src, &_res_fp64);
        *res = _res_fp64;
    }
    else {
        decx::err::Unsupported_Type<true>(handle);
    }
}



template <bool _async_call, bool _is_max>
static void decx::reduce::_dev_vector_cmp_caller(decx::_GPU_Vector* src, double* res, de::DH* handle, const uint32_t _stream_id)
{
    if (src->Type() == decx::_DATA_TYPES_FLAGS_::_FP32_) {
        float _res_fp32;
        _SELECTED_CALL_(decx::reduce::dev_vector_reduce_cmp_fp32<_is_max>, src, &_res_fp32);
        *res = _res_fp32;
    }
    else if (src->Type() == decx::_DATA_TYPES_FLAGS_::_UINT8_) {
        uint8_t _res_u8;
        _SELECTED_CALL_(decx::reduce::dev_vector_reduce_cmp_u8<_is_max>, src, &_res_u8);
        *res = _res_u8;
    }
    else if (src->Type() == decx::_DATA_TYPES_FLAGS_::_FP16_) {
        de::Half _res_fp16;
        _SELECTED_CALL_(decx::reduce::dev_vector_reduce_cmp_fp16<_is_max>, src, &_res_fp16);
        *res = de::Half2Float(_res_fp16);
    }
    else if (src->Type() == decx::_DATA_TYPES_FLAGS_::_FP64_) {
        double _res_fp64;
        _SELECTED_CALL_(decx::reduce::dev_vector_reduce_cmp_fp64<_is_max>, src, &_res_fp64);
        *res = _res_fp64;
    }
    else {
        decx::err::Unsupported_Type<true>(handle);
    }
}




_DECX_API_ de::DH de::cuda::Sum(de::Vector& src, double* res)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);

    decx::reduce::_vector_sum_caller<false>(_src, res, &handle);

    return handle;
}


_DECX_API_ de::DH de::cuda::Sum_Async(de::Vector& src, double* res, de::DecxStream& S)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);

    decx::reduce::_vector_sum_caller<true>(_src, res, &handle, S.Get_ID());

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

    decx::reduce::_vector_cmp_caller<false, true>(_src, res, &handle);

    return handle;
}


_DECX_API_ de::DH de::cuda::Max_Async(de::Vector& src, double* res, de::DecxStream& S)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);

    decx::reduce::_vector_cmp_caller<true, true>(_src, res, &handle, S.Get_ID());

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

    decx::reduce::_vector_cmp_caller<false, false>(_src, res, &handle);

    return handle;
}


_DECX_API_ de::DH de::cuda::Min_Async(de::Vector& src, double* res, de::DecxStream& S)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);

    decx::reduce::_vector_cmp_caller<true, false>(_src, res, &handle, S.Get_ID());

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

    decx::reduce::_dev_vector_sum_caller<false>(_src, res, &handle);

    return handle;
}

_DECX_API_ de::DH de::cuda::Sum_Async(de::GPU_Vector& src, double* res, de::DecxStream& S)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);

    decx::reduce::_dev_vector_sum_caller<true>(_src, res, &handle, S.Get_ID());

    return handle;
}


_DECX_API_ de::DH de::cuda::Max(de::GPU_Vector& src, double* res)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);

    decx::reduce::_dev_vector_cmp_caller<false, true>(_src, res, &handle);

    return handle;
}


_DECX_API_ de::DH de::cuda::Max_Async(de::GPU_Vector& src, double* res, de::DecxStream& S)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);

    decx::reduce::_dev_vector_cmp_caller<true, true>(_src, res, &handle, S.Get_ID());

    return handle;
}



_DECX_API_ de::DH de::cuda::Min(de::GPU_Vector& src, double* res)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);

    decx::reduce::_dev_vector_cmp_caller<false, false>(_src, res, &handle);

    return handle;
}


_DECX_API_ de::DH de::cuda::Min_Async(de::GPU_Vector& src, double* res, de::DecxStream& S)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::CUDA_Not_init<true>(&handle);
        return handle;
    }

    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);

    decx::reduce::_dev_vector_cmp_caller<true, false>(_src, res, &handle, S.Get_ID());

    return handle;
}


#ifdef _SELECTED_CALL_
#undef _SELECTED_CALL_
#endif