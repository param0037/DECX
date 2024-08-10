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
#include "../../../../common/FP16/float_half_convert.h"
#include "../Vector_reduce.h"


namespace decx
{
    namespace reduce
    {
        template <bool _async_call>
        static void _vector_sum_caller(decx::_Vector* src, de::Number* res, de::DH *handle, const uint32_t _fp16_accu, const uint32_t _stream_id = 0);
        template <bool _async_call>
        static void _dev_vector_sum_caller(decx::_GPU_Vector* src, de::Number* res, de::DH *handle, const uint32_t _fp16_accu, const uint32_t _stream_id = 0);

        template <bool _async_call, bool _is_max>
        static void _vector_cmp_caller(decx::_Vector* src, de::Number* res, de::DH* handle, const uint32_t _stream_id = 0);
        template <bool _async_call, bool _is_max>
        static void _dev_vector_cmp_caller(decx::_GPU_Vector* src, de::Number* res, de::DH* handle, const uint32_t _stream_id = 0);
    }
}


template <bool _async_call>
static void decx::reduce::_vector_sum_caller(decx::_Vector* src, de::Number* res, de::DH *handle, const uint32_t _fp16_accu, const uint32_t _stream_id)
{
    if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP32_) {
        decx::reduce::vector_reduce_sum_fp32(src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_UINT8_) {
        decx::reduce::vector_reduce_sum_u8_i32(src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP16_) {
        decx::reduce::vector_reduce_sum_fp16(src, res, _fp16_accu);
    }
    else {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE, UNSUPPORTED_TYPE);
    }
}


template <bool _async_call>
static void decx::reduce::_dev_vector_sum_caller(decx::_GPU_Vector* src, de::Number* res, de::DH* handle, const uint32_t _fp16_accu, const uint32_t _stream_id)
{
    if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP32_) {
        decx::reduce::dev_vector_reduce_sum_fp32(src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_UINT8_) {
        decx::reduce::dev_vector_reduce_sum_u8_i32(src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP16_) {
        decx::reduce::dev_vector_reduce_sum_fp16(src, res, _fp16_accu);
    }
    else {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE, UNSUPPORTED_TYPE);
    }
}


template <bool _async_call, bool _is_max>
static void decx::reduce::_vector_cmp_caller(decx::_Vector* src, de::Number* res, de::DH* handle, const uint32_t _stream_id)
{
    if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP32_) {
        decx::reduce::vector_reduce_cmp_fp32<_is_max>(src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_UINT8_) {
        decx::reduce::vector_reduce_cmp_u8<_is_max>(src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP16_) {
        decx::reduce::vector_reduce_cmp_fp16<_is_max>(src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP64_) {
        decx::reduce::vector_reduce_cmp_fp64<_is_max>(src, res);
    }
    else {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE, UNSUPPORTED_TYPE);
    }
}



template <bool _async_call, bool _is_max>
static void decx::reduce::_dev_vector_cmp_caller(decx::_GPU_Vector* src, de::Number* res, de::DH* handle, const uint32_t _stream_id)
{
    if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP32_) {
        decx::reduce::dev_vector_reduce_cmp_fp32<_is_max>(src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_UINT8_) {
        decx::reduce::dev_vector_reduce_cmp_u8<_is_max>(src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP16_) {
        decx::reduce::dev_vector_reduce_cmp_fp16<_is_max>(src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP64_) {
        decx::reduce::dev_vector_reduce_cmp_fp64<_is_max>(src, res);
    }
    else {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE, UNSUPPORTED_TYPE);
    }
}




_DECX_API_ de::DH de::cuda::Sum(de::Vector& src, de::Number* res, const uint32_t _fp16_accu)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);

    decx::reduce::_vector_sum_caller<false>(_src, res, &handle, _fp16_accu);

    return handle;
}


_DECX_API_ de::DH de::cuda::Max(de::Vector& src, de::Number* res)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);

    decx::reduce::_vector_cmp_caller<false, true>(_src, res, &handle);

    return handle;
}



_DECX_API_ de::DH de::cuda::Min(de::Vector& src, de::Number* res)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_Vector* _src = dynamic_cast<decx::_Vector*>(&src);

    decx::reduce::_vector_cmp_caller<false, false>(_src, res, &handle);

    return handle;
}



_DECX_API_ de::DH de::cuda::Sum(de::GPU_Vector& src, de::Number* res, const uint32_t _fp16_accu)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);

    decx::reduce::_dev_vector_sum_caller<false>(_src, res, &handle, _fp16_accu);

    return handle;
}


_DECX_API_ de::DH de::cuda::Max(de::GPU_Vector& src, de::Number* res)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);

    decx::reduce::_dev_vector_cmp_caller<false, true>(_src, res, &handle);

    return handle;
}


_DECX_API_ de::DH de::cuda::Min(de::GPU_Vector& src, de::Number* res)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Vector* _src = dynamic_cast<decx::_GPU_Vector*>(&src);

    decx::reduce::_dev_vector_cmp_caller<false, false>(_src, res, &handle);

    return handle;
}

#ifdef _SELECTED_CALL_P2_
#undef _SELECTED_CALL_P2_
#endif

#ifdef _SELECTED_CALL_P3_
#undef _SELECTED_CALL_P3_
#endif