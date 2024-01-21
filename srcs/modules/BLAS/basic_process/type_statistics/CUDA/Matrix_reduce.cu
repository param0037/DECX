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


#define _SELECTED_CALL_P2_(__func_name, _param1, _param2) {                                                        \
    if (_async_call) {      decx::async::register_async_task(_stream_id, __func_name, _param1, _param2); }      \
    else {  __func_name(_param1, _param2); }                                                                    \
}


#define _SELECTED_CALL_P3_(__func_name, _param1, _param2, _param3) {                                                        \
    if (_async_call) {      decx::async::register_async_task(_stream_id, __func_name, _param1, _param2, _param3); }      \
    else {  __func_name(_param1, _param2, _param3); }                                                                    \
}


namespace decx
{
    namespace reduce
    {
        template <bool _async_call>
        static void reduce_sum2D_full_caller(decx::_Matrix* src, de::DecxNumber* res, const uint32_t _fp16_accu, const uint32_t _stream_id = 0);
        template <bool _async_call>
        static void dev_reduce_sum2D_full_caller(decx::_GPU_Matrix* src, de::DecxNumber* res, const uint32_t _fp16_accu, const uint32_t _stream_id = 0);


        template <bool _is_max, bool _async_call>
        static void reduce_cmp2D_full_caller(decx::_Matrix* src, de::DecxNumber* res, const uint32_t _stream_id = 0);
        template <bool _is_max, bool _async_call>
        static void dev_reduce_cmp2D_full_caller(decx::_GPU_Matrix* src, de::DecxNumber* res, const uint32_t _stream_id = 0);
    }
}


template <bool _async_call>
static void decx::reduce::reduce_sum2D_full_caller(decx::_Matrix* src, de::DecxNumber* res, const uint32_t _fp16_accu, const uint32_t _stream_id)
{
    if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP32_) {
        _SELECTED_CALL_P2_(decx::reduce::matrix_reduce2D_full_sum_fp32, src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP16_) {
        _SELECTED_CALL_P3_(decx::reduce::matrix_reduce2D_full_sum_fp16, src, res, _fp16_accu);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_UINT8_) {
        _SELECTED_CALL_P2_(decx::reduce::matrix_reduce2D_full_sum_u8_i32, src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP64_) {
        _SELECTED_CALL_P2_(decx::reduce::matrix_reduce2D_full_sum_fp64, src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_INT32_) {
        _SELECTED_CALL_P2_(decx::reduce::matrix_reduce2D_full_sum_i32, src, res);
    }
}



template <bool _async_call>
static void decx::reduce::dev_reduce_sum2D_full_caller(decx::_GPU_Matrix* src, de::DecxNumber* res, const uint32_t _fp16_accu, const uint32_t _stream_id)
{
    if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP32_) {
        _SELECTED_CALL_P2_(decx::reduce::dev_matrix_reduce2D_full_sum_fp32, src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP16_) {
        _SELECTED_CALL_P3_(decx::reduce::dev_matrix_reduce2D_full_sum_fp16, src, res, _fp16_accu);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_UINT8_) {
        _SELECTED_CALL_P2_(decx::reduce::dev_matrix_reduce2D_full_sum_u8_i32, src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP64_) {
        _SELECTED_CALL_P2_(decx::reduce::dev_matrix_reduce2D_full_sum_fp64, src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_INT32_) {
        _SELECTED_CALL_P2_(decx::reduce::dev_matrix_reduce2D_full_sum_i32, src, res);
    }
}



template <bool _is_max, bool _async_call>
static void decx::reduce::reduce_cmp2D_full_caller(decx::_Matrix* src, de::DecxNumber* res, const uint32_t _stream_id)
{
    if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP32_) {
        _SELECTED_CALL_P2_(decx::reduce::matrix_reduce2D_full_cmp_fp32<_is_max>, src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP16_) {
        _SELECTED_CALL_P2_(decx::reduce::matrix_reduce2D_full_cmp_fp16<_is_max>, src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_UINT8_) {
        _SELECTED_CALL_P2_(decx::reduce::matrix_reduce2D_full_cmp_u8<_is_max>, src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP64_) {
        _SELECTED_CALL_P2_(decx::reduce::matrix_reduce2D_full_cmp_fp64<_is_max>, src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_INT32_) {
        _SELECTED_CALL_P2_(decx::reduce::matrix_reduce2D_full_cmp_int32<_is_max>, src, res);
    }
}



template <bool _is_max, bool _async_call>
static void decx::reduce::dev_reduce_cmp2D_full_caller(decx::_GPU_Matrix* src, de::DecxNumber* res, const uint32_t _stream_id)
{
    if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP32_) {
        _SELECTED_CALL_P2_(decx::reduce::dev_matrix_reduce2D_full_cmp_fp32<_is_max>, src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP16_) {
        _SELECTED_CALL_P2_(decx::reduce::dev_matrix_reduce2D_full_cmp_fp16<_is_max>, src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_UINT8_) {
        _SELECTED_CALL_P2_(decx::reduce::dev_matrix_reduce2D_full_cmp_u8<_is_max>, src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_FP64_) {
        _SELECTED_CALL_P2_(decx::reduce::dev_matrix_reduce2D_full_cmp_fp64<_is_max>, src, res);
    }
    else if (src->Type() == de::_DATA_TYPES_FLAGS_::_INT32_) {
        _SELECTED_CALL_P2_(decx::reduce::dev_matrix_reduce2D_full_cmp_int32<_is_max>, src, res);
    }
}



_DECX_API_ de::DH de::cuda::Sum(de::Matrix& src, de::DecxNumber& res, const uint32_t _fp16_accu)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::reduce::reduce_sum2D_full_caller<false>(_src, &res, _fp16_accu);

    return handle;
}



_DECX_API_ de::DH de::cuda::Sum_Async(de::Matrix& src, de::DecxNumber& res, const uint32_t _fp16_accu, de::DecxStream& S)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::reduce::reduce_sum2D_full_caller<true>(_src, &res, _fp16_accu, S.Get_ID());

    return handle;
}



_DECX_API_ de::DH de::cuda::Max(de::Matrix& src, de::DecxNumber& res)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);

    decx::reduce::reduce_cmp2D_full_caller<true, false>(_src, &res);

    return handle;
}



_DECX_API_ de::DH de::cuda::Max_Async(de::Matrix& src, de::DecxNumber& res, de::DecxStream& S)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::reduce::reduce_cmp2D_full_caller<true, true>(_src, &res, S.Get_ID());

    return handle;
}



_DECX_API_ de::DH de::cuda::Min(de::Matrix& src, de::DecxNumber& res)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::reduce::reduce_cmp2D_full_caller<false, false>(_src, &res);

    return handle;
}




_DECX_API_ de::DH de::cuda::Min_Async(de::Matrix& src, de::DecxNumber& res, de::DecxStream& S)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::reduce::reduce_cmp2D_full_caller<false, true>(_src, &res, S.Get_ID());

    return handle;
}




_DECX_API_ de::DH de::cuda::Sum(de::GPU_Matrix& src, de::DecxNumber& res, const uint32_t _fp16_accu)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);

    decx::reduce::dev_reduce_sum2D_full_caller<false>(_src, &res, _fp16_accu);

    return handle;
}




_DECX_API_ de::DH de::cuda::Sum_Async(de::GPU_Matrix& src, de::DecxNumber& res, const uint32_t _fp16_accu, de::DecxStream& S)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);

    decx::reduce::dev_reduce_sum2D_full_caller<true>(_src, &res, _fp16_accu, S.Get_ID());

    return handle;
}




_DECX_API_ de::DH de::cuda::Max(de::GPU_Matrix& src, de::DecxNumber& res)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::reduce::dev_reduce_cmp2D_full_caller<true, false>(_src, &res);

    return handle;
}




_DECX_API_ de::DH de::cuda::Max_Async(de::GPU_Matrix& src, de::DecxNumber& res, de::DecxStream& S)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::reduce::dev_reduce_cmp2D_full_caller<true, true>(_src, &res, S.Get_ID());

    return handle;
}



_DECX_API_ de::DH de::cuda::Min(de::GPU_Matrix& src, de::DecxNumber& res)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::reduce::dev_reduce_cmp2D_full_caller<false, false>(_src, &res);

    return handle;
}



_DECX_API_ de::DH de::cuda::Min_Async(de::GPU_Matrix& src, de::DecxNumber& res, de::DecxStream& S)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::reduce::dev_reduce_cmp2D_full_caller<false, true>(_src, &res, S.Get_ID());

    return handle;
}


// ---------------------------------------------- 1-way --------------------------------------------


namespace decx
{
    namespace reduce
    {
        template <bool _async_call>
        static void reduce_sum2D_1way_caller(decx::_Matrix* src, decx::_Vector* dst, const int32_t _reduce2D_mode, const uint32_t _fp16_accu, const uint32_t _stream_id = 0);
        template <bool _async_call>
        static void dev_reduce_sum2D_1way_caller(decx::_GPU_Matrix* src, decx::_GPU_Vector* dst, const int32_t _reduce2D_mode, const uint32_t _fp16_accu, const uint32_t _stream_id = 0);

        template <bool _is_max, bool _async_call>
        static void reduce_cmp2D_1way_caller(decx::_Matrix* src, decx::_Vector* dst, const int32_t _reduce2D_mode, const uint32_t _stream_id = 0);
        template <bool _is_max, bool _async_call>
        static void dev_reduce_cmp2D_1way_caller(decx::_GPU_Matrix* src, decx::_GPU_Vector* dst, const int32_t _reduce2D_mode, const uint32_t _stream_id = 0);
    }
}



template <bool _async_call>
static void decx::reduce::reduce_sum2D_1way_caller(decx::_Matrix* src, decx::_Vector* dst, const int32_t _reduce2D_mode, const uint32_t _fp16_accu, const uint32_t _stream_id)
{
    if (_reduce2D_mode == de::REDUCE_METHOD::_REDUCE2D_H_) {
        switch (src->Type())
        {
        case de::_DATA_TYPES_FLAGS_::_FP32_:
            dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP32_, src->Height());
            _SELECTED_CALL_P2_(decx::reduce::matrix_reduce2D_1way_sum_fp32<true>, src, dst);
            break;

        case de::_DATA_TYPES_FLAGS_::_FP64_:
            dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP64_, src->Height());
            _SELECTED_CALL_P2_(decx::reduce::matrix_reduce2D_1way_sum_fp64<true>, src, dst);
            break;

        case de::_DATA_TYPES_FLAGS_::_FP16_:
            dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP32_, src->Height());
            _SELECTED_CALL_P3_(decx::reduce::matrix_reduce2D_1way_sum_fp16<true>, src, dst, _fp16_accu);
            break;

        case de::_DATA_TYPES_FLAGS_::_UINT8_:
            dst->re_construct(de::_DATA_TYPES_FLAGS_::_INT32_, src->Height());
            _SELECTED_CALL_P2_(decx::reduce::matrix_reduce2D_1way_sum_u8_i32<true>, src, dst);
            break;

        default:
            break;
        }
    }
    else if (_reduce2D_mode == de::REDUCE_METHOD::_REDUCE2D_V_) {
        switch (src->Type())
        {
        case de::_DATA_TYPES_FLAGS_::_FP32_:
            dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP32_, src->Width());
            _SELECTED_CALL_P2_(decx::reduce::matrix_reduce2D_1way_sum_fp32<false>, src, dst);
            break;

        case de::_DATA_TYPES_FLAGS_::_FP64_:
            dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP64_, src->Width());
            _SELECTED_CALL_P2_(decx::reduce::matrix_reduce2D_1way_sum_fp64<false>, src, dst);
            break;

        case de::_DATA_TYPES_FLAGS_::_FP16_:
            dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP32_, src->Width());
            _SELECTED_CALL_P3_(decx::reduce::matrix_reduce2D_1way_sum_fp16<false>, src, dst, _fp16_accu);
            break;

        case de::_DATA_TYPES_FLAGS_::_UINT8_:
            dst->re_construct(de::_DATA_TYPES_FLAGS_::_INT32_, src->Width());
            _SELECTED_CALL_P2_(decx::reduce::matrix_reduce2D_1way_sum_u8_i32<false>, src, dst);
            break;
        default:
            break;
        }
    }
}



template <bool _is_max, bool _async_call>
static void decx::reduce::reduce_cmp2D_1way_caller(decx::_Matrix* src, decx::_Vector* dst, const int32_t _reduce2D_mode, const uint32_t _stream_id)
{
    if (_reduce2D_mode == de::REDUCE_METHOD::_REDUCE2D_H_) {
        switch (src->Type())
        {
        case de::_DATA_TYPES_FLAGS_::_FP32_:
            _SELECTED_CALL_P2_((decx::reduce::matrix_reduce2D_1way_cmp_fp32<_is_max, true>), src, dst);
            break;

        case de::_DATA_TYPES_FLAGS_::_FP16_:
            _SELECTED_CALL_P2_((decx::reduce::matrix_reduce2D_1way_cmp_fp16<_is_max, true>), src, dst);
            break;

        case de::_DATA_TYPES_FLAGS_::_UINT8_:
            _SELECTED_CALL_P2_((decx::reduce::matrix_reduce2D_1way_cmp_u8<_is_max, true>), src, dst);
            break;

        default:
            break;
        }
    }
    else if (_reduce2D_mode == de::REDUCE_METHOD::_REDUCE2D_V_) {
        switch (src->Type())
        {
        case de::_DATA_TYPES_FLAGS_::_FP32_:
            _SELECTED_CALL_P2_((decx::reduce::matrix_reduce2D_1way_cmp_fp32<_is_max, false>), src, dst);
            break;

        case de::_DATA_TYPES_FLAGS_::_FP16_:
            _SELECTED_CALL_P2_((decx::reduce::matrix_reduce2D_1way_cmp_fp16<_is_max, false>), src, dst);
            break;

        case de::_DATA_TYPES_FLAGS_::_UINT8_:
            _SELECTED_CALL_P2_((decx::reduce::matrix_reduce2D_1way_cmp_u8<_is_max, false>), src, dst);
            break;
        default:
            break;
        }
    }
}



template <bool _async_call>
static void decx::reduce::dev_reduce_sum2D_1way_caller(decx::_GPU_Matrix* src, decx::_GPU_Vector* dst, const int32_t _reduce2D_mode, const uint32_t _fp16_accu,
    const uint32_t _stream_id)
{
    if (_reduce2D_mode == de::REDUCE_METHOD::_REDUCE2D_H_) {
        switch (src->Type())
        {
        case de::_DATA_TYPES_FLAGS_::_FP32_:
            dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP32_, src->Height());
            _SELECTED_CALL_P2_(decx::reduce::dev_matrix_reduce2D_1way_sum_fp32<true>, src, dst);
            break;

        case de::_DATA_TYPES_FLAGS_::_FP64_:
            dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP64_, src->Height());
            _SELECTED_CALL_P2_(decx::reduce::dev_matrix_reduce2D_1way_sum_fp64<true>, src, dst);
            break;

        case de::_DATA_TYPES_FLAGS_::_FP16_:
            dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP32_, src->Height());
            _SELECTED_CALL_P3_(decx::reduce::dev_matrix_reduce2D_1way_sum_fp16<true>, src, dst, _fp16_accu);
            break;

        case de::_DATA_TYPES_FLAGS_::_UINT8_:
            dst->re_construct(de::_DATA_TYPES_FLAGS_::_INT32_, src->Height());
            _SELECTED_CALL_P2_(decx::reduce::dev_matrix_reduce2D_1way_sum_u8_i32<true>, src, dst);
            break;

        default:
            break;
        }
    }
    else if (_reduce2D_mode == de::REDUCE_METHOD::_REDUCE2D_V_) {
        switch (src->Type())
        {
        case de::_DATA_TYPES_FLAGS_::_FP32_:
            dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP32_, src->Width());
            _SELECTED_CALL_P2_(decx::reduce::dev_matrix_reduce2D_1way_sum_fp32<false>, src, dst);
            break;

        case de::_DATA_TYPES_FLAGS_::_FP64_:
            dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP64_, src->Width());
            _SELECTED_CALL_P2_(decx::reduce::dev_matrix_reduce2D_1way_sum_fp64<false>, src, dst);
            break;

        case de::_DATA_TYPES_FLAGS_::_FP16_:
            dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP32_, src->Width());
            _SELECTED_CALL_P3_(decx::reduce::dev_matrix_reduce2D_1way_sum_fp16<false>, src, dst, _fp16_accu);
            break;

        case de::_DATA_TYPES_FLAGS_::_UINT8_:
            dst->re_construct(de::_DATA_TYPES_FLAGS_::_INT32_, src->Width());
            _SELECTED_CALL_P2_(decx::reduce::dev_matrix_reduce2D_1way_sum_u8_i32<false>, src, dst);
            break;
        default:
            break;
        }
    }
}




template <bool _is_max, bool _async_call>
static void decx::reduce::dev_reduce_cmp2D_1way_caller(decx::_GPU_Matrix* src, decx::_GPU_Vector* dst, const int32_t _reduce2D_mode, const uint32_t _stream_id)
{
    if (_reduce2D_mode == de::REDUCE_METHOD::_REDUCE2D_H_) {
        dst->re_construct(src->Type(), src->Height());
        switch (src->Type())
        {
        case de::_DATA_TYPES_FLAGS_::_FP32_:
            _SELECTED_CALL_P2_((decx::reduce::dev_matrix_reduce2D_1way_cmp_fp32<_is_max, true>), src, dst);
            break;

        case de::_DATA_TYPES_FLAGS_::_FP16_:
            _SELECTED_CALL_P2_((decx::reduce::dev_matrix_reduce2D_1way_cmp_fp16<_is_max, true>), src, dst);
            break;

        case de::_DATA_TYPES_FLAGS_::_UINT8_:
            _SELECTED_CALL_P2_((decx::reduce::dev_matrix_reduce2D_1way_cmp_u8<_is_max, true>), src, dst);
            break;

        default:
            break;
        }
    }
    else if (_reduce2D_mode == de::REDUCE_METHOD::_REDUCE2D_V_) {
        dst->re_construct(src->Type(), src->Width());
        switch (src->Type())
        {
        case de::_DATA_TYPES_FLAGS_::_FP32_:
            _SELECTED_CALL_P2_((decx::reduce::dev_matrix_reduce2D_1way_cmp_fp32<_is_max, false>), src, dst);
            break;

        case de::_DATA_TYPES_FLAGS_::_FP16_:
            _SELECTED_CALL_P2_((decx::reduce::dev_matrix_reduce2D_1way_cmp_fp16<_is_max, false>), src, dst);
            break;

        case de::_DATA_TYPES_FLAGS_::_UINT8_:
            _SELECTED_CALL_P2_((decx::reduce::dev_matrix_reduce2D_1way_cmp_u8<_is_max, false>), src, dst);
            break;
        default:
            break;
        }
    }
}




_DECX_API_ de::DH de::cuda::Sum(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode, const uint32_t _fp16_accu)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (_reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_V_ && _reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_H_) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag, MEANINGLESS_FLAG);
        return handle;
    }
    decx::reduce::reduce_sum2D_1way_caller<false>(_src, _dst, _reduce2D_mode, _fp16_accu);

    return handle;
}



_DECX_API_ de::DH de::cuda::Sum_Async(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode, const uint32_t _fp16_accu, de::DecxStream& S)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (_reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_V_ && _reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_H_) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag, MEANINGLESS_FLAG);
        return handle;
    }
    decx::reduce::reduce_sum2D_1way_caller<true>(_src, _dst, _reduce2D_mode, _fp16_accu, S.Get_ID());

    return handle;
}



_DECX_API_ de::DH de::cuda::Sum(de::GPU_Matrix& src, de::GPU_Vector& dst, const int _reduce2D_mode, const uint32_t _fp16_accu)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Vector* _dst = dynamic_cast<decx::_GPU_Vector*>(&dst);

    if (_reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_V_ && _reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_H_) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag, MEANINGLESS_FLAG);
        return handle;
    }
    decx::reduce::dev_reduce_sum2D_1way_caller<false>(_src, _dst, _reduce2D_mode, _fp16_accu);

    return handle;
}



_DECX_API_ de::DH de::cuda::Sum_Async(de::GPU_Matrix& src, de::GPU_Vector& dst, const int _reduce2D_mode, const uint32_t _fp16_accu, de::DecxStream& S)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Vector* _dst = dynamic_cast<decx::_GPU_Vector*>(&dst);

    if (_reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_V_ && _reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_H_) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag, MEANINGLESS_FLAG);
        return handle;
    }
    decx::reduce::dev_reduce_sum2D_1way_caller<true>(_src, _dst, _fp16_accu, _reduce2D_mode);

    return handle;
}



_DECX_API_ de::DH de::cuda::Max(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (_reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_V_ && _reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_H_) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag, MEANINGLESS_FLAG);
        return handle;
    }
    decx::reduce::reduce_cmp2D_1way_caller<true, false>(_src, _dst, _reduce2D_mode);

    return handle;
}




_DECX_API_ de::DH de::cuda::Max_Async(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode, de::DecxStream& S)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (_reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_V_ && _reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_H_) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag, MEANINGLESS_FLAG);
        return handle;
    }
    decx::reduce::reduce_cmp2D_1way_caller<true, true>(_src, _dst, _reduce2D_mode, S.Get_ID());

    return handle;
}



_DECX_API_ de::DH de::cuda::Max(de::GPU_Matrix& src, de::GPU_Vector& dst, const int _reduce2D_mode)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Vector* _dst = dynamic_cast<decx::_GPU_Vector*>(&dst);

    if (_reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_V_ && _reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_H_) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag, MEANINGLESS_FLAG);
        return handle;
    }
    decx::reduce::dev_reduce_cmp2D_1way_caller<true, false>(_src, _dst, _reduce2D_mode);

    return handle;
}



_DECX_API_ de::DH de::cuda::Max_Async(de::GPU_Matrix& src, de::GPU_Vector& dst, const int _reduce2D_mode, de::DecxStream& S)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Vector* _dst = dynamic_cast<decx::_GPU_Vector*>(&dst);

    if (_reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_V_ && _reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_H_) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag, MEANINGLESS_FLAG);
        return handle;
    }
    decx::reduce::dev_reduce_cmp2D_1way_caller<true, true>(_src, _dst, _reduce2D_mode, S.Get_ID());

    return handle;
}



_DECX_API_ de::DH de::cuda::Min(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (_reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_V_ && _reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_H_) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag, MEANINGLESS_FLAG);
        return handle;
    }
    decx::reduce::reduce_cmp2D_1way_caller<false, false>(_src, _dst, _reduce2D_mode);

    return handle;
}



_DECX_API_ de::DH de::cuda::Min_Async(de::Matrix& src, de::Vector& dst, const int _reduce2D_mode, de::DecxStream& S)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    if (_reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_V_ && _reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_H_) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag, MEANINGLESS_FLAG);
        return handle;
    }
    decx::reduce::reduce_cmp2D_1way_caller<false, true>(_src, _dst, _reduce2D_mode, S.Get_ID());

    return handle;
}



_DECX_API_ de::DH de::cuda::Min(de::GPU_Matrix& src, de::GPU_Vector& dst, const int _reduce2D_mode)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Vector* _dst = dynamic_cast<decx::_GPU_Vector*>(&dst);

    if (_reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_V_ && _reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_H_) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag, MEANINGLESS_FLAG);
        return handle;
    }
    decx::reduce::dev_reduce_cmp2D_1way_caller<false, false>(_src, _dst, _reduce2D_mode);

    return handle;
}




_DECX_API_ de::DH de::cuda::Min_Async(de::GPU_Matrix& src, de::GPU_Vector& dst, const int _reduce2D_mode, de::DecxStream& S)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Vector* _dst = dynamic_cast<decx::_GPU_Vector*>(&dst);

    if (_reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_V_ && _reduce2D_mode != de::REDUCE_METHOD::_REDUCE2D_H_) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag, MEANINGLESS_FLAG);
        return handle;
    }
    decx::reduce::dev_reduce_cmp2D_1way_caller<false, true>(_src, _dst, _reduce2D_mode, S.Get_ID());

    return handle;
}



#ifdef _SELECTED_CALL_P2_
#undef _SELECTED_CALL_P2_
#endif



#ifdef _SELECTED_CALL_P3_
#undef _SELECTED_CALL_P3_
#endif