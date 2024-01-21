/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "Dot_Product.cuh"
#include "DP1D_callers.cuh"


#define _SELECTED_CALL_P3_(__func_name, _param1, _param2, _param3) {                                                        \
    if (_async_call) {      decx::async::register_async_task(_stream_id, __func_name, _param1, _param2, _param3); }      \
    else {  __func_name(_param1, _param2, _param3); }                                                                    \
}


#define _SELECTED_CALL_P4_(__func_name, _param1, _param2, _param3, _param4) {                                                        \
    if (_async_call) {      decx::async::register_async_task(_stream_id, __func_name, _param1, _param2, _param3, _param4); }      \
    else {  __func_name(_param1, _param2, _param3, _param4); }                                                                    \
}


namespace decx
{
    namespace dot
    {
        template <bool _async_call>
        static void _vector_dot_caller(decx::_Vector* A, decx::_Vector* B, de::DecxNumber* res, de::DH* handle, const uint32_t _fp16_accu, const uint32_t _stream_id = 0);
        template <bool _async_call>
        static void _dev_vector_dot_caller(decx::_GPU_Vector* A, decx::_GPU_Vector* B, de::DecxNumber* res, de::DH* handle, const uint32_t _fp16_accu, const uint32_t _stream_id = 0);
    }
}



template <bool _async_call>
static void decx::dot::_vector_dot_caller(decx::_Vector* A, decx::_Vector* B, de::DecxNumber* res, de::DH* handle, const uint32_t _fp16_accu, const uint32_t _stream_id)
{
    if (A->Type() == de::_DATA_TYPES_FLAGS_::_FP32_) {
        _SELECTED_CALL_P3_(decx::dot::vector_dot_fp32, A, B, res);
    }
    else if (A->Type() == de::_DATA_TYPES_FLAGS_::_FP16_) {
        _SELECTED_CALL_P4_(decx::dot::vector_dot_fp16, A, B, res, _fp16_accu);
    }
    else if (A->Type() == de::_DATA_TYPES_FLAGS_::_FP64_) {
        _SELECTED_CALL_P3_(decx::dot::vector_dot_fp64, A, B, res);
    }
    else if (A->Type() == de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_) {
        _SELECTED_CALL_P3_(decx::dot::vector_dot_cplxf, A, B, res);
    }
    else {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE, UNSUPPORTED_TYPE);
    }
}



template <bool _async_call>
static void decx::dot::_dev_vector_dot_caller(decx::_GPU_Vector* A, decx::_GPU_Vector* B, de::DecxNumber* res, de::DH* handle, const uint32_t _fp16_accu, const uint32_t _stream_id)
{
    if (A->Type() == de::_DATA_TYPES_FLAGS_::_FP32_) {
        _SELECTED_CALL_P3_(decx::dot::dev_vector_dot_fp32, A, B, res);
    }
    else if (A->Type() == de::_DATA_TYPES_FLAGS_::_FP16_) {
        _SELECTED_CALL_P4_(decx::dot::dev_vector_dot_fp16, A, B, res, _fp16_accu);
    }
    else if (A->Type() == de::_DATA_TYPES_FLAGS_::_FP64_) {
        _SELECTED_CALL_P3_(decx::dot::dev_vector_dot_fp64, A, B, res);
    }
    else if (A->Type() == de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_) {
        _SELECTED_CALL_P3_(decx::dot::dev_vector_dot_cplxf, A, B, res);
    }
    else {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE, UNSUPPORTED_TYPE);
    }
}



_DECX_API_ de::DH de::cuda::Dot_product(de::Vector& A, de::Vector& B, de::DecxNumber& res, const uint32_t _fp16_accu)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_Vector* _A = dynamic_cast<decx::_Vector*>(&A);
    decx::_Vector* _B = dynamic_cast<decx::_Vector*>(&B);

    decx::dot::_vector_dot_caller<false>(_A, _B, &res, &handle, _fp16_accu);

    return handle;
}



_DECX_API_ de::DH de::cuda::Dot_product(de::GPU_Vector& A, de::GPU_Vector& B, de::DecxNumber& res, const uint32_t _fp16_accu)
{
    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Vector* _A = dynamic_cast<decx::_GPU_Vector*>(&A);
    decx::_GPU_Vector* _B = dynamic_cast<decx::_GPU_Vector*>(&B);

    decx::dot::_dev_vector_dot_caller<false>(_A, _B, &res, &handle, _fp16_accu);

    return handle;
}




#ifdef _SELECTED_CALL_P3_
#undef _SELECTED_CALL_P3_
#endif

#ifdef _SELECTED_CALL_P4_
#undef _SELECTED_CALL_P4_
#endif