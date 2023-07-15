/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "GPU_Vector_type_cast.cuh"


_DECX_API_ de::DH
de::cuda::TypeCast(de::GPU_Vector& src, de::GPU_Vector& dst, const int cvt_method)
{
    using namespace decx::type_cast;

    de::DH handle;
    if (!decx::cuda::_is_CUDA_init()) {
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
        return handle;
    }

    decx::err::Success(&handle);

    const decx::_GPU_Vector* _src = dynamic_cast<const decx::_GPU_Vector*>(&src);
    decx::_GPU_Vector* _dst = dynamic_cast<decx::_GPU_Vector*>(&dst);

    if (!_src->is_init()) {
        Print_Error_Message(4, CLASS_NOT_INIT);
        decx::err::ClassNotInit(&handle);
        return handle;
    }

    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        decx::err::CUDA_Stream_access_fail(&handle);
        return handle;
    }

    if (cvt_method == TypeCast_Method::CVT_FP32_FP64) {
        _dst->re_construct(decx::_DATA_TYPES_FLAGS_::_FP64_, _src->length);

        _mm128_cvtfp32_fp64_caller1D(static_cast<float4*>(_src->Vec.ptr), 
            static_cast<double2*>(_dst->Vec.ptr), _dst->_length / 4, S);
    }
    else if (cvt_method == TypeCast_Method::CVT_FP64_FP32) {
        _dst->re_construct(decx::_DATA_TYPES_FLAGS_::_FP32_, _src->length);

        _mm128_cvtfp64_fp32_caller1D(static_cast<double2*>(_src->Vec.ptr),
            static_cast<float4*>(_dst->Vec.ptr), _src->_length / 4, S);
    }
    else if (cvt_method == TypeCast_Method::CVT_INT32_FP32) {
        _dst->re_construct(decx::_DATA_TYPES_FLAGS_::_FP32_, _src->length);
        
        decx::type_cast::_mm128_cvti32_fp32_caller1D((int4*)_src->Vec.ptr, (float4*)_dst->Vec.ptr, _src->_length / 4, S);
    }
    else if (cvt_method == TypeCast_Method::CVT_UINT8_INT32) {
        _dst->re_construct(decx::_DATA_TYPES_FLAGS_::_INT32_, _src->length);
        
        decx::type_cast::_mm128_cvtfp32_i32_caller1D(
            (float4*)_src->Vec.ptr, (int4*)_dst->Vec.ptr, _dst->_length / 4, S);
    }
    else {
        Print_Error_Message(4, MEANINGLESS_FLAG);
        decx::err::InvalidParam(&handle);
        return handle;
    }

    return handle;
}