/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "GPU_Matrix_type_cast.cuh"
#include "../type_cast_methods.h"


_DECX_API_ de::DH 
de::cuda::TypeCast(de::GPU_Matrix& src, de::GPU_Matrix& dst, const int cvt_method)
{
    using namespace decx::type_cast;

    de::DH handle;
    if (!decx::cuP.is_init) {
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
        return handle;
    }

    decx::err::Success(&handle);

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    if (!_src->_init) {
        Print_Error_Message(4, CLASS_NOT_INIT);
        decx::err::ClassNotInit(&handle);
        return handle;
    }

    decx::cuda_stream* S = NULL;
    S = decx::CStream.stream_accessor_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        decx::err::CUDA_Stream_access_fail(&handle);
        return handle;
    }

    if (cvt_method == TypeCast_Method::CVT_FP32_FP64) {
        decx::type_cast::_mm128_cvtfp32_fp64_caller2D(
            (float4*)_src->Mat.ptr, (double2*)_dst->Mat.ptr, make_ulong2(_dst->pitch / 4, _src->height), _src->pitch, _dst->pitch, S);
    }
    else if (cvt_method == TypeCast_Method::CVT_FP64_FP32) {
        decx::type_cast::_mm128_cvtfp64_fp32_caller2D(
            (double2*)_src->Mat.ptr, (float4*)_dst->Mat.ptr, make_ulong2(_src->pitch / 4, _src->height), _src->pitch, _dst->pitch, S);
    }
    else if (cvt_method == TypeCast_Method::CVT_INT32_FP32) {
        _dst->re_construct(decx::_DATA_TYPES_FLAGS_::_FP32_, _src->width, _src->height);

        decx::type_cast::_mm128_cvti32_fp32_caller1D((int4*)_src->Mat.ptr, (float4*)_dst->Mat.ptr, _src->_element_num / 4, S);
    }
    else if (cvt_method == TypeCast_Method::CVT_UINT8_INT32) {
        _dst->re_construct(decx::_DATA_TYPES_FLAGS_::_INT32_, _src->width, _src->height);

        decx::type_cast::_mm128_cvtfp32_i32_caller1D(
            (float4*)_src->Mat.ptr, (int4*)_dst->Mat.ptr, _dst->_element_num / 4, S);
    }
    else {
        Print_Error_Message(4, MEANINGLESS_FLAG);
        decx::err::InvalidParam(&handle);
        return handle;
    }

    return handle;
}