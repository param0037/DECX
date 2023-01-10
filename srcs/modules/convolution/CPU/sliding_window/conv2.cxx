/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "fp32/conv2_fp32.h"
#include "fp64/conv2_fp64.h"
#include "uint8/conv2_uint8.h"
#include "fp32/conv2_fp32_SK.h"
#include "fp64/conv2_fp64_SK.h"
#include "fp32/conv2_fp32_MK.h"
#include "fp64/conv2_fp64_MK.h"
#include "../../../classes/Matrix.h"
#include "../../../classes/MatrixArray.h"
#include "../../conv_utils.h"




namespace decx
{
    namespace cpu {
        void Conv2_fp32(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, const int flag, de::DH* handle);


        void Conv2_fp64(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, const int flag, de::DH* handle);


        void Conv2_uint8(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, const int flag, de::DH* handle, const int _output_type);
    }
}



void decx::cpu::Conv2_fp32(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, const int flag, de::DH* handle)
{
    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        dst->re_construct(src->type, src->width - kernel->width + 1, src->height - kernel->height + 1, decx::DATA_STORE_TYPE::Page_Default);
        decx::conv::_conv2_fp32_NB(src, kernel, dst, handle);
        break;

    case decx::conv_property::de_conv_zero_compensate:
        dst->re_construct(src->type, src->width, src->height, decx::DATA_STORE_TYPE::Page_Default);
        decx::conv::_conv2_fp32_BC(src, kernel, dst, handle);
        break;

    default:
        break;
    }

}


void decx::cpu::Conv2_fp64(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, const int flag, de::DH* handle)
{
    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        dst->re_construct(src->type, src->width - kernel->width + 1, src->height - kernel->height + 1, decx::DATA_STORE_TYPE::Page_Default);
        decx::conv::_conv2_fp64_NB(src, kernel, dst, handle);
        break;

    case decx::conv_property::de_conv_zero_compensate:
        dst->re_construct(src->type, src->width, src->height, decx::DATA_STORE_TYPE::Page_Default);
        decx::conv::_conv2_fp64_BC(src, kernel, dst, handle);
        break;

    default:
        break;
    }
}


void decx::cpu::Conv2_uint8(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, const int flag, de::DH* handle, const int _output_type)
{
    decx::PtrInfo<void> tmp_ker;
    if (decx::alloc::_host_virtual_page_malloc(&tmp_ker, kernel->total_bytes)) {
        decx::err::AllocateFailure(handle);
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }

    switch (kernel->type)
    {
    case decx::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::_cpy2D_plane((uint8_t*)kernel->Mat.ptr, (uint8_t*)tmp_ker.ptr, kernel->pitch, kernel->width,
            make_uint2(kernel->width, kernel->height));
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::_cpy2D_plane((float*)kernel->Mat.ptr, (float*)tmp_ker.ptr, kernel->pitch, kernel->width,
            make_uint2(kernel->width, kernel->height));
        break;

    default:
        break;
    }

    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        dst->re_construct(_output_type, src->width - kernel->width + 1, src->height - kernel->height + 1, decx::DATA_STORE_TYPE::Page_Default);
        decx::conv::_conv2_uint8_NB(src, kernel, tmp_ker.ptr, dst, handle, _output_type);
        break;

    case decx::conv_property::de_conv_zero_compensate:
        dst->re_construct(_output_type, src->width, src->height, decx::DATA_STORE_TYPE::Page_Default);
        decx::conv::_conv2_uint8_BC(src, kernel, tmp_ker.ptr, dst, handle, _output_type);
        break;

    default:
        break;
    }

    decx::alloc::_host_virtual_page_dealloc(&tmp_ker);
}


namespace decx
{
    namespace conv {
        void Conv2_single_channel_fp32(decx::_MatrixArray* _src, decx::_Matrix* _kernel, decx::_MatrixArray* _dst, const int flag, de::DH* handle);


        void Conv2_multi_channel_fp32(decx::_MatrixArray* _src, decx::_MatrixArray* _kernel, decx::_MatrixArray* _dst, const int flag, de::DH* handle);


        void Conv2_single_channel_fp64(decx::_MatrixArray* _src, decx::_Matrix* _kernel, decx::_MatrixArray* _dst, const int flag, de::DH* handle);


        void Conv2_multi_channel_fp64(decx::_MatrixArray* _src, decx::_MatrixArray* _kernel, decx::_MatrixArray* _dst, const int flag, de::DH* handle);
    }
}


void decx::conv::Conv2_single_channel_fp32(decx::_MatrixArray* _src, decx::_Matrix* _kernel, decx::_MatrixArray* _dst, const int flag, de::DH *handle)
{
    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(handle);
        Print_Error_Message(4, CPU_NOT_INIT);
    }

    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        _dst->re_construct(_src->type, _src->width - _kernel->width + 1, _src->height - _kernel->height + 1, _src->ArrayNumber, 
            decx::DATA_STORE_TYPE::Page_Default);
        decx::conv::_conv2_fp32_SK_NB(_src, _kernel, _dst, handle);
        break;

    case decx::conv_property::de_conv_zero_compensate:
        _dst->re_construct(_src->type, _src->width, _src->height, _src->ArrayNumber, decx::DATA_STORE_TYPE::Page_Default);
        decx::conv::_conv2_fp32_SK_BC(_src, _kernel, _dst, handle);
        break;

    default:
        break;
    }
}




void decx::conv::Conv2_multi_channel_fp32(decx::_MatrixArray* _src, decx::_MatrixArray* _kernel, decx::_MatrixArray* _dst, const int flag, de::DH* handle)
{
    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        _dst->re_construct(_src->type, _src->width - _kernel->width + 1, _src->height - _kernel->height + 1, _src->ArrayNumber,
            decx::DATA_STORE_TYPE::Page_Default);
        decx::conv::_conv2_fp32_MK_NB(_src, _kernel, _dst, handle);
        break;

    case decx::conv_property::de_conv_zero_compensate:
        _dst->re_construct(_src->type, _src->width, _src->height, _src->ArrayNumber, decx::DATA_STORE_TYPE::Page_Default);
        decx::conv::_conv2_fp32_MK_BC(_src, _kernel, _dst, handle);
        break;

    default:
        break;
    }
}



void decx::conv::Conv2_single_channel_fp64(decx::_MatrixArray* _src, decx::_Matrix* _kernel, decx::_MatrixArray* _dst, const int flag, de::DH* handle)
{
    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(handle);
        Print_Error_Message(4, CPU_NOT_INIT);
    }

    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        _dst->re_construct(_src->type, _src->width - _kernel->width + 1, _src->height - _kernel->height + 1, _src->ArrayNumber, 
            decx::DATA_STORE_TYPE::Page_Default);
        decx::conv::_conv2_fp64_SK_NB(_src, _kernel, _dst, handle);
        break;

    case decx::conv_property::de_conv_zero_compensate:
        _dst->re_construct(_src->type, _src->width, _src->height, _src->ArrayNumber, decx::DATA_STORE_TYPE::Page_Default);
        decx::conv::_conv2_fp64_SK_BC(_src, _kernel, _dst, handle);
        break;

    default:
        break;
    }
}



void decx::conv::Conv2_multi_channel_fp64(decx::_MatrixArray* _src, decx::_MatrixArray* _kernel, decx::_MatrixArray* _dst, const int flag, de::DH* handle)
{
    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(handle);
        Print_Error_Message(4, CPU_NOT_INIT);
    }

    switch (flag)
    {
    case decx::conv_property::de_conv_no_compensate:
        _dst->re_construct(_src->type, _src->width - _kernel->width + 1, _src->height - _kernel->height + 1, _src->ArrayNumber,
            decx::DATA_STORE_TYPE::Page_Default);
        decx::conv::_conv2_fp64_MK_NB(_src, _kernel, _dst, handle);
        break;

    case decx::conv_property::de_conv_zero_compensate:
        _dst->re_construct(_src->type, _src->width, _src->height, _src->ArrayNumber, decx::DATA_STORE_TYPE::Page_Default);
        decx::conv::_conv2_fp64_MK_BC(_src, _kernel, _dst, handle);
        break;

    default:
        break;
    }
}



namespace de
{
    namespace cpu
    {
        _DECX_API_ de::DH Conv2(de::Matrix& src, de::Matrix& kernel, de::Matrix& dst, const int flag, const int _output_type);


        _DECX_API_ de::DH Conv2_single_channel(de::MatrixArray& src, de::Matrix& kernel, de::MatrixArray& dst, const int flag);


        _DECX_API_ de::DH Conv2_multi_channel(de::MatrixArray& src, de::MatrixArray& kernel, de::MatrixArray& dst, const int flag);
    }
}



_DECX_API_
de::DH de::cpu::Conv2(de::Matrix& src, de::Matrix& kernel, de::Matrix& dst, const int flag, const int _output_type)
{
    de::DH handle;

    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _kernel = dynamic_cast<decx::_Matrix*>(&kernel);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);
    
    switch (_src->type)
    {
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::cpu::Conv2_fp32(_src, _kernel, _dst, flag, &handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP64_:
        decx::cpu::Conv2_fp64(_src, _kernel, _dst, flag, &handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::cpu::Conv2_uint8(_src, _kernel, _dst, flag, &handle, _output_type);
        break;
    default:
        break;
    }
    
    decx::err::Success(&handle);
    return handle;
}



_DECX_API_
de::DH de::cpu::Conv2_single_channel(de::MatrixArray& src, de::Matrix& kernel, de::MatrixArray& dst, const int flag)
{
    de::DH handle;

    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::_MatrixArray* _src = dynamic_cast<decx::_MatrixArray*>(&src);
    decx::_Matrix* _kernel = dynamic_cast<decx::_Matrix*>(&kernel);
    decx::_MatrixArray* _dst = dynamic_cast<decx::_MatrixArray*>(&dst);

    switch (_src->type)
    {
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::conv::Conv2_single_channel_fp32(_src, _kernel, _dst, flag, &handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP64_:
        decx::conv::Conv2_single_channel_fp64(_src, _kernel, _dst, flag, &handle);
        break;
    default:
        break;
    }

    decx::err::Success(&handle);
    return handle;
}


_DECX_API_
de::DH de::cpu::Conv2_multi_channel(de::MatrixArray& src, de::MatrixArray& kernel, de::MatrixArray& dst, const int flag)
{
    de::DH handle;

    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    decx::_MatrixArray* _src = dynamic_cast<decx::_MatrixArray*>(&src);
    decx::_MatrixArray* _kernel = dynamic_cast<decx::_MatrixArray*>(&kernel);
    decx::_MatrixArray* _dst = dynamic_cast<decx::_MatrixArray*>(&dst);
    
    switch (_src->type)
    {
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        decx::conv::Conv2_multi_channel_fp32(_src, _kernel, _dst, flag, &handle);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP64_:
        decx::conv::Conv2_multi_channel_fp64(_src, _kernel, _dst, flag, &handle);
        break;
    default:
        break;
    }

    decx::err::Success(&handle);
    return handle;
}