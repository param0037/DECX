/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/


#include "fp32/conv2_fp32.h"
#include "fp64/conv2_fp64.h"
#include "uint8/conv2_uint8.h"
#include "fp32/conv2_fp32_SK.h"
#include "fp64/conv2_fp64_SK.h"
#include "fp32/conv2_fp32_MK.h"
#include "fp64/conv2_fp64_MK.h"
#include "../../../../classes/Matrix.h"
#include "../../../../classes/MatrixArray.h"
#include "../../conv_utils.h"

#include "Filter2.h"
#include "../../../../BLAS/basic_process/extension/extend_flags.h"


namespace decx
{
    namespace conv 
    {
        template <bool _print>
        void Filter2D_fp32(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, const int flag, de::DH* handle);

        template <bool _print>
        void Filter2D_fp64(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, const int flag, de::DH* handle);

        template <bool _print>
        void Filter2D_uint8(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, const int flag, de::DH* handle, const de::_DATA_TYPES_FLAGS_ _output_type);
    }
}


template <bool _print>
void decx::conv::Filter2D_fp32(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, const int flag, de::DH* handle)
{
    switch (flag)
    {
    case de::extend_label::_EXTEND_NONE_:
        dst->re_construct(src->Type(), src->Width() - kernel->Width() + 1, src->Height() - kernel->Height() + 1);
        decx::conv::_conv2_fp32_NB<_print>(src, kernel, dst, handle);
        break;

    case de::extend_label::_EXTEND_CONSTANT_:
        dst->re_construct(src->Type(), src->Width(), src->Height());
        decx::conv::_conv2_fp32_BC<_print>(src, kernel, dst, handle);
        break;

    default:
        break;
    }

}


template <bool _print>
void decx::conv::Filter2D_fp64(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, const int flag, de::DH* handle)
{
    switch (flag)
    {
    case de::extend_label::_EXTEND_NONE_:
        dst->re_construct(src->Type(), src->Width() - kernel->Width() + 1, src->Height() - kernel->Height() + 1);
        decx::conv::_conv2_fp64_NB<_print>(src, kernel, dst, handle);
        break;

    case de::extend_label::_EXTEND_CONSTANT_:
        dst->re_construct(src->Type(), src->Width(), src->Height());
        decx::conv::_conv2_fp64_BC<_print>(src, kernel, dst, handle);
        break;

    default:
        break;
    }
}


template <bool _print>
void decx::conv::Filter2D_uint8(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, const int flag, de::DH* handle, 
    const de::_DATA_TYPES_FLAGS_ _output_type)
{
    decx::PtrInfo<void> tmp_ker;
    if (decx::alloc::_host_virtual_page_malloc(&tmp_ker, kernel->get_total_bytes())) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            ALLOC_FAIL);
        return;
    }
    
    switch (kernel->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::_cpy2D_plane((uint8_t*)kernel->Mat.ptr, (uint8_t*)tmp_ker.ptr, kernel->Pitch(), kernel->Width(),
            make_uint2(kernel->Width(), kernel->Height()));
        break;

    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::_cpy2D_plane((float*)kernel->Mat.ptr, (float*)tmp_ker.ptr, kernel->Pitch(), kernel->Width(),
            make_uint2(kernel->Width(), kernel->Height()));
        break;

    default:
        break;
    }
    
    switch (flag)
    {
    case de::extend_label::_EXTEND_NONE_:
        dst->re_construct(_output_type, src->Width() - kernel->Width() + 1, src->Height() - kernel->Height() + 1);
        decx::conv::_conv2_uint8_NB<_print>(src, kernel, tmp_ker.ptr, dst, handle, _output_type);
        break;

    case de::extend_label::_EXTEND_CONSTANT_:
        dst->re_construct(_output_type, src->Width(), src->Height());
        decx::conv::_conv2_uint8_BC<_print>(src, kernel, tmp_ker.ptr, dst, handle, _output_type);
        break;

    default:
        break;
    }

    decx::alloc::_host_virtual_page_dealloc(&tmp_ker);
}


namespace decx
{
    namespace conv 
    {
        void Conv2_single_channel_fp32(decx::_MatrixArray* _src, decx::_Matrix* _kernel, decx::_MatrixArray* _dst, const int flag, de::DH* handle);


        void Conv2_multi_channel_fp32(decx::_MatrixArray* _src, decx::_MatrixArray* _kernel, decx::_MatrixArray* _dst, const int flag, de::DH* handle);


        void Conv2_single_channel_fp64(decx::_MatrixArray* _src, decx::_Matrix* _kernel, decx::_MatrixArray* _dst, const int flag, de::DH* handle);


        void Conv2_multi_channel_fp64(decx::_MatrixArray* _src, decx::_MatrixArray* _kernel, decx::_MatrixArray* _dst, const int flag, de::DH* handle);
    }
}



void decx::conv::Conv2_single_channel_fp32(decx::_MatrixArray* _src, decx::_Matrix* _kernel, decx::_MatrixArray* _dst, const int flag, de::DH *handle)
{
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return;
    }

    switch (flag)
    {
    case de::extend_label::_EXTEND_NONE_:
        _dst->re_construct(_src->Type(), _src->Width() - _kernel->Width() + 1, _src->Height() - _kernel->Height() + 1, _src->Array_num());
        decx::conv::_conv2_fp32_SK_NB(_src, _kernel, _dst, handle);
        break;

    case de::extend_label::_EXTEND_CONSTANT_:
        _dst->re_construct(_src->Type(), _src->Width(), _src->Height(), _src->Array_num());
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
    case de::extend_label::_EXTEND_NONE_:
        _dst->re_construct(_src->Type(), _src->Width() - _kernel->Width() + 1, _src->Height() - _kernel->Height() + 1, _src->Array_num());
        decx::conv::_conv2_fp32_MK_NB(_src, _kernel, _dst, handle);
        break;

    case de::extend_label::_EXTEND_CONSTANT_:
        _dst->re_construct(_src->Type(), _src->Width(), _src->Height(), _src->Array_num());
        decx::conv::_conv2_fp32_MK_BC(_src, _kernel, _dst, handle);
        break;

    default:
        break;
    }
}



void decx::conv::Conv2_single_channel_fp64(decx::_MatrixArray* _src, decx::_Matrix* _kernel, decx::_MatrixArray* _dst, const int flag, de::DH* handle)
{
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return;
    }

    switch (flag)
    {
    case de::extend_label::_EXTEND_NONE_:
        _dst->re_construct(_src->Type(), _src->Width() - _kernel->Width() + 1, _src->Height() - _kernel->Height() + 1, _src->Array_num());
        decx::conv::_conv2_fp64_SK_NB(_src, _kernel, _dst, handle);
        break;

    case de::extend_label::_EXTEND_CONSTANT_:
        _dst->re_construct(_src->Type(), _src->Width(), _src->Height(), _src->Array_num());
        decx::conv::_conv2_fp64_SK_BC(_src, _kernel, _dst, handle);
        break;

    default:
        break;
    }
}



void decx::conv::Conv2_multi_channel_fp64(decx::_MatrixArray* _src, decx::_MatrixArray* _kernel, decx::_MatrixArray* _dst, const int flag, de::DH* handle)
{
    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return;
    }

    switch (flag)
    {
    case de::extend_label::_EXTEND_NONE_:
        _dst->re_construct(_src->Type(), _src->Width() - _kernel->Width() + 1, _src->Height() - _kernel->Height() + 1, _src->Array_num());
        decx::conv::_conv2_fp64_MK_NB(_src, _kernel, _dst, handle);
        break;

    case de::extend_label::_EXTEND_CONSTANT_:
        _dst->re_construct(_src->Type(), _src->Width(), _src->Height(), _src->Array_num());
        decx::conv::_conv2_fp64_MK_BC(_src, _kernel, _dst, handle);
        break;

    default:
        break;
    }
}




_DECX_API_ de::DH 
de::cpu::Filter2D(de::Matrix& src, de::Matrix& kernel, de::Matrix& dst, const int flag, const de::_DATA_TYPES_FLAGS_ _output_type)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _kernel = dynamic_cast<decx::_Matrix*>(&kernel);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::conv::Filter2D_fp32<true>(_src, _kernel, _dst, flag, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::conv::Filter2D_fp64<true>(_src, _kernel, _dst, flag, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::conv::Filter2D_uint8<true>(_src, _kernel, _dst, flag, &handle, _output_type);
        break;
    default:
        break;
    }

    decx::err::Success(&handle);
    return handle;
}



_DECX_API_ de::DH 
de::cpu::Filter2D_single_channel(de::MatrixArray& src, de::Matrix& kernel, de::MatrixArray& dst, const int flag)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_MatrixArray* _src = dynamic_cast<decx::_MatrixArray*>(&src);
    decx::_Matrix* _kernel = dynamic_cast<decx::_Matrix*>(&kernel);
    decx::_MatrixArray* _dst = dynamic_cast<decx::_MatrixArray*>(&dst);

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::conv::Conv2_single_channel_fp32(_src, _kernel, _dst, flag, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::conv::Conv2_single_channel_fp64(_src, _kernel, _dst, flag, &handle);
        break;
    default:
        break;
    }

    decx::err::Success(&handle);
    return handle;
}




_DECX_API_ de::DH 
de::cpu::Filter2D_multi_channel(de::MatrixArray& src, de::MatrixArray& kernel, de::MatrixArray& dst, const int flag)
{
    de::DH handle;

    if (!decx::cpu::_is_CPU_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CPU_not_init,
            CPU_NOT_INIT);
        return handle;
    }

    decx::_MatrixArray* _src = dynamic_cast<decx::_MatrixArray*>(&src);
    decx::_MatrixArray* _kernel = dynamic_cast<decx::_MatrixArray*>(&kernel);
    decx::_MatrixArray* _dst = dynamic_cast<decx::_MatrixArray*>(&dst);
    
    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::conv::Conv2_multi_channel_fp32(_src, _kernel, _dst, flag, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        decx::conv::Conv2_multi_channel_fp64(_src, _kernel, _dst, flag, &handle);
        break;
    default:
        break;
    }

    decx::err::Success(&handle);
    return handle;
}
