/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "fp32/conv2_border_ignored_fp32.h"
#include "fp16/conv2_border_ignore_fp16.h"
#include "fp32/conv2_border_const_fp32.h"
#include "fp16/conv2_border_const_fp16.h"
#include "uint8/conv2_border_ignored_uint8.h"
#include "uint8/conv2_border_const_uint8.h"
#include "Filter2.h"
#include "../../../../BLAS/basic_process/extension/extend_flags.h"


namespace decx
{
    namespace conv
    {
        template <bool _print>
        void Conv2_fp32(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, const uint flag, de::DH* handle);
        

        template <bool _print>
        void Conv2_fp16(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, const uint conv_flag, const int accu_flag, de::DH* handle);


        template <bool _print>
        void Conv2_uint8(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, 
            const int flag, const de::_DATA_TYPES_FLAGS_ output_type, de::DH* handle);
    }
}



template <bool _print>
void decx::conv::Conv2_fp32(decx::_Matrix* _src, decx::_Matrix* _kernel, decx::_Matrix* _dst, const uint flag, 
    de::DH* handle)
{
    switch (flag)
    {
    case de::extend_label::_EXTEND_NONE_:
        decx::conv::conv2_fp32_border_ignore<_print>(_src, _kernel, _dst, handle);
        break;

    case de::extend_label::_EXTEND_CONSTANT_:
        decx::conv::conv2_fp32_border_zero<_print>(_src, _kernel, _dst, handle);
        break;
    default:
        break;
    }
}




template <bool _print>
void decx::conv::Conv2_uint8(decx::_Matrix* _src, decx::_Matrix* _kernel, decx::_Matrix* _dst, 
    const int flag, const de::_DATA_TYPES_FLAGS_ output_type, de::DH* handle)
{
    decx::conv::_cuda_conv2_uc8_uc8_preset _conv2_preset;
    _conv2_preset.memcpy_flag = cudaMemcpyHostToDevice;
    _conv2_preset._Kparams._src_confs.gen_matrix_configs(_src);
    _conv2_preset._Kparams._kernel_confs.gen_matrix_configs(_kernel);

    switch (flag)
    {
    case de::extend_label::_EXTEND_NONE_:
        _dst->re_construct(output_type, _src->Width() - _kernel->Width() + 1,
            _src->Height() - _kernel->Height() + 1);

        _conv2_preset._Kparams._dst_confs.gen_matrix_configs(_dst);

        if (output_type == de::_DATA_TYPES_FLAGS_::_UINT8_) {
            decx::conv::conv2_uc8_uc8_NB<_print>(&_conv2_preset, handle);
        }
        else {
            decx::conv::conv2_uc8_fp32_NB<_print>(&_conv2_preset, handle);
        }
        break;

    case de::extend_label::_EXTEND_CONSTANT_:
        _dst->re_construct(output_type, _src->Width(), _src->Height());

        _conv2_preset._Kparams._dst_confs.gen_matrix_configs(_dst);

        if (output_type == de::_DATA_TYPES_FLAGS_::_UINT8_) {
            decx::conv::conv2_uc8_uc8_BC<_print>(&_conv2_preset, handle);
        }
        else {
            decx::conv::conv2_uc8_fp32_BC<_print>(&_conv2_preset, handle);
        }
        break;

    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag, MEANINGLESS_FLAG);
        break;
    }

    _conv2_preset.release();
}



template <bool _print>
void decx::conv::Conv2_fp16(decx::_Matrix* _src, decx::_Matrix* _kernel, decx::_Matrix* _dst, const uint conv_flag, const int accu_flag,
    de::DH* handle)
{
    switch (conv_flag)
    {
    case de::extend_label::_EXTEND_NONE_:
        decx::conv::conv2_fp16_border_ignore<true>(_src, _kernel, _dst, handle, accu_flag);
        break;

    case de::extend_label::_EXTEND_CONSTANT_:
        decx::conv::conv2_fp16_border_zero<true>(_src, _kernel, _dst, handle, accu_flag);
        break;
    default:
        break;
    }
}



_DECX_API_
de::DH de::cuda::Filter2D(de::Matrix& src, de::Matrix& kernel, de::Matrix& dst,
    const int conv_flag, const int accu_flag, const de::_DATA_TYPES_FLAGS_ output_type)
{
    de::DH handle;
    decx::err::Success(&handle);

    if (!decx::cuda::_is_CUDA_init()) {
        
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _kernel = dynamic_cast<decx::_Matrix*>(&kernel);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::conv::Conv2_fp32<true>(_src, _kernel, _dst, conv_flag, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::conv::Conv2_fp16<true>(_src, _kernel, _dst, conv_flag, accu_flag, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::conv::Conv2_uint8<true>(_src, _kernel, _dst, conv_flag, output_type, &handle);
        break;
    default:
        break;
    }
    return handle;
}