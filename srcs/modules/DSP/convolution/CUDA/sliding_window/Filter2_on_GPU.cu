/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "fp32/dev_conv2_border_const_fp32.h"
#include "fp32/dev_conv2_border_ignored_fp32.h"
#include "fp16/dev_conv2_border_const_fp16.h"
#include "fp16/dev_conv2_border_ignored_fp16.h"
#include "uint8/dev_conv2_border_const_uint8.h"
#include "uint8/dev_conv2_border_ignored_uint8.h"
#include "Filter2.h"
#include "../../../../BLAS/basic_process/extension/extend_flags.h"


namespace decx
{
    namespace conv
    {
        template <bool _print>
        void dev_Conv2_on_GPU_fp32(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel, decx::_GPU_Matrix* dst, const uint flag, de::DH* handle);


        template <bool _print>
        void dev_Conv2_on_GPU_fp16(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel, decx::_GPU_Matrix* dst,
            const uint conv_flag, const int accu_flag, de::DH* handle);


        template <bool _print>
        void dev_Conv2_on_GPU_uint8(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel, decx::_GPU_Matrix* dst,
            const int flag, const de::_DATA_TYPES_FLAGS_ output_type, de::DH* handle);
    }
}


template <bool _print>
void decx::conv::dev_Conv2_on_GPU_fp32(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel, decx::_GPU_Matrix* dst, const uint flag, de::DH* handle)
{
    switch (flag)
    {
    case decx::bp::extend_label::_EXTEND_NONE_:
        decx::conv::dev_conv2_fp32_border_ignore<_print>(src, kernel, dst, handle);
        break;

    case decx::bp::extend_label::_EXTEND_CONSTANT_:
        decx::conv::dev_conv2_fp32_border_zero<_print>(src, kernel, dst, handle);
        break;
    default:
        break;
    }
}


template <bool _print>
void decx::conv::dev_Conv2_on_GPU_fp16(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel, decx::_GPU_Matrix* dst,
    const uint conv_flag, const int accu_flag, de::DH* handle)
{
    switch (conv_flag)
    {
    case decx::bp::extend_label::_EXTEND_NONE_:
        decx::conv::dev_conv2_fp16_border_ignore<_print>(src, kernel, dst, handle, accu_flag);
        break;

    case decx::bp::extend_label::_EXTEND_CONSTANT_:
        decx::conv::dev_conv2_fp16_border_zero<_print>(src, kernel, dst, handle, accu_flag);
        break;
    default:
        break;
    }
}



template <bool _print>
void decx::conv::dev_Conv2_on_GPU_uint8(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel, decx::_GPU_Matrix* dst, 
    const int flag, const de::_DATA_TYPES_FLAGS_ output_type, de::DH* handle)
{
    decx::conv::_cuda_conv2_uc8_uc8_preset _conv2_preset;
    _conv2_preset.memcpy_flag = cudaMemcpyDeviceToDevice;
    _conv2_preset._Kparams._src_confs.gen_matrix_configs(src);
    _conv2_preset._Kparams._kernel_confs.gen_matrix_configs(kernel);

    switch (flag)
    {
    case decx::bp::extend_label::_EXTEND_NONE_:
        dst->re_construct(output_type, src->Width() - kernel->Width() + 1,
            src->Height() - kernel->Height() + 1);

        _conv2_preset._Kparams._dst_confs.gen_matrix_configs(dst);

        if (output_type == de::_DATA_TYPES_FLAGS_::_UINT8_) {
            decx::conv::dev_conv2_uc8_uc8_NB<_print>(&_conv2_preset, handle);
        }
        else {
            decx::conv::dev_conv2_uc8_fp32_NB<_print>(&_conv2_preset, handle);
        }
        break;
    case decx::bp::extend_label::_EXTEND_CONSTANT_:
        dst->re_construct(output_type, src->Width(), src->Height());

        _conv2_preset._Kparams._dst_confs.gen_matrix_configs(dst);

        if (output_type == de::_DATA_TYPES_FLAGS_::_UINT8_) {
            decx::conv::dev_conv2_uc8_uc8_BC<_print>(&_conv2_preset, handle);
        }
        else {
            decx::conv::dev_conv2_uc8_fp32_BC<_print>(&_conv2_preset, handle);
        }
        break;
    default:
        break;
    }

    _conv2_preset.release();
}



_DECX_API_
de::DH de::cuda::Filter2D(de::GPU_Matrix& src, de::GPU_Matrix& kernel, de::GPU_Matrix& dst,
    const int conv_flag, const int accu_flag, const de::_DATA_TYPES_FLAGS_ output_type)
{
    de::DH handle;
    decx::err::Success(&handle);

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Matrix* _kernel = dynamic_cast<decx::_GPU_Matrix*>(&kernel);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::conv::dev_Conv2_on_GPU_fp32<true>(_src, _kernel, _dst, conv_flag, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::conv::dev_Conv2_on_GPU_fp16<true>(_src, _kernel, _dst, conv_flag, accu_flag, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::conv::dev_Conv2_on_GPU_uint8<true>(_src, _kernel, _dst, conv_flag, output_type, &handle);
        break;
    default:
        break;
    }
    return handle;
}



 
void decx::cuda::dev_Filter2D_Raw_API(decx::_GPU_Matrix* _src, decx::_GPU_Matrix* _kernel, decx::_GPU_Matrix* _dst,
    const int conv_flag, const int accu_flag, const de::_DATA_TYPES_FLAGS_ output_type, de::DH* handle)
{
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return;
    }

    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::conv::dev_Conv2_on_GPU_fp32<true>(_src, _kernel, _dst, conv_flag, handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::conv::dev_Conv2_on_GPU_fp16<true>(_src, _kernel, _dst, conv_flag, accu_flag, handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::conv::dev_Conv2_on_GPU_uint8<true>(_src, _kernel, _dst, conv_flag, output_type, handle);
        break;
    default:
        break;
    }
}



_DECX_API_ void de::cuda::Filter2D_Async(de::GPU_Matrix& src, de::GPU_Matrix& kernel, de::GPU_Matrix& dst,
    const int conv_flag, const int accu_flag, const de::_DATA_TYPES_FLAGS_ output_type, de::DecxStream& S)
{
    decx::_GPU_Matrix* _src = dynamic_cast<decx::_GPU_Matrix*>(&src);
    decx::_GPU_Matrix* _kernel = dynamic_cast<decx::_GPU_Matrix*>(&kernel);
    decx::_GPU_Matrix* _dst = dynamic_cast<decx::_GPU_Matrix*>(&dst);

    de::DH handle;

    decx::async::register_async_task((uint64_t)S.Get_ID(), decx::cuda::dev_Filter2D_Raw_API, _src, _kernel, _dst, conv_flag, accu_flag, output_type, S.Get_last_handle());
}