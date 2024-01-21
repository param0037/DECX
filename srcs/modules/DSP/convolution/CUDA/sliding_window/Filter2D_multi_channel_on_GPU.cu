/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "uint8/dev_conv2_border_const_MC_uint8.h"
#include "uint8/dev_conv2_border_ignored_MC_uint8.h"
#include "fp32/dev_conv2_border_ignored_MC_fp32.h"
#include "fp32/dev_conv2_border_const_MC_fp32.h"
#include "fp16/dev_conv2_border_ignored_MC_fp16.h"
#include "fp16/dev_conv2_border_const_MC_fp16.h"
#include "Filter2.h"
#include "../../../../BLAS/basic_process/extension/extend_flags.h"


namespace decx
{
    namespace conv {
        template <bool _print>
        static void dev_filter2D_uc8_SK_organiser(decx::_GPU_MatrixArray* src, decx::_GPU_Matrix* kernel, decx::_GPU_MatrixArray* dst, 
            const int conv_flag, const de::_DATA_TYPES_FLAGS_ output_type, de::DH* handle);


        template <bool _print>
        static void dev_filter2D_uc8_MK_organiser(decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
            const int conv_flag, const de::_DATA_TYPES_FLAGS_ output_type, de::DH* handle);


        template <bool _print>
        static void dev_filter2D_fp32_SK_organiser(decx::_GPU_MatrixArray* src, decx::_GPU_Matrix* kernel, decx::_GPU_MatrixArray* dst,
            const int conv_flag, de::DH* handle);


        template <bool _print>
        static void dev_filter2D_fp32_MK_organiser(decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
            const int conv_flag, de::DH* handle);


        template <bool _print>
        static void dev_filter2D_fp16_SK_organiser(decx::_GPU_MatrixArray* src, decx::_GPU_Matrix* kernel, decx::_GPU_MatrixArray* dst,
            const int conv_flag, const int _accu_flag, de::DH* handle);


        template <bool _print>
        static void dev_filter2D_fp16_MK_organiser(decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
            const int conv_flag, const int _accu_flag, de::DH* handle);
    }
}


template <bool _print> static void 
decx::conv::dev_filter2D_uc8_SK_organiser(decx::_GPU_MatrixArray* src, decx::_GPU_Matrix* kernel, decx::_GPU_MatrixArray* dst, 
    const int conv_flag, const de::_DATA_TYPES_FLAGS_ output_type, de::DH* handle)
{
    decx::conv::_cuda_conv2_uc8_uc8_preset _conv2_preset;
    _conv2_preset.memcpy_flag = cudaMemcpyDeviceToDevice;
    _conv2_preset._Kparams._src_confs.gen_matrix_configs(src);
    _conv2_preset._Kparams._kernel_confs.gen_matrix_configs(kernel);

    switch (conv_flag)
    {
    case decx::bp::extend_label::_EXTEND_NONE_:
        dst->re_construct(output_type, src->Width() - kernel->Width() + 1,
            src->Height() - kernel->Height() + 1, src->MatrixNumber());

        _conv2_preset._Kparams._dst_confs.gen_matrix_configs(dst);

        if (output_type == de::_DATA_TYPES_FLAGS_::_UINT8_) {
            decx::conv::dev_conv2_uc8_uc8_NB_SK<_print, false>(&_conv2_preset, handle);
        }
        else {
            decx::conv::dev_conv2_uc8_fp32_NB_SK<_print, false>(&_conv2_preset, handle);
        }
        break;

    case decx::bp::extend_label::_EXTEND_CONSTANT_:
        dst->re_construct(output_type, src->Width(), src->Height(), src->MatrixNumber());

        _conv2_preset._Kparams._dst_confs.gen_matrix_configs(dst);

        if (output_type == de::_DATA_TYPES_FLAGS_::_UINT8_) {
            decx::conv::dev_conv2_uc8_uc8_BC_SK<_print, false>(&_conv2_preset, handle);
        }
        else {
            decx::conv::dev_conv2_uc8_fp32_BC_SK<_print, false>(&_conv2_preset, handle);
        }
        break;

    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag, MEANINGLESS_FLAG);
        break;
    }

    _conv2_preset.release();
}




template <bool _print> static void 
decx::conv::dev_filter2D_uc8_MK_organiser(decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst, 
    const int conv_flag, const de::_DATA_TYPES_FLAGS_ output_type, de::DH* handle)
{
    decx::conv::_cuda_conv2_uc8_uc8_preset _conv2_preset;
    _conv2_preset.memcpy_flag = cudaMemcpyDeviceToDevice;
    _conv2_preset._Kparams._src_confs.gen_matrix_configs(src);
    _conv2_preset._Kparams._kernel_confs.gen_matrix_configs(kernel);

    switch (conv_flag)
    {
    case decx::bp::extend_label::_EXTEND_NONE_:
        dst->re_construct(output_type, src->Width() - kernel->Width() + 1,
            src->Height() - kernel->Height() + 1, src->MatrixNumber());

        _conv2_preset._Kparams._dst_confs.gen_matrix_configs(dst);

        if (output_type == de::_DATA_TYPES_FLAGS_::_UINT8_) {
            decx::conv::dev_conv2_uc8_uc8_NB_SK<_print, true>(&_conv2_preset, handle);
        }
        else {
            decx::conv::dev_conv2_uc8_fp32_NB_SK<_print, true>(&_conv2_preset, handle);
        }
        break;

    case decx::bp::extend_label::_EXTEND_CONSTANT_:
        dst->re_construct(output_type, src->Width(), src->Height(), src->MatrixNumber());

        _conv2_preset._Kparams._dst_confs.gen_matrix_configs(dst);

        if (output_type == de::_DATA_TYPES_FLAGS_::_UINT8_) {
            decx::conv::dev_conv2_uc8_uc8_BC_SK<_print, true>(&_conv2_preset, handle);
        }
        else {
            decx::conv::dev_conv2_uc8_fp32_BC_SK<_print, true>(&_conv2_preset, handle);
        }
        break;

    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag, MEANINGLESS_FLAG);
        break;
    }

    _conv2_preset.release();
}





template <bool _print> static void 
decx::conv::dev_filter2D_fp32_SK_organiser(decx::_GPU_MatrixArray* src, decx::_GPU_Matrix* kernel, decx::_GPU_MatrixArray* dst, 
    const int conv_flag, de::DH* handle)
{
    switch (conv_flag)
    {
    case decx::bp::extend_label::_EXTEND_NONE_:
        dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP32_, src->Width() - kernel->Width() + 1,
            src->Height() - kernel->Height() + 1, src->MatrixNumber());

        decx::conv::dev_conv2_fp32_NB_SK<_print>(src, kernel, dst, handle);
        break;

    case decx::bp::extend_label::_EXTEND_CONSTANT_:
        dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP32_, src->Width(), src->Height(), src->MatrixNumber());

        decx::conv::dev_conv2_fp32_BC_SK<_print>(src, kernel, dst, handle);
        break;

    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag, MEANINGLESS_FLAG);
        break;
    }
}




template <bool _print> static void
decx::conv::dev_filter2D_fp32_MK_organiser(decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
    const int conv_flag, de::DH* handle)
{
    switch (conv_flag)
    {
    case decx::bp::extend_label::_EXTEND_NONE_:
        dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP32_, src->Width() - kernel->Width() + 1,
            src->Height() - kernel->Height() + 1, src->MatrixNumber());

        decx::conv::dev_conv2_fp32_NB_MK<_print>(src, kernel, dst, handle);
        break;

    case decx::bp::extend_label::_EXTEND_CONSTANT_:
        dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP32_, src->Width(), src->Height(), src->MatrixNumber());

        decx::conv::dev_conv2_fp32_BC_MK<_print>(src, kernel, dst, handle);
        break;
    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag, MEANINGLESS_FLAG);
        break;
    }
}



template <bool _print> static void
decx::conv::dev_filter2D_fp16_SK_organiser(decx::_GPU_MatrixArray* src, decx::_GPU_Matrix* kernel, decx::_GPU_MatrixArray* dst,
    const int conv_flag, const int _accu_flag, de::DH* handle)
{
    switch (conv_flag)
    {
    case decx::bp::extend_label::_EXTEND_NONE_:
        dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP16_, src->Width() - kernel->Width() + 1,
            src->Height() - kernel->Height() + 1, src->MatrixNumber());

        decx::conv::dev_conv2_fp16_NB_SK<_print>(src, kernel, dst, _accu_flag, handle);
        break;

    case decx::bp::extend_label::_EXTEND_CONSTANT_:
        dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP16_, src->Width(), src->Height(), src->MatrixNumber());

        decx::conv::dev_conv2_fp16_BC_SK<_print>(src, kernel, dst, _accu_flag, handle);
        break;

    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag, MEANINGLESS_FLAG);
        break;
    }
}




template <bool _print> static void
decx::conv::dev_filter2D_fp16_MK_organiser(decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
    const int conv_flag, const int _accu_flag, de::DH* handle)
{
    switch (conv_flag)
    {
    case decx::bp::extend_label::_EXTEND_NONE_:
        dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP16_, src->Width() - kernel->Width() + 1,
            src->Height() - kernel->Height() + 1, src->MatrixNumber());

        decx::conv::dev_conv2_fp16_NB_MK<_print>(src, kernel, dst, _accu_flag, handle);
        break;

    case decx::bp::extend_label::_EXTEND_CONSTANT_:
        dst->re_construct(de::_DATA_TYPES_FLAGS_::_FP16_, src->Width(), src->Height(), src->MatrixNumber());

        decx::conv::dev_conv2_fp16_BC_MK<_print>(src, kernel, dst, _accu_flag, handle);
        break;

    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ErrorFlag, MEANINGLESS_FLAG);
        break;
    }
}



_DECX_API_ de::DH
de::cuda::Filter2D_multi_channel(de::GPU_MatrixArray& src, de::GPU_Matrix& kernel, de::GPU_MatrixArray& dst, 
    const int conv_flag, const int accu_flag, const de::_DATA_TYPES_FLAGS_ output_type)
{
    de::DH handle;
    decx::err::Success<true>(&handle);

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_MatrixArray* _src = dynamic_cast<decx::_GPU_MatrixArray*>(&src);
    decx::_GPU_Matrix* _kernel = dynamic_cast<decx::_GPU_Matrix*>(&kernel);
    decx::_GPU_MatrixArray* _dst = dynamic_cast<decx::_GPU_MatrixArray*>(&dst);

    switch (_src->type)
    {
    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::conv::dev_filter2D_uc8_SK_organiser<true>(_src, _kernel, _dst, conv_flag, output_type, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::conv::dev_filter2D_fp32_SK_organiser<true>(_src, _kernel, _dst, conv_flag, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::conv::dev_filter2D_fp16_SK_organiser<true>(_src, _kernel, _dst, conv_flag, accu_flag, &handle);
        break;

    default:
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE, UNSUPPORTED_TYPE);
        break;
    }

    return handle;
}




_DECX_API_ de::DH
de::cuda::Filter2D_multi_channel(de::GPU_MatrixArray& src, de::GPU_MatrixArray& kernel, de::GPU_MatrixArray& dst, 
    const int conv_flag, const int accu_flag, const de::_DATA_TYPES_FLAGS_ output_type)
{
    de::DH handle;
    decx::err::Success<true>(&handle);

    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return handle;
    }

    decx::_GPU_MatrixArray* _src = dynamic_cast<decx::_GPU_MatrixArray*>(&src);
    decx::_GPU_MatrixArray* _kernel = dynamic_cast<decx::_GPU_MatrixArray*>(&kernel);
    decx::_GPU_MatrixArray* _dst = dynamic_cast<decx::_GPU_MatrixArray*>(&dst);

    switch (_src->type)
    {
    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::conv::dev_filter2D_uc8_MK_organiser<true>(_src, _kernel, _dst, conv_flag, output_type, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::conv::dev_filter2D_fp32_MK_organiser<true>(_src, _kernel, _dst, conv_flag, &handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::conv::dev_filter2D_fp16_MK_organiser<true>(_src, _kernel, _dst, conv_flag, accu_flag, &handle);
        break;

    default:
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE, UNSUPPORTED_TYPE);
        break;
    }

    return handle;
}




void decx::cuda::dev_Filter2D_MC_SK_Raw_API(decx::_GPU_MatrixArray* src, decx::_GPU_Matrix* kernel, decx::_GPU_MatrixArray* dst,
    const int conv_flag, const int accu_flag, const de::_DATA_TYPES_FLAGS_ output_type, de::DH* handle)
{
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return;
    }

    switch (src->type)
    {
    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::conv::dev_filter2D_uc8_SK_organiser<false>(src, kernel, dst, conv_flag, output_type, handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::conv::dev_filter2D_fp32_SK_organiser<false>(src, kernel, dst, conv_flag, handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::conv::dev_filter2D_fp16_SK_organiser<false>(src, kernel, dst, conv_flag, accu_flag, handle);
        break;

    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE, UNSUPPORTED_TYPE);
        break;
    }

    if (handle->error_type == decx::DECX_error_types::DECX_SUCCESS) {
        decx::err::Success<false>(handle);
    }
}




void decx::cuda::dev_Filter2D_MC_MK_Raw_API(decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
    const int conv_flag, const int accu_flag, const de::_DATA_TYPES_FLAGS_ output_type, de::DH* handle)
{
    if (!decx::cuda::_is_CUDA_init()) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_not_init, CUDA_NOT_INIT);
        return;
    }

    switch (src->type)
    {
    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::conv::dev_filter2D_uc8_MK_organiser<false>(src, kernel, dst, conv_flag, output_type, handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::conv::dev_filter2D_fp32_MK_organiser<false>(src, kernel, dst, conv_flag, handle);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP16_:
        decx::conv::dev_filter2D_fp16_MK_organiser<false>(src, kernel, dst, conv_flag, accu_flag, handle);
        break;

    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE, UNSUPPORTED_TYPE);
        break;
    }

    if (handle->error_type == decx::DECX_error_types::DECX_SUCCESS) {
        decx::err::Success<false>(handle);
    }
}