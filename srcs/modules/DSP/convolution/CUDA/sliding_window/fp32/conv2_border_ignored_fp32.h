/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CONV2_BORDER_IGNORED_H_
#define _CONV2_BORDER_IGNORED_H_

#include "../../../../../core/basic.h"
#include "sconv2_kernel_callers.h"
#include "../conv2_SW_configs.h"


namespace decx
{
    namespace conv {
        template <bool _print, uint _bound_H, uint _bound_W, bool _for_MK = false>
        static void _conv2_fp32_NB_pre_process(decx::conv::_cuda_conv2_fp32_preset* _conv2_preset, de::DH* handle);


        template <bool _print>
        /**
        * 8 x 8
        * This function complete a comand of a convolutipon kernel, which is in type of border-ignored
        * In order to save the device memory, just allocate memory of suitable size
        */
        static void _Conv2_NB_R8x8(decx::conv::_cuda_conv2_fp32_preset *_conv2_preset, de::DH* handle);


        template <bool _print>
        /*\
        * 16 x 16
        * This function complete a comand of a convolutipon kernel, which is in type of border-ignored
        * In order to save the device memory, just allocate memory of suitable size
        */
        static void _Conv2_NB_R16x16(decx::conv::_cuda_conv2_fp32_preset *_conv2_preset, de::DH* handle);


        template <bool _print>
        /*\
        * 8 x 16
        * This function complete a comand of a convolutipon kernel, which is in type of border-ignored
        * In order to save the device memory, just allocate memory of suitable size
        */
        static void _Conv2_NB_R8x16(decx::conv::_cuda_conv2_fp32_preset *_conv2_preset, de::DH* handle);


        template <bool _print>
        /*\
        * 16 x 8
        * This function complete a comand of a convolutipon kernel, which is in type of border-ignored
        * In order to save the device memory, just allocate memory of suitable size
        */
        static void _Conv2_NB_R16x8(decx::conv::_cuda_conv2_fp32_preset *_conv2_preset, de::DH* handle);


        template <bool _print>
        static void conv2_fp32_border_ignore(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH* handle);
    }
}


template <bool _print, uint _bound_H, uint _bound_W, bool _for_MK>
static void decx::conv::_conv2_fp32_NB_pre_process(decx::conv::_cuda_conv2_fp32_preset*     _conv2_preset, 
                                                   de::DH*                                  handle)
{
    decx::conv::_cuConv2_kernel_params* k_params = &_conv2_preset->_Kparams;

    decx::conv::_cuda_conv2_fp32_NB_buf_dims_config<_bound_H, _bound_W>(k_params);

    _conv2_preset->S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (_conv2_preset->S == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_STREAM, CUDA_STREAM_ACCESS_FAIL);
        return;
    }

    _conv2_preset->E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (_conv2_preset->E == NULL) {
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_CUDA_EVENT, CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    _conv2_preset->_cuda_conv2_malloc<_print, float, float>(handle);
    if (!_for_MK) {
        _conv2_preset->_cuda_conv2_memcpyH2D<float>();
    }
}



template <bool _print>
static void decx::conv::_Conv2_NB_R8x8(decx::conv::_cuda_conv2_fp32_preset *_conv2_preset, de::DH* handle)
{
    decx::conv::_conv2_fp32_NB_pre_process<_print, bounded_kernel_R8, bounded_kernel_R8>(_conv2_preset, handle);

    decx::conv::conv2_fp32_kernel_r8x8((float4*)_conv2_preset->src_buf.ptr, (float*)_conv2_preset->ker_buf.ptr, (float4*)_conv2_preset->dst_buf.ptr,
        make_uint2(_conv2_preset->_Kparams.kernel_shift.x, _conv2_preset->_Kparams.kernel_shift.y),
        make_uint2(_conv2_preset->_Kparams.src_buf_dims.x / 4, _conv2_preset->_Kparams.src_buf_dims.y),
        make_uint2(_conv2_preset->_Kparams.dst_dims.x / 4, _conv2_preset->_Kparams.dst_dims.y),
        _conv2_preset->_Kparams.ker_dims, _conv2_preset->S);

    _conv2_preset->_cuda_conv2_memcpyD2H<float>();

    _conv2_preset->E->event_record(_conv2_preset->S);
    _conv2_preset->E->synchronize();
}


    
template <bool _print>
static void decx::conv::_Conv2_NB_R16x16(decx::conv::_cuda_conv2_fp32_preset *_conv2_preset, de::DH* handle)
{
    decx::conv::_conv2_fp32_NB_pre_process<_print, bounded_kernel_R16, bounded_kernel_R16>(_conv2_preset, handle);

    decx::conv::conv2_fp32_kernel_r16x16((float4*)_conv2_preset->src_buf.ptr, (float*)_conv2_preset->ker_buf.ptr, (float4*)_conv2_preset->dst_buf.ptr,
        make_uint2(_conv2_preset->_Kparams.kernel_shift.x,       _conv2_preset->_Kparams.kernel_shift.y),
        make_uint2(_conv2_preset->_Kparams.src_buf_dims.x / 4,   _conv2_preset->_Kparams.src_buf_dims.y),
        make_uint2(_conv2_preset->_Kparams.dst_dims.x / 4,       _conv2_preset->_Kparams.dst_dims.y),
        _conv2_preset->_Kparams.ker_dims, _conv2_preset->S);

    _conv2_preset->_cuda_conv2_memcpyD2H<float>();

    _conv2_preset->E->event_record(_conv2_preset->S);
    _conv2_preset->E->synchronize();
}



template <bool _print>
static void decx::conv::_Conv2_NB_R8x16(decx::conv::_cuda_conv2_fp32_preset *_conv2_preset, de::DH* handle)
{
    decx::conv::_conv2_fp32_NB_pre_process<_print, bounded_kernel_R8, bounded_kernel_R16>(_conv2_preset, handle);

    decx::conv::conv2_fp32_kernel_r8x16((float4*)_conv2_preset->src_buf.ptr, (float*)_conv2_preset->ker_buf.ptr, (float4*)_conv2_preset->dst_buf.ptr,
        make_uint2(_conv2_preset->_Kparams.kernel_shift.x,       _conv2_preset->_Kparams.kernel_shift.y),
        make_uint2(_conv2_preset->_Kparams.src_buf_dims.x / 4,   _conv2_preset->_Kparams.src_buf_dims.y),
        make_uint2(_conv2_preset->_Kparams.dst_dims.x / 4,       _conv2_preset->_Kparams.dst_dims.y),
        _conv2_preset->_Kparams.ker_dims, _conv2_preset->S);

    _conv2_preset->_cuda_conv2_memcpyD2H<float>();

    _conv2_preset->E->event_record(_conv2_preset->S);
    _conv2_preset->E->synchronize();
}



template <bool _print>
static void decx::conv::_Conv2_NB_R16x8(decx::conv::_cuda_conv2_fp32_preset *_conv2_preset, de::DH* handle)
{
    decx::conv::_conv2_fp32_NB_pre_process<_print, bounded_kernel_R16, bounded_kernel_R8>(_conv2_preset, handle);

    decx::conv::conv2_fp32_kernel_r16x8((float4*)_conv2_preset->src_buf.ptr, (float*)_conv2_preset->ker_buf.ptr, (float4*)_conv2_preset->dst_buf.ptr,
        make_uint2(_conv2_preset->_Kparams.kernel_shift.x,       _conv2_preset->_Kparams.kernel_shift.y),
        make_uint2(_conv2_preset->_Kparams.src_buf_dims.x / 4,   _conv2_preset->_Kparams.src_buf_dims.y),
        make_uint2(_conv2_preset->_Kparams.dst_dims.x / 4,       _conv2_preset->_Kparams.dst_dims.y),
        _conv2_preset->_Kparams.ker_dims, _conv2_preset->S);

    _conv2_preset->_cuda_conv2_memcpyD2H<float>();

    _conv2_preset->E->event_record(_conv2_preset->S);
    _conv2_preset->E->synchronize();
}



template <bool _print>
static void decx::conv::conv2_fp32_border_ignore(decx::_Matrix* src, decx::_Matrix* kernel, decx::_Matrix* dst, de::DH* handle)
{
    int2 half_ker_dim;
    half_ker_dim.x = kernel->Width() / 2;                half_ker_dim.y = kernel->Height() / 2;

    dst->re_construct(src->Type(), src->Width() - (half_ker_dim.x * 2),
        src->Height() - (half_ker_dim.y * 2));

    decx::conv::_cuda_conv2_fp32_preset _conv2_preset;
    _conv2_preset.memcpy_flag = cudaMemcpyHostToDevice;
    _conv2_preset._Kparams._src_confs.gen_matrix_configs(src);
    _conv2_preset._Kparams._kernel_confs.gen_matrix_configs(kernel);
    _conv2_preset._Kparams._dst_confs.gen_matrix_configs(dst);
    
    if (half_ker_dim.x < bounded_kernel_R8 + 1) {
        if (half_ker_dim.y < bounded_kernel_R8 + 1) {
            decx::conv::_Conv2_NB_R8x8<_print>(&_conv2_preset, handle);
        }
        else {
            decx::conv::_Conv2_NB_R16x8<_print>(&_conv2_preset, handle);
        }
    }
    else {
        if (half_ker_dim.y < bounded_kernel_R8 + 1) {
            decx::conv::_Conv2_NB_R8x16<_print>(&_conv2_preset, handle);
        }
        else {
            decx::conv::_Conv2_NB_R16x16<_print>(&_conv2_preset, handle);
        }
    }

    _conv2_preset.release();
}


#endif