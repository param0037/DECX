/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _DEV_CONV2_BORDER_CONST_UINT8_H_
#define _DEV_CONV2_BORDER_CONST_UINT8_H_


#include "conv2_border_const_uint8.h"
#include "../../../../../classes/GPU_Matrix.h"


namespace decx
{
    namespace conv 
    {
        template<bool _print>
        static void dev_conv2_uc8_uc8_BC(decx::conv::_cuda_conv2_uc8_uc8_preset* _conv2_preset, de::DH* handle);


        template<bool _print>
        static void dev_conv2_uc8_fp32_BC(decx::conv::_cuda_conv2_uc8_uc8_preset* _conv2_preset, de::DH* handle);
    }
}



template<bool _print>
static void decx::conv::dev_conv2_uc8_uc8_BC(decx::conv::_cuda_conv2_uc8_uc8_preset* _conv2_preset, de::DH* handle)
{
    decx::conv::_conv2_uint8_BC_pre_process<_print, uint8_t>(_conv2_preset, handle);

    decx::conv::conv2_uc8_kfp32_kernels_caller(
        (float4*)_conv2_preset->src_buf.ptr,
        (float*)_conv2_preset->ker_buf.ptr,
        (float2*)_conv2_preset->_Kparams._dst_confs._ptr,
        _conv2_preset->_Kparams.src_buf_dims.x / 16,
        _conv2_preset->_Kparams.dst_dims.x / 8,
        _conv2_preset->_Kparams.ker_dims,
        _conv2_preset->_Kparams.kernel_shift,
        make_uint2(_conv2_preset->_Kparams.dst_dims.x / 8, _conv2_preset->_Kparams.dst_dims.y),
        _conv2_preset->S);

    _conv2_preset->E->event_record(_conv2_preset->S);
    _conv2_preset->E->synchronize();
}




template<bool _print>
static void decx::conv::dev_conv2_uc8_fp32_BC(decx::conv::_cuda_conv2_uc8_uc8_preset* _conv2_preset, de::DH* handle)
{
    decx::conv::_conv2_uint8_BC_pre_process<_print, float>(_conv2_preset, handle);

    decx::conv::conv2_uc8_fp32_kfp32_kernels_caller(
        (float4*)_conv2_preset->src_buf.ptr,
        (float*)_conv2_preset->ker_buf.ptr,
        (float4*)_conv2_preset->_Kparams._dst_confs._ptr,
        _conv2_preset->_Kparams.src_buf_dims.x / 16,
        _conv2_preset->_Kparams.dst_dims.x / 4,
        _conv2_preset->_Kparams.ker_dims,
        _conv2_preset->_Kparams.kernel_shift,
        make_uint2(_conv2_preset->_Kparams.dst_dims.x / 8, _conv2_preset->_Kparams.dst_dims.y),
        _conv2_preset->S);

    _conv2_preset->E->event_record(_conv2_preset->S);
    _conv2_preset->E->synchronize();
}


#endif