/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _CONV2_BORDER_IGNORED_UINT8_H_
#define _CONV2_BORDER_IGNORED_UINT8_H_


#include "conv2_uint8_kernel_callers.h"
#include "../../../../core/cudaStream_management/cudaStream_package.h"
#include "../../../../core/cudaStream_management/cudaEvent_package.h"
#include "../../../../classes/Matrix.h"
#include "../../../conv_utils.h"



namespace decx
{
    namespace conv 
    {
        template <bool _print, typename _dst_type, bool _for_MK = false>
        static void _conv2_uint8_NB_pre_process(decx::conv::_cuda_conv2_uc8_uc8_preset* _conv2_preset, de::DH* handle);


        template<bool _print>
        static void conv2_uc8_uc8_NB(decx::conv::_cuda_conv2_uc8_uc8_preset* _conv2_preset, de::DH* handle);


        template<bool _print>
        static void conv2_uc8_fp32_NB(decx::conv::_cuda_conv2_uc8_uc8_preset* _conv2_preset, de::DH* handle);
    }
}



template <bool _print, typename _dst_type, bool _for_MK>
static void decx::conv::_conv2_uint8_NB_pre_process(decx::conv::_cuda_conv2_uc8_uc8_preset* _conv2_preset, de::DH* handle)
{
    decx::conv::_cuConv2_kernel_params* k_params = &_conv2_preset->_Kparams;

    decx::conv::_cuda_conv2_uint8_NB_buf_dims_config(k_params);

    const uint half_kerH = k_params->ker_dims.y / 2;
    const uint half_kerW = k_params->ker_dims.x / 2;

    _conv2_preset->S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (_conv2_preset->S == NULL) {
        decx::err::CUDA_Stream_access_fail<_print>(handle);
        return;
    }
    _conv2_preset->E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (_conv2_preset->E == NULL) {
        decx::err::CUDA_Event_access_fail<_print>(handle);
        return;
    }

    _conv2_preset->_cuda_conv2_malloc<_print, _dst_type, float>(handle);

    if (!_for_MK) {
        _conv2_preset->_cuda_conv2_memcpyH2D<float>();          // copy the datas of src from host to device
    }
}



template<bool _print>
static void decx::conv::conv2_uc8_uc8_NB(decx::conv::_cuda_conv2_uc8_uc8_preset* _conv2_preset, de::DH* handle)
{
    decx::conv::_conv2_uint8_NB_pre_process<_print, uint8_t>(_conv2_preset, handle);

    decx::conv::conv2_uc8_kfp32_kernels_caller(
        (float4*)_conv2_preset->src_buf.ptr,
        (float*)_conv2_preset->ker_buf.ptr,
        (float2*)_conv2_preset->dst_buf.ptr,
        _conv2_preset->_Kparams.src_buf_dims.x / 16,
        _conv2_preset->_Kparams.dst_dims.x / 8,
        _conv2_preset->_Kparams.ker_dims,
        _conv2_preset->_Kparams.kernel_shift,
        make_uint2(_conv2_preset->_Kparams.dst_dims.x / 8, _conv2_preset->_Kparams.dst_dims.y),
        _conv2_preset->S);

    _conv2_preset->_cuda_conv2_memcpyD2H<uint8_t>();

    _conv2_preset->E->event_record(_conv2_preset->S);
    _conv2_preset->E->synchronize();
}




template<bool _print>
static void decx::conv::conv2_uc8_fp32_NB(decx::conv::_cuda_conv2_uc8_uc8_preset* _conv2_preset, de::DH* handle)
{
    decx::conv::_conv2_uint8_NB_pre_process<_print, float>(_conv2_preset, handle);
    
    decx::conv::conv2_uc8_fp32_kfp32_kernels_caller(
        (float4*)_conv2_preset->src_buf.ptr,
        (float*)_conv2_preset->ker_buf.ptr,
        (float4*)_conv2_preset->dst_buf.ptr,
        _conv2_preset->_Kparams.src_buf_dims.x / 16, 
        _conv2_preset->_Kparams.dst_dims.x / 4, 
        _conv2_preset->_Kparams.ker_dims,
        _conv2_preset->_Kparams.kernel_shift,
        make_uint2(_conv2_preset->_Kparams.dst_dims.x / 8, _conv2_preset->_Kparams.dst_dims.y),
        _conv2_preset->S);
    
    _conv2_preset->_cuda_conv2_memcpyD2H<float>();

    _conv2_preset->E->event_record(_conv2_preset->S);
    _conv2_preset->E->synchronize();
}



#endif