/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _DEV_CONV2_BORDER_IGNORED_MC_UINT8_H_
#define _DEV_CONV2_BORDER_IGNORED_MC_UINT8_H_


#include "../conv2_SW_configs.h"
#include "conv2_border_ignored_uint8.h"
#include "../../../../../classes/MatrixArray.h"


namespace decx
{
    namespace conv {
        template <bool _print, bool _is_MK>
        static void dev_conv2_uc8_uc8_NB_SK(decx::conv::_cuda_conv2_uc8_uc8_preset* _ccac, de::DH* handle);


        template <bool _print, bool _is_MK>
        static void dev_conv2_uc8_fp32_NB_SK(decx::conv::_cuda_conv2_uc8_uc8_preset* _ccac, de::DH* handle);
    }
}


template <bool _print, bool _is_MK> static void 
decx::conv::dev_conv2_uc8_uc8_NB_SK(decx::conv::_cuda_conv2_uc8_uc8_preset* _conv2_preset, de::DH* handle)
{
    // generate every parameters convolution needs
    decx::conv::_conv2_uint8_NB_pre_process<_print, uint8_t, true>(_conv2_preset, handle);

    const uint _loop = _conv2_preset->_Kparams._src_confs._matrix_num;

    if (!_is_MK) {
        _conv2_preset->_cuda_conv2_MC_kernel_memcpy_from_host<_is_MK, float>(0);
    }

    for (int i = 0; i < _loop; ++i) 
    {
        // copy data from host to deivce, usinng stream[0] and event[0]
        _conv2_preset->_cuda_conv2_MC_src_memcpy_from_host(i);
        if (_is_MK) {
            _conv2_preset->_cuda_conv2_MC_kernel_memcpy_from_host<_is_MK, float>(i);
        }
        
        decx::conv::conv2_uc8_kfp32_kernels_caller((float4*)_conv2_preset->src_buf.ptr,
                                                   (float*)_conv2_preset->ker_buf.ptr,
                                                   (float2*)_conv2_preset->_Kparams._dst_confs._ptr_array[i],
                                                   _conv2_preset->_Kparams.src_buf_dims.x / 16, 
                                                   _conv2_preset->_Kparams.dst_dims.x / 8, 
                                                   _conv2_preset->_Kparams.ker_dims,
                                                   _conv2_preset->_Kparams.kernel_shift,
                                                   make_uint2(_conv2_preset->_Kparams.dst_dims.x / 8, _conv2_preset->_Kparams.dst_dims.y),
                                                   _conv2_preset->S);
    }

    _conv2_preset->E->event_record(_conv2_preset->S);
    _conv2_preset->E->synchronize();
}




template <bool _print, bool _is_MK> static void 
decx::conv::dev_conv2_uc8_fp32_NB_SK(decx::conv::_cuda_conv2_uc8_uc8_preset* _conv2_preset, de::DH* handle)
{
    // generate every parameters convolution needs
    decx::conv::_conv2_uint8_NB_pre_process<_print, float>(_conv2_preset, handle);

    const uint _loop = _conv2_preset->_Kparams._src_confs._matrix_num;

    if (!_is_MK) {
        _conv2_preset->_cuda_conv2_MC_kernel_memcpy_from_host<_is_MK, float>(0);
    }

    for (int i = 0; i < _loop; ++i) 
    {
        // copy data from host to deivce, usinng stream[0] and event[0]
        _conv2_preset->_cuda_conv2_MC_src_memcpy_from_host(i);
        if (_is_MK) {
            _conv2_preset->_cuda_conv2_MC_kernel_memcpy_from_host<_is_MK, float>(i);
        }
        
        decx::conv::conv2_uc8_fp32_kfp32_kernels_caller((float4*)_conv2_preset->src_buf.ptr,
                                                   (float*)_conv2_preset->ker_buf.ptr,
                                                   (float4*)_conv2_preset->_Kparams._dst_confs._ptr_array[i],
                                                   _conv2_preset->_Kparams.src_buf_dims.x / 16, 
                                                   _conv2_preset->_Kparams.dst_dims.x / 4, 
                                                   _conv2_preset->_Kparams.ker_dims,
                                                   _conv2_preset->_Kparams.kernel_shift,
                                                   make_uint2(_conv2_preset->_Kparams.dst_dims.x / 8, _conv2_preset->_Kparams.dst_dims.y),
                                                   _conv2_preset->S);
    }

    _conv2_preset->E->event_record(_conv2_preset->S);
    _conv2_preset->E->synchronize();
}




#endif