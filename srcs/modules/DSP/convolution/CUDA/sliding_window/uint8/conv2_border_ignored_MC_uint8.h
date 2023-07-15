/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CONV2_BORDER_IGNORED_MC_UINT8_H_
#define _CONV2_BORDER_IGNORED_MC_UINT8_H_


#include "../conv2_SW_configs.h"
#include "conv2_border_ignored_uint8.h"
#include "../../../../classes/MatrixArray.h"


namespace decx
{
    namespace conv {
        /*
        * Should be called before k_params->src_confs, k_params->kernel_confs and k_params->dst_confs
        * are all configured !
        */
        template <bool _print, typename _src_type, typename _kernel_type, typename _dst_type>
        static void conv2_uint8_NB_SK_pre_process(decx::conv::_CCAC_uint8* _ccac, de::DH* handle);


        template <bool _print, bool _is_MK>
        static void conv2_uc8_uc8_NB_SK(decx::conv::_CCAC_uint8* _ccac, de::DH* handle);


        template <bool _print, bool _is_MK>
        static void conv2_uc8_fp32_NB_SK(decx::conv::_CCAC_uint8* _ccac, de::DH* handle);
    }
}


template <bool _print, typename _src_type, typename _kernel_type, typename _dst_type>
static void decx::conv::conv2_uint8_NB_SK_pre_process(decx::conv::_CCAC_uint8* _ccac, de::DH* handle)
{
    // generate the params for buffers
    decx::conv::_cuda_conv2_uint8_NB_buf_dims_config(&_ccac->_Kparams);

    const uint half_kerH = _ccac->_Kparams.ker_dims.y / 2;
    const uint half_kerW = _ccac->_Kparams.ker_dims.x / 2;

    // get cudaStream and cudaEvent
    _ccac->generate_stream_and_event<_print>(handle);
    // malloc buffers on device, and bind pointer to buffer stage
    _ccac->_conv2_MK_alloc<_print, sizeof(_src_type), sizeof(_dst_type), sizeof(_kernel_type)>(handle);
}



template <bool _print, bool _is_MK> static void 
decx::conv::conv2_uc8_uc8_NB_SK(decx::conv::_CCAC_uint8* _ccac, de::DH* handle)
{
    // generate every parameters convolution needs
    decx::conv::conv2_uint8_NB_SK_pre_process<_print, uint8_t, float, uint8_t>(_ccac, handle);

    _ccac->memcpy_src_async_H2D<0, _is_MK, float>(0);     // copy data from host to deivce, usinng stream[0] and event[0]
    if (!_is_MK) {
        _ccac->memcpy_kernel_async_H2D<0, float>(0);
    }
    _ccac->E[0]->synchronize();

    const uint _loop = _ccac->_Kparams._src_confs._matrix_num;
    // kernel pointers
    float4* read_ptr = NULL;
    float2* write_ptr = NULL;
    float* kernel_ptr = NULL;

    for (int i = 0; i < _loop; ++i) 
    {
        if (i > 0) {
            _ccac->memcpy_dst_async_D2H<2, uint8_t>(i - 1);
        }

        _ccac->get_kernel_read_write_ptr<_is_MK>((void**)&read_ptr, (void**)&write_ptr, (void**)&kernel_ptr);

        decx::conv::conv2_uc8_kfp32_kernels_caller(read_ptr, kernel_ptr, write_ptr,
                                                   _ccac->_Kparams.src_buf_dims.x / 16, 
                                                   _ccac->_Kparams.dst_dims.x / 8, 
                                                   _ccac->_Kparams.ker_dims,
                                                   _ccac->_Kparams.kernel_shift,
                                                   make_uint2(_ccac->_Kparams.dst_dims.x / 8, _ccac->_Kparams.dst_dims.y),
                                                   _ccac->S[1]);
        _ccac->E[1]->event_record(_ccac->S[1]);

        if (i < _loop - 1) {
            _ccac->memcpy_src_async_H2D<0, _is_MK, float>(i + 1);     // copy data from host to deivce, usinng stream[0] and event[0]
        }

        _ccac->synchronize_among_all_events();

        _ccac->after_kernel();
    }

    _ccac->memcpy_dst_async_D2H<2, uint8_t>(_loop - 1);

    _ccac->E[2]->synchronize();
}



template <bool _print, bool _is_MK> static void 
decx::conv::conv2_uc8_fp32_NB_SK(decx::conv::_CCAC_uint8* _ccac, de::DH* handle)
{
    // generate every parameters convolution needs
    decx::conv::conv2_uint8_NB_SK_pre_process<_print, uint8_t, float, float>(_ccac, handle);

    _ccac->memcpy_src_async_H2D<0, _is_MK, float>(0);     // copy data from host to deivce, usinng stream[0] and event[0]
    if (!_is_MK) {
        _ccac->memcpy_kernel_async_H2D<0, float>(0);
    }
    _ccac->E[0]->synchronize();

    const uint _loop = _ccac->_Kparams._src_confs._matrix_num;

    // kernel pointers
    float4* read_ptr = NULL;
    float4* write_ptr = NULL;
    float* kernel_ptr = NULL;

    for (int i = 0; i < _loop; ++i) 
    {
        if (i > 0) {
            _ccac->memcpy_dst_async_D2H<2, float>(i - 1);
        }

        _ccac->get_kernel_read_write_ptr<_is_MK>((void**)&read_ptr, (void**)&write_ptr, (void**)&kernel_ptr);

        decx::conv::conv2_uc8_fp32_kfp32_kernels_caller(read_ptr,
                                                   kernel_ptr,
                                                   write_ptr,
                                                   _ccac->_Kparams.src_buf_dims.x / 16, 
                                                   _ccac->_Kparams.dst_dims.x / 4, 
                                                   _ccac->_Kparams.ker_dims,
                                                   _ccac->_Kparams.kernel_shift,
                                                   make_uint2(_ccac->_Kparams.dst_dims.x / 8, _ccac->_Kparams.dst_dims.y),
                                                   _ccac->S[1]);
        _ccac->E[1]->event_record(_ccac->S[1]);

        if (i < _loop - 1) {
            _ccac->memcpy_src_async_H2D<0, _is_MK, float>(i + 1);     // copy data from host to deivce, usinng stream[0] and event[0]
        }

        _ccac->synchronize_among_all_events();

        _ccac->after_kernel();
    }

    _ccac->memcpy_dst_async_D2H<2, float>(_loop - 1);

    _ccac->E[2]->synchronize();
}


#endif