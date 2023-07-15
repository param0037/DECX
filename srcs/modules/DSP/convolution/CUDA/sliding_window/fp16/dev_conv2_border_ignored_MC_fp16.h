/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _DEV_CONV2_BORDER_IGNORED_MC_FP16_H_
#define _DEV_CONV2_BORDER_IGNORED_MC_FP16_H_


#include "../conv2_SW_configs.h"
#include "conv2_border_ignore_fp16.h"
#include "../../../../classes/MatrixArray.h"


namespace decx
{
    namespace conv 
    {
        template <bool _print, bool _is_MK>
        static void dev_conv2_fp16_NB_SK_R8x8(decx::conv::_cuda_conv2_fp16_preset *_ccac, const int _accu_flag, de::DH* handle);


        template <bool _print, bool _is_MK>
        static void dev_conv2_fp16_NB_SK_R8x16(decx::conv::_cuda_conv2_fp16_preset *_ccac, const int _accu_flag, de::DH* handle);


        template <bool _print, bool _is_MK>
        static void dev_conv2_fp16_NB_SK_R16x8(decx::conv::_cuda_conv2_fp16_preset *_ccac, const int _accu_flag, de::DH* handle);


        template <bool _print, bool _is_MK>
        static void dev_conv2_fp16_NB_SK_R16x16(decx::conv::_cuda_conv2_fp16_preset *_ccac, const int _accu_flag, de::DH* handle);


        template <bool _print>
        static void dev_conv2_fp16_NB_SK(decx::_GPU_MatrixArray* src, decx::_GPU_Matrix* kernel, decx::_GPU_MatrixArray* dst, 
            const int _accu_flag, de::DH* handle);


        template <bool _print>
        static void dev_conv2_fp16_NB_MK(decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
            const int _accu_flag, de::DH* handle);
    }
}



template <bool _print, bool _is_MK> static void 
decx::conv::dev_conv2_fp16_NB_SK_R8x8(decx::conv::_cuda_conv2_fp16_preset *_conv2_preset, const int _accu_flag, de::DH* handle)
{
    // generate every parameters convolution needs
    decx::conv::_conv2_fp16_NB_pre_process<_print, bounded_kernel_R8, bounded_kernel_R8, true>(_conv2_preset, handle);

    if (!_is_MK) {
        _conv2_preset->_cuda_conv2_MC_kernel_memcpy_from_host<_is_MK, de::Half>(0);
    }

    const uint _loop = _conv2_preset->_Kparams._src_confs._matrix_num;
    
    for (int i = 0; i < _loop; ++i) 
    {
        _conv2_preset->_cuda_conv2_MC_src_memcpy_from_host(i);
        if (_is_MK) {
            _conv2_preset->_cuda_conv2_MC_kernel_memcpy_from_host<_is_MK, de::Half>(i);
        }

        decx::conv::conv2_fp16_kernel_r8x8((float4*)_conv2_preset->src_buf.ptr,
                                            (de::Half*)_conv2_preset->ker_buf.ptr,
                                            (float4*)_conv2_preset->_Kparams._dst_confs._ptr_array[i],
                                            _conv2_preset->_Kparams.kernel_shift,
                                            make_uint2(_conv2_preset->_Kparams.src_buf_dims.x / 8,           _conv2_preset->_Kparams.src_buf_dims.y),
                                            make_uint2(_conv2_preset->_Kparams.dst_dims.x / 8,               _conv2_preset->_Kparams.dst_dims.y),
                                            make_uint2(_conv2_preset->_Kparams.ker_dims.x, _conv2_preset->_Kparams.ker_dims.y),              
                                            _conv2_preset->S, _accu_flag);

    }

    _conv2_preset->E->event_record(_conv2_preset->S);
    _conv2_preset->E->synchronize();
}



template <bool _print, bool _is_MK> static void 
decx::conv::dev_conv2_fp16_NB_SK_R8x16(decx::conv::_cuda_conv2_fp16_preset *_conv2_preset, const int _accu_flag, de::DH* handle)
{
    // generate every parameters convolution needs
    decx::conv::_conv2_fp16_NB_pre_process<_print, bounded_kernel_R8, bounded_kernel_R16, true>(_conv2_preset, handle);

    if (!_is_MK) {
        _conv2_preset->_cuda_conv2_MC_kernel_memcpy_from_host<_is_MK, de::Half>(0);
    }

    const uint _loop = _conv2_preset->_Kparams._src_confs._matrix_num;
    
    for (int i = 0; i < _loop; ++i) 
    {
        _conv2_preset->_cuda_conv2_MC_src_memcpy_from_host(i);
        if (_is_MK) {
            _conv2_preset->_cuda_conv2_MC_kernel_memcpy_from_host<_is_MK, de::Half>(i);
        }

        decx::conv::conv2_fp16_kernel_r8x16((float4*)_conv2_preset->src_buf.ptr,
                                            (de::Half*)_conv2_preset->ker_buf.ptr,
                                            (float4*)_conv2_preset->_Kparams._dst_confs._ptr_array[i],
                                            _conv2_preset->_Kparams.kernel_shift,
                                            make_uint2(_conv2_preset->_Kparams.src_buf_dims.x / 8,           _conv2_preset->_Kparams.src_buf_dims.y),
                                            make_uint2(_conv2_preset->_Kparams.dst_dims.x / 8,               _conv2_preset->_Kparams.dst_dims.y),
                                            make_uint2(_conv2_preset->_Kparams.ker_dims.x, _conv2_preset->_Kparams.ker_dims.y),              
                                            _conv2_preset->S, _accu_flag);

    }

    _conv2_preset->E->event_record(_conv2_preset->S);
    _conv2_preset->E->synchronize();
}




template <bool _print, bool _is_MK> static void 
decx::conv::dev_conv2_fp16_NB_SK_R16x8(decx::conv::_cuda_conv2_fp16_preset *_conv2_preset, const int _accu_flag, de::DH* handle)
{
    // generate every parameters convolution needs
    decx::conv::_conv2_fp16_NB_pre_process<_print, bounded_kernel_R16, bounded_kernel_R8, true>(_conv2_preset, handle);

    if (!_is_MK) {
        _conv2_preset->_cuda_conv2_MC_kernel_memcpy_from_host<_is_MK, de::Half>(0);
    }

    const uint _loop = _conv2_preset->_Kparams._src_confs._matrix_num;
    
    for (int i = 0; i < _loop; ++i) 
    {
        _conv2_preset->_cuda_conv2_MC_src_memcpy_from_host(i);
        if (_is_MK) {
            _conv2_preset->_cuda_conv2_MC_kernel_memcpy_from_host<_is_MK, de::Half>(i);
        }

        decx::conv::conv2_fp16_kernel_r16x8((float4*)_conv2_preset->src_buf.ptr,
                                            (de::Half*)_conv2_preset->ker_buf.ptr,
                                            (float4*)_conv2_preset->_Kparams._dst_confs._ptr_array[i],
                                            _conv2_preset->_Kparams.kernel_shift,
                                            make_uint2(_conv2_preset->_Kparams.src_buf_dims.x / 8,           _conv2_preset->_Kparams.src_buf_dims.y),
                                            make_uint2(_conv2_preset->_Kparams.dst_dims.x / 8,               _conv2_preset->_Kparams.dst_dims.y),
                                            make_uint2(_conv2_preset->_Kparams.ker_dims.x, _conv2_preset->_Kparams.ker_dims.y),              
                                            _conv2_preset->S, _accu_flag);

    }

    _conv2_preset->E->event_record(_conv2_preset->S);
    _conv2_preset->E->synchronize();
}




template <bool _print, bool _is_MK> static void 
decx::conv::dev_conv2_fp16_NB_SK_R16x16(decx::conv::_cuda_conv2_fp16_preset *_conv2_preset, const int _accu_flag, de::DH* handle)
{
    // generate every parameters convolution needs
    decx::conv::_conv2_fp16_NB_pre_process<_print, bounded_kernel_R16, bounded_kernel_R16, true>(_conv2_preset, handle);

    if (!_is_MK) {
        _conv2_preset->_cuda_conv2_MC_kernel_memcpy_from_host<_is_MK, de::Half>(0);
    }

    const uint _loop = _conv2_preset->_Kparams._src_confs._matrix_num;
    
    for (int i = 0; i < _loop; ++i) 
    {
        _conv2_preset->_cuda_conv2_MC_src_memcpy_from_host(i);
        if (_is_MK) {
            _conv2_preset->_cuda_conv2_MC_kernel_memcpy_from_host<_is_MK, de::Half>(i);
        }

        decx::conv::conv2_fp16_kernel_r16x16((float4*)_conv2_preset->src_buf.ptr,
                                            (de::Half*)_conv2_preset->ker_buf.ptr,
                                            (float4*)_conv2_preset->_Kparams._dst_confs._ptr_array[i],
                                            _conv2_preset->_Kparams.kernel_shift,
                                            make_uint2(_conv2_preset->_Kparams.src_buf_dims.x / 8,           _conv2_preset->_Kparams.src_buf_dims.y),
                                            make_uint2(_conv2_preset->_Kparams.dst_dims.x / 8,               _conv2_preset->_Kparams.dst_dims.y),
                                            make_uint2(_conv2_preset->_Kparams.ker_dims.x, _conv2_preset->_Kparams.ker_dims.y),              
                                            _conv2_preset->S, _accu_flag);

    }

    _conv2_preset->E->event_record(_conv2_preset->S);
    _conv2_preset->E->synchronize();
}




template <bool _print>
static void decx::conv::dev_conv2_fp16_NB_SK(decx::_GPU_MatrixArray* src, decx::_GPU_Matrix* kernel, decx::_GPU_MatrixArray* dst, const int _accu_flag, de::DH* handle)
{
    int2 half_ker_dim;
    half_ker_dim.x = kernel->Width() / 2;                half_ker_dim.y = kernel->Height() / 2;
    
    decx::conv::_cuda_conv2_fp16_preset _conv2_preset;
    _conv2_preset.memcpy_flag = cudaMemcpyDeviceToDevice;
    _conv2_preset._Kparams._src_confs.gen_matrix_configs(src);
    _conv2_preset._Kparams._kernel_confs.gen_matrix_configs(kernel);
    _conv2_preset._Kparams._dst_confs.gen_matrix_configs(dst);

    if (half_ker_dim.x < bounded_kernel_R8 + 1) {
        if (half_ker_dim.y < bounded_kernel_R8 + 1) {
            decx::conv::dev_conv2_fp16_NB_SK_R8x8<_print, false>(&_conv2_preset, _accu_flag, handle);
        }
        else {
            decx::conv::dev_conv2_fp16_NB_SK_R16x8<_print, false>(&_conv2_preset, _accu_flag, handle);
        }
    }
    else {
        if (half_ker_dim.y < bounded_kernel_R8 + 1) {
            decx::conv::dev_conv2_fp16_NB_SK_R8x16<_print, false>(&_conv2_preset, _accu_flag, handle);
        }
        else {
            decx::conv::dev_conv2_fp16_NB_SK_R16x16<_print, false>(&_conv2_preset, _accu_flag, handle);
        }
    }

    _conv2_preset.release();
}





template <bool _print>
static void decx::conv::dev_conv2_fp16_NB_MK(decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst, const int _accu_flag, de::DH* handle)
{
    int2 half_ker_dim;
    half_ker_dim.x = kernel->Width() / 2;                half_ker_dim.y = kernel->Height() / 2;

    decx::conv::_cuda_conv2_fp16_preset _conv2_preset;
    _conv2_preset.memcpy_flag = cudaMemcpyDeviceToDevice;
    _conv2_preset._Kparams._src_confs.gen_matrix_configs(src);
    _conv2_preset._Kparams._kernel_confs.gen_matrix_configs(kernel);
    _conv2_preset._Kparams._dst_confs.gen_matrix_configs(dst);

    if (half_ker_dim.x < bounded_kernel_R8 + 1) {
        if (half_ker_dim.y < bounded_kernel_R8 + 1) {
            decx::conv::dev_conv2_fp16_NB_SK_R8x8<_print, true>(&_conv2_preset, _accu_flag, handle);
        }
        else {
            decx::conv::dev_conv2_fp16_NB_SK_R16x8<_print, true>(&_conv2_preset, _accu_flag, handle);
        }
    }
    else {
        if (half_ker_dim.y < bounded_kernel_R8 + 1) {
            decx::conv::dev_conv2_fp16_NB_SK_R8x16<_print, true>(&_conv2_preset, _accu_flag, handle);
        }
        else {
            decx::conv::dev_conv2_fp16_NB_SK_R16x16<_print, true>(&_conv2_preset, _accu_flag, handle);
        }
    }

    _conv2_preset.release();
}



#endif