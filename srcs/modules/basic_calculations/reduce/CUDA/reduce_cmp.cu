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


#include "reduce_callers.cuh"
#include "../../../core/allocators.h"


template <bool _is_max>
void decx::reduce::cuda_reduce1D_cmp_fp32_caller_Async(decx::reduce::cuda_reduce1D_configs<float>* _kp_configs, decx::cuda_stream* S)
{
    std::vector<decx::reduce::RWPK_1D<float>>& _rwpk_arr = _kp_configs->get_rwpk();
    decx::reduce::RWPK_1D<float> _rwpk;
    const float _fill_val = _kp_configs->get_fill_val();

    for (int i = 0; i < _rwpk_arr.size(); ++i) {
        _rwpk = _rwpk_arr[i];
        
        decx::reduce::GPUK::cu_block_reduce_cmp1D_fp32<_is_max> << <_rwpk._grid_len, _rwpk._block_len,
            0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src, (float*)_rwpk._dst, _rwpk._proc_len_v, 
                _rwpk._proc_len_v1, _fill_val);
    }
}

template void decx::reduce::cuda_reduce1D_cmp_fp32_caller_Async<true> (decx::reduce::cuda_reduce1D_configs<float>*, decx::cuda_stream*);
template void decx::reduce::cuda_reduce1D_cmp_fp32_caller_Async<false>(decx::reduce::cuda_reduce1D_configs<float>*, decx::cuda_stream*);



template <bool _is_max>
void decx::reduce::cuda_reduce1D_cmp_int32_caller_Async(decx::reduce::cuda_reduce1D_configs<int32_t>* _kp_configs, decx::cuda_stream* S)
{
    std::vector<decx::reduce::RWPK_1D<int32_t>>& _rwpk_arr = _kp_configs->get_rwpk();
    decx::reduce::RWPK_1D<int32_t> _rwpk;
    const int32_t _fill_val = _kp_configs->get_fill_val();

    for (int i = 0; i < _rwpk_arr.size(); ++i) {
        _rwpk = _rwpk_arr[i];

        decx::reduce::GPUK::cu_block_reduce_cmp1D_int32<_is_max> << <_rwpk._grid_len, _rwpk._block_len,
            0, S->get_raw_stream_ref() >> > ((int4*)_rwpk._src, (int32_t*)_rwpk._dst, _rwpk._proc_len_v,
                _rwpk._proc_len_v1, _fill_val);
    }
}

template void decx::reduce::cuda_reduce1D_cmp_int32_caller_Async<true>(decx::reduce::cuda_reduce1D_configs<int32_t>*, decx::cuda_stream*);
template void decx::reduce::cuda_reduce1D_cmp_int32_caller_Async<false>(decx::reduce::cuda_reduce1D_configs<int32_t>*, decx::cuda_stream*);



template <bool _is_max>
void decx::reduce::cuda_reduce1D_cmp_fp64_caller_Async(decx::reduce::cuda_reduce1D_configs<double>* _kp_configs, decx::cuda_stream* S)
{
    std::vector<decx::reduce::RWPK_1D<double>>& _rwpk_arr = _kp_configs->get_rwpk();
    decx::reduce::RWPK_1D<double> _rwpk;
    const double _fill_val = _kp_configs->get_fill_val();

    for (int i = 0; i < _rwpk_arr.size(); ++i) {
        _rwpk = _rwpk_arr[i];

        decx::reduce::GPUK::cu_block_reduce_cmp1D_fp64<_is_max> << <_rwpk._grid_len, _rwpk._block_len,
            0, S->get_raw_stream_ref() >> > ((double2*)_rwpk._src, (double*)_rwpk._dst, _rwpk._proc_len_v, 
                _rwpk._proc_len_v1, _fill_val);
    }
}

template void decx::reduce::cuda_reduce1D_cmp_fp64_caller_Async<true>(decx::reduce::cuda_reduce1D_configs<double>*, decx::cuda_stream*);
template void decx::reduce::cuda_reduce1D_cmp_fp64_caller_Async<false>(decx::reduce::cuda_reduce1D_configs<double>*, decx::cuda_stream*);




template <bool _is_max>
void decx::reduce::cuda_reduce1D_cmp_fp16_caller_Async(decx::reduce::cuda_reduce1D_configs<de::Half>* _kp_configs, decx::cuda_stream* S)
{
    std::vector<decx::reduce::RWPK_1D<de::Half>>& _rwpk_arr = _kp_configs->get_rwpk();
    decx::reduce::RWPK_1D<de::Half> _rwpk;
    const de::Half _fill_val = _kp_configs->get_fill_val();

    for (int i = 0; i < _rwpk_arr.size(); ++i) {
        _rwpk = _rwpk_arr[i];

        decx::reduce::GPUK::cu_block_reduce_cmp1D_fp16<_is_max> << <_rwpk._grid_len, _rwpk._block_len,
            0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src, (half*)_rwpk._dst, _rwpk._proc_len_v, 
                _rwpk._proc_len_v1, *((half*)&_fill_val));
    }
}

template void decx::reduce::cuda_reduce1D_cmp_fp16_caller_Async<true> (decx::reduce::cuda_reduce1D_configs<de::Half>*, decx::cuda_stream*);
template void decx::reduce::cuda_reduce1D_cmp_fp16_caller_Async<false>(decx::reduce::cuda_reduce1D_configs<de::Half>*, decx::cuda_stream*);




template <bool _is_max>
void decx::reduce::cuda_reduce1D_cmp_u8_caller_Async(decx::reduce::cuda_reduce1D_configs<uint8_t>* _kp_configs, decx::cuda_stream* S)
{
    std::vector<decx::reduce::RWPK_1D<uint8_t>>& _rwpk_arr = _kp_configs->get_rwpk();
    decx::reduce::RWPK_1D<uint8_t> _rwpk;
    const uint8_t _fill_val = _kp_configs->get_fill_val();

    for (int i = 0; i < _rwpk_arr.size(); ++i) {
        _rwpk = _rwpk_arr[i];

        decx::reduce::GPUK::cu_block_reduce_cmp1D_u8<_is_max> << <_rwpk._grid_len, _rwpk._block_len,
            0, S->get_raw_stream_ref() >> > ((int4*)_rwpk._src, (uint8_t*)_rwpk._dst, _rwpk._proc_len_v, 
                _rwpk._proc_len_v1, _fill_val);
    }
}


template void decx::reduce::cuda_reduce1D_cmp_u8_caller_Async<true>(decx::reduce::cuda_reduce1D_configs<uint8_t>*, decx::cuda_stream*);
template void decx::reduce::cuda_reduce1D_cmp_u8_caller_Async<false>(decx::reduce::cuda_reduce1D_configs<uint8_t>*, decx::cuda_stream*);


// ----------------------------------------------------- 2D -----------------------------------------------------------

template <bool _is_max>
void decx::reduce::reduce_cmp2D_h_fp32_Async(decx::reduce::cuda_reduce2D_1way_configs<float>* _configs, decx::cuda_stream* S)
{
    const std::vector<decx::reduce::RWPK_2D>& _rwpks = _configs->get_rwpks();
    decx::reduce::RWPK_2D _rwpk;

    for (int i = 0; i < _rwpks.size()/* - 1*/; ++i) {
        _rwpk = _rwpks[i];
        
        decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp32<_is_max> << <_rwpk._grid_dims, _rwpk._block_dims,
            0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src,       (float*)_rwpk._dst, 
                                             _rwpk._calc_pitch_src,     _rwpk._calc_pitch_dst,      _rwpk._calc_proc_dims);
    }
}

template void decx::reduce::reduce_cmp2D_h_fp32_Async<true>(decx::reduce::cuda_reduce2D_1way_configs<float>* _configs, decx::cuda_stream* S);
template void decx::reduce::reduce_cmp2D_h_fp32_Async<false>(decx::reduce::cuda_reduce2D_1way_configs<float>* _configs, decx::cuda_stream* S);




template <bool _is_max>
void decx::reduce::reduce_cmp2D_v_fp32_Async(decx::reduce::cuda_reduce2D_1way_configs<float>* _configs, decx::cuda_stream* S)
{
    const std::vector<decx::reduce::RWPK_2D>& _rwpks = _configs->get_rwpks();
    decx::reduce::RWPK_2D _rwpk;

    for (int i = 0; i < _rwpks.size(); ++i) 
    {
        _rwpk = _rwpks[i];
        decx::reduce::GPUK::cu_warp_reduce_cmp2D_v_fp32<_is_max> << <_rwpk._grid_dims, _rwpk._block_dims,
            0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src,       (float4*)_rwpk._dst, 
                                             _rwpk._calc_pitch_src,     _rwpk._calc_pitch_dst,      _rwpk._calc_proc_dims);
    }
}

template void decx::reduce::reduce_cmp2D_v_fp32_Async<true>(decx::reduce::cuda_reduce2D_1way_configs<float>*, decx::cuda_stream*); 
template void decx::reduce::reduce_cmp2D_v_fp32_Async<false>(decx::reduce::cuda_reduce2D_1way_configs<float>*, decx::cuda_stream*);



template <bool _is_max>
void decx::reduce::reduce_cmp2D_v_fp16_Async(decx::reduce::cuda_reduce2D_1way_configs<de::Half>* _configs, decx::cuda_stream* S)
{
    const std::vector<decx::reduce::RWPK_2D>& _rwpks = _configs->get_rwpks();
    decx::reduce::RWPK_2D _rwpk;

    for (int i = 0; i < _rwpks.size(); ++i) {
        _rwpk = _rwpks[i];

        decx::reduce::GPUK::cu_warp_reduce_cmp2D_v_fp16<_is_max> << <_rwpk._grid_dims, _rwpk._block_dims,
            0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src,       (float4*)_rwpk._dst,
                                             _rwpk._calc_pitch_src,     _rwpk._calc_pitch_dst,      _rwpk._calc_proc_dims);
    }
}

template void decx::reduce::reduce_cmp2D_v_fp16_Async<true>(decx::reduce::cuda_reduce2D_1way_configs<de::Half>*, decx::cuda_stream*);
template void decx::reduce::reduce_cmp2D_v_fp16_Async<false>(decx::reduce::cuda_reduce2D_1way_configs<de::Half>*, decx::cuda_stream*);



template <bool _is_max>
void decx::reduce::reduce_cmp2D_v_u8_Async(decx::reduce::cuda_reduce2D_1way_configs<uint8_t>* _configs, decx::cuda_stream* S)
{
    const std::vector<decx::reduce::RWPK_2D>& _rwpks = _configs->get_rwpks();
    decx::reduce::RWPK_2D _rwpk;

    for (int i = 0; i < _rwpks.size(); ++i) {
        _rwpk = _rwpks[i];

        decx::reduce::GPUK::cu_warp_reduce_cmp2D_v_u8<_is_max> << <_rwpk._grid_dims, _rwpk._block_dims,
            0, S->get_raw_stream_ref() >> > ((int4*)_rwpk._src, (int4*)_rwpk._dst,
                _rwpk._calc_pitch_src, _rwpk._calc_pitch_dst, _rwpk._calc_proc_dims);
    }
}

template void decx::reduce::reduce_cmp2D_v_u8_Async<true>(decx::reduce::cuda_reduce2D_1way_configs<uint8_t>*, decx::cuda_stream*);
template void decx::reduce::reduce_cmp2D_v_u8_Async<false>(decx::reduce::cuda_reduce2D_1way_configs<uint8_t>*, decx::cuda_stream*);




template <bool _is_max>
void decx::reduce::reduce_cmp2D_h_fp16_Async(decx::reduce::cuda_reduce2D_1way_configs<de::Half>* _configs, decx::cuda_stream* S)
{
    const std::vector<decx::reduce::RWPK_2D>& _rwpks = _configs->get_rwpks();
    decx::reduce::RWPK_2D _rwpk;

    for (int i = 0; i < _rwpks.size()/* - 1*/; ++i) {
        _rwpk = _rwpks[i];
        decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp16<_is_max> << <_rwpk._grid_dims, _rwpk._block_dims,
            0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src,       (__half*)_rwpk._dst, 
                                                _rwpk._calc_pitch_src,     _rwpk._calc_pitch_dst,      _rwpk._calc_proc_dims);
    }

    /*_rwpk = _rwpks[_rwpks.size() - 1];
    decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp16_transp<_is_max> << <_rwpk._grid_dims, _rwpk._block_dims,
        0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src,       (__half*)_rwpk._dst, 
                                            _rwpk._calc_pitch_src,     _rwpk._calc_pitch_dst,      _rwpk._calc_proc_dims);*/
}

template void decx::reduce::reduce_cmp2D_h_fp16_Async<true>(decx::reduce::cuda_reduce2D_1way_configs<de::Half>*, decx::cuda_stream*);
template void decx::reduce::reduce_cmp2D_h_fp16_Async<false>(decx::reduce::cuda_reduce2D_1way_configs<de::Half>*, decx::cuda_stream*);




template <bool _is_max>
void decx::reduce::reduce_cmp2D_h_u8_Async(decx::reduce::cuda_reduce2D_1way_configs<uint8_t>* _configs, decx::cuda_stream* S)
{
    const std::vector<decx::reduce::RWPK_2D>& _rwpks = _configs->get_rwpks();
    decx::reduce::RWPK_2D _rwpk;

    for (int i = 0; i < _rwpks.size()/* - 1*/; ++i) {
        _rwpk = _rwpks[i];

        decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_u8<_is_max> << <_rwpk._grid_dims, _rwpk._block_dims,
            0, S->get_raw_stream_ref() >> > ((int4*)_rwpk._src, (uint8_t*)_rwpk._dst,
                _rwpk._calc_pitch_src, _rwpk._calc_pitch_dst, _rwpk._calc_proc_dims);
    }

    /*_rwpk = _rwpks[_rwpks.size() - 1];

    decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_u8_transp<_is_max> << <_rwpk._grid_dims, _rwpk._block_dims,
        0, S->get_raw_stream_ref() >> > ((int4*)_rwpk._src, (uint8_t*)_rwpk._dst,
            _rwpk._calc_pitch_src, _rwpk._calc_pitch_dst, _rwpk._calc_proc_dims);*/
}

template void decx::reduce::reduce_cmp2D_h_u8_Async<true>(decx::reduce::cuda_reduce2D_1way_configs<uint8_t>*, decx::cuda_stream*);
template void decx::reduce::reduce_cmp2D_h_u8_Async<false>(decx::reduce::cuda_reduce2D_1way_configs<uint8_t>*, decx::cuda_stream*);



template <bool _is_max>
const void* decx::reduce::reduce_cmp2D_full_fp32_Async(decx::reduce::cuda_reduce1D_configs<float>* _kp_configs, const void* src_ptr, const uint2 proc_dims,
    decx::cuda_stream* S, const bool _more_than_flatten)
{
    decx::reduce::RWPK_2D rwpk_flatten = _kp_configs->get_rwpk_flatten();

    decx::reduce::GPUK::cu_warp_reduce_cmp2D_flatten_fp32<_is_max> << <rwpk_flatten._grid_dims, rwpk_flatten._block_dims,
        0, S->get_raw_stream_ref() >> > ((float4*)src_ptr, (float*)_kp_configs->get_src(), rwpk_flatten._calc_pitch_src, 
            rwpk_flatten._calc_proc_dims, _kp_configs->get_fill_val());
    
    if (_more_than_flatten) {
        decx::reduce::cuda_reduce1D_cmp_fp32_caller_Async<_is_max>(_kp_configs, S);

        return _kp_configs->get_dst();
    }
    else {
        return _kp_configs->get_src();
    }
}


template const void* decx::reduce::reduce_cmp2D_full_fp32_Async<true>(decx::reduce::cuda_reduce1D_configs<float>*, const void*, const uint2,
    decx::cuda_stream*, const bool);
template const void* decx::reduce::reduce_cmp2D_full_fp32_Async<false>(decx::reduce::cuda_reduce1D_configs<float>*, const void*, const uint2,
    decx::cuda_stream*, const bool);




template <bool _is_max>
const void* decx::reduce::reduce_cmp2D_full_int32_Async(decx::reduce::cuda_reduce1D_configs<int32_t>* _kp_configs, const void* src_ptr, const uint2 proc_dims,
    decx::cuda_stream* S, const bool _more_than_flatten)
{
    decx::reduce::RWPK_2D rwpk_flatten = _kp_configs->get_rwpk_flatten();

    decx::reduce::GPUK::cu_warp_reduce_cmp2D_flatten_int32<_is_max> << <rwpk_flatten._grid_dims, rwpk_flatten._block_dims,
        0, S->get_raw_stream_ref() >> > ((int4*)src_ptr, (int32_t*)_kp_configs->get_src(), rwpk_flatten._calc_pitch_src, 
            rwpk_flatten._calc_proc_dims, _kp_configs->get_fill_val());
    
    if (_more_than_flatten) {
        decx::reduce::cuda_reduce1D_cmp_int32_caller_Async<_is_max>(_kp_configs, S);

        return _kp_configs->get_dst();
    }
    else {
        return _kp_configs->get_src();
    }
}


template const void* decx::reduce::reduce_cmp2D_full_int32_Async<true>(decx::reduce::cuda_reduce1D_configs<int32_t>*, const void*, const uint2,
    decx::cuda_stream*, const bool);
template const void* decx::reduce::reduce_cmp2D_full_int32_Async<false>(decx::reduce::cuda_reduce1D_configs<int32_t>*, const void*, const uint2,
    decx::cuda_stream*, const bool);




template <bool _is_max>
const void* decx::reduce::reduce_cmp2D_full_fp64_Async(decx::reduce::cuda_reduce1D_configs<double>* _kp_configs, const void* src_ptr, const uint2 proc_dims,
    decx::cuda_stream* S, const bool _more_than_flatten)
{
    decx::reduce::RWPK_2D rwpk_flatten = _kp_configs->get_rwpk_flatten();

    decx::reduce::GPUK::cu_warp_reduce_cmp2D_flatten_fp64<_is_max> << <rwpk_flatten._grid_dims, rwpk_flatten._block_dims,
        0, S->get_raw_stream_ref() >> > ((double2*)src_ptr, (double*)_kp_configs->get_src(), rwpk_flatten._calc_pitch_src,
            rwpk_flatten._calc_proc_dims, _kp_configs->get_fill_val());

    if (_more_than_flatten) {
        decx::reduce::cuda_reduce1D_cmp_fp64_caller_Async<_is_max>(_kp_configs, S);

        return _kp_configs->get_dst();
    }
    else {
        return _kp_configs->get_src();
    }
}


template const void* decx::reduce::reduce_cmp2D_full_fp64_Async<true>(decx::reduce::cuda_reduce1D_configs<double>*, const void*, const uint2,
    decx::cuda_stream*, const bool);
template const void* decx::reduce::reduce_cmp2D_full_fp64_Async<false>(decx::reduce::cuda_reduce1D_configs<double>*, const void*, const uint2,
    decx::cuda_stream*, const bool);



template <bool _is_max>
const void* decx::reduce::reduce_cmp2D_full_fp16_Async(decx::reduce::cuda_reduce1D_configs<de::Half>* _kp_configs, const void* src_ptr, const uint2 proc_dims,
    decx::cuda_stream* S, const bool _more_than_flatten)
{
    decx::reduce::RWPK_2D rwpk_flatten = _kp_configs->get_rwpk_flatten();

    de::Half _fill_val = _kp_configs->get_fill_val();

    decx::reduce::GPUK::cu_warp_reduce_cmp2D_flatten_fp16<_is_max> << <rwpk_flatten._grid_dims, rwpk_flatten._block_dims,
        0, S->get_raw_stream_ref() >> > ((float4*)src_ptr, (half*)_kp_configs->get_src(), rwpk_flatten._calc_pitch_src, 
            rwpk_flatten._calc_proc_dims, *((__half*)&_fill_val));

    if (_more_than_flatten) {
        decx::reduce::cuda_reduce1D_cmp_fp16_caller_Async<_is_max>(_kp_configs, S);

        return _kp_configs->get_dst();
    }
    else {
        return _kp_configs->get_src();
    }
}


template const void* decx::reduce::reduce_cmp2D_full_fp16_Async<true>(decx::reduce::cuda_reduce1D_configs<de::Half>*, const void*, const uint2,
    decx::cuda_stream*, const bool);
template const void* decx::reduce::reduce_cmp2D_full_fp16_Async<false>(decx::reduce::cuda_reduce1D_configs<de::Half>*, const void*, const uint2,
    decx::cuda_stream*, const bool);




template <bool _is_max>
const void* decx::reduce::reduce_cmp2D_full_u8_Async(decx::reduce::cuda_reduce1D_configs<uint8_t>* _kp_configs, const void* src_ptr, const uint2 proc_dims,
    decx::cuda_stream* S, const bool _more_than_flatten)
{
    decx::reduce::RWPK_2D rwpk_flatten = _kp_configs->get_rwpk_flatten();

    decx::reduce::GPUK::cu_warp_reduce_cmp2D_flatten_u8<_is_max> << <rwpk_flatten._grid_dims, rwpk_flatten._block_dims,
        0, S->get_raw_stream_ref() >> > ((int4*)src_ptr, (uint8_t*)_kp_configs->get_src(), rwpk_flatten._calc_pitch_src,
            rwpk_flatten._calc_proc_dims, _kp_configs->get_fill_val());

    if (_more_than_flatten) {
        decx::reduce::cuda_reduce1D_cmp_u8_caller_Async<_is_max>(_kp_configs, S);

        return _kp_configs->get_dst();
    }
    else {
        return _kp_configs->get_src();
    }
}


template const void* decx::reduce::reduce_cmp2D_full_u8_Async<true>(decx::reduce::cuda_reduce1D_configs<uint8_t>*, const void*, const uint2,
    decx::cuda_stream*, const bool);
template const void* decx::reduce::reduce_cmp2D_full_u8_Async<false>(decx::reduce::cuda_reduce1D_configs<uint8_t>*, const void*, const uint2,
    decx::cuda_stream*, const bool);