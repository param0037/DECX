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



void decx::reduce::cuda_reduce1D_sum_fp32_caller_Async(decx::reduce::cuda_reduce1D_configs<float>* _kp_configs, decx::cuda_stream* S)
{
    std::vector<decx::reduce::RWPK_1D<float>>& _rwpk_arr = _kp_configs->get_rwpk();
    decx::reduce::RWPK_1D<float> _rwpk;

    for (int i = 0; i < _rwpk_arr.size(); ++i) {
        _rwpk = _rwpk_arr[i];

        decx::reduce::GPUK::cu_block_reduce_sum1D_fp32 << <_rwpk._grid_len, _rwpk._block_len,
            0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src, (float*)_rwpk._dst, _rwpk._proc_len_v, _rwpk._proc_len_v1);
    }
}




void decx::reduce::cuda_reduce1D_sum_fp64_caller_Async(decx::reduce::cuda_reduce1D_configs<double>* _kp_configs, decx::cuda_stream* S)
{
    std::vector<decx::reduce::RWPK_1D<double>>& _rwpk_arr = _kp_configs->get_rwpk();
    decx::reduce::RWPK_1D<double> _rwpk;

    for (int i = 0; i < _rwpk_arr.size(); ++i) {
        _rwpk = _rwpk_arr[i];

        decx::reduce::GPUK::cu_block_reduce_sum1D_fp64 << <_rwpk._grid_len, _rwpk._block_len,
            0, S->get_raw_stream_ref() >> > ((double2*)_rwpk._src, (double*)_rwpk._dst, _rwpk._proc_len_v, _rwpk._proc_len_v1);
    }
}




void decx::reduce::cuda_reduce1D_sum_i32_caller_Async(decx::reduce::cuda_reduce1D_configs<int32_t>* _kp_configs, decx::cuda_stream* S)
{
    std::vector<decx::reduce::RWPK_1D<int32_t>>& _rwpk_arr = _kp_configs->get_rwpk();
    decx::reduce::RWPK_1D<int32_t> _rwpk;

    for (int i = 0; i < _rwpk_arr.size(); ++i) {
        _rwpk = _rwpk_arr[i];

        decx::reduce::GPUK::cu_block_reduce_sum1D_int32 << <_rwpk._grid_len, _rwpk._block_len,
            0, S->get_raw_stream_ref() >> > ((int4*)_rwpk._src, (int32_t*)_rwpk._dst, _rwpk._proc_len_v, _rwpk._proc_len_v1);
    }
}





void decx::reduce::cuda_reduce1D_sum_u8_i32_caller_Async(decx::reduce::cuda_reduce1D_configs<uint8_t>* _kp_configs, decx::cuda_stream* S)
{
    std::vector<decx::reduce::RWPK_1D<uint8_t>>& _rwpk_arr = _kp_configs->get_rwpk();
    decx::reduce::RWPK_1D<uint8_t> _rwpk;

    _rwpk = _rwpk_arr[0];
    decx::reduce::GPUK::cu_block_reduce_sum1D_u8_i32 << <_rwpk._grid_len, _rwpk._block_len,
        0, S->get_raw_stream_ref() >> > ((int4*)_rwpk._src, (int32_t*)_rwpk._dst, _rwpk._proc_len_v, _rwpk._proc_len_v1);

    for (int i = 1; i < _rwpk_arr.size(); ++i) 
    {
        _rwpk = _rwpk_arr[i];
        decx::reduce::GPUK::cu_block_reduce_sum1D_int32 << <_rwpk._grid_len, _rwpk._block_len,
            0, S->get_raw_stream_ref() >> > ((int4*)_rwpk._src, (int32_t*)_rwpk._dst, _rwpk._proc_len_v, _rwpk._proc_len_v1);
    }
}



void decx::reduce::cuda_reduce1D_sum_fp16_caller_Async(decx::reduce::cuda_reduce1D_configs<de::Half>* _kp_configs, decx::cuda_stream* S,
    const uint32_t _fp16_accu)
{
    std::vector<decx::reduce::RWPK_1D<de::Half>>& _rwpk_arr = _kp_configs->get_rwpk();
    decx::reduce::RWPK_1D<de::Half> _rwpk;

    if (_fp16_accu == decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1) {
        _rwpk = _rwpk_arr[0];

        decx::reduce::GPUK::cu_block_reduce_sum1D_fp16_L1 << <_rwpk._grid_len, _rwpk._block_len,
            0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src, (float*)_rwpk._dst, _rwpk._proc_len_v, _rwpk._proc_len_v1);
    }

    for (int i = (_fp16_accu == decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1 ? 1 : 0); i < _rwpk_arr.size(); ++i) {
        _rwpk = _rwpk_arr[i];
        switch (_fp16_accu)
        {
        case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1:
            decx::reduce::GPUK::cu_block_reduce_sum1D_fp32 << <_rwpk._grid_len, _rwpk._block_len,
                0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src, (float*)_rwpk._dst, _rwpk._proc_len_v, _rwpk._proc_len_v1);
            break;

        case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L2:
            decx::reduce::GPUK::cu_block_reduce_sum1D_fp16_L2 << <_rwpk._grid_len, _rwpk._block_len,
                0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src, (__half*)_rwpk._dst, _rwpk._proc_len_v, _rwpk._proc_len_v1);
            break;

        case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L3:
            decx::reduce::GPUK::cu_block_reduce_sum1D_fp16_L3 << <_rwpk._grid_len, _rwpk._block_len,
                0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src, (__half*)_rwpk._dst, _rwpk._proc_len_v, _rwpk._proc_len_v1);
            break;
        default:
            break;
        }
    }
}

// ---------------------------------------------------------- 2D ------------------------------------------------------------


void decx::reduce::reduce_sum2D_h_fp32_Async(decx::reduce::cuda_reduce2D_1way_configs<float>* _configs, decx::cuda_stream* S)
{
    const std::vector<decx::reduce::RWPK_2D>& _rwpks = _configs->get_rwpks();
    decx::reduce::RWPK_2D _rwpk;

    for (int i = 0; i < _rwpks.size(); ++i) {
        _rwpk = _rwpks[i];
        
        decx::reduce::GPUK::cu_warp_reduce_sum2D_h_fp32 << <_rwpk._grid_dims, _rwpk._block_dims,
            0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src,       
                                             (float*)_rwpk._dst, 
                                             _rwpk._calc_pitch_src,     
                                             _rwpk._calc_pitch_dst,      
                                             _rwpk._calc_proc_dims);
    }
}



void decx::reduce::reduce_sum2D_h_fp64_Async(decx::reduce::cuda_reduce2D_1way_configs<double>* _configs, decx::cuda_stream* S)
{
    const std::vector<decx::reduce::RWPK_2D>& _rwpks = _configs->get_rwpks();
    decx::reduce::RWPK_2D _rwpk;

    for (int i = 0; i < _rwpks.size(); ++i) {
        _rwpk = _rwpks[i];
        
        decx::reduce::GPUK::cu_warp_reduce_sum2D_h_fp64 << <_rwpk._grid_dims, _rwpk._block_dims,
            0, S->get_raw_stream_ref() >> > ((double2*)_rwpk._src,       (double*)_rwpk._dst, 
                                             _rwpk._calc_pitch_src,     _rwpk._calc_pitch_dst,      _rwpk._calc_proc_dims);
    }
}




void decx::reduce::reduce_sum2D_h_fp16_Async(decx::reduce::cuda_reduce2D_1way_configs<de::Half>* _configs, decx::cuda_stream* S,
    const uint32_t _fp16_accu)
{
    const std::vector<decx::reduce::RWPK_2D>& _rwpks = _configs->get_rwpks();
    decx::reduce::RWPK_2D _rwpk;

    const bool _changed_lane = (_fp16_accu == decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1);
    
    if (_changed_lane) {
        _rwpk = _rwpks[0];

        decx::reduce::GPUK::cu_warp_reduce_sum2D_h_fp16_L1 << <_rwpk._grid_dims, _rwpk._block_dims,
            0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src, (float*)_rwpk._dst,
                _rwpk._calc_pitch_src, _rwpk._calc_pitch_dst, _rwpk._calc_proc_dims);
    }

    for (int i = (_changed_lane ? 1 : 0); i < _rwpks.size(); ++i) {
        _rwpk = _rwpks[i];
        switch (_fp16_accu)
        {
        case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1:
            decx::reduce::GPUK::cu_warp_reduce_sum2D_h_fp32 << <_rwpk._grid_dims, _rwpk._block_dims,
                0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src, (float*)_rwpk._dst,
                    _rwpk._calc_pitch_src, _rwpk._calc_pitch_dst, _rwpk._calc_proc_dims);
            break;

        case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L2:
            decx::reduce::GPUK::cu_warp_reduce_sum2D_h_fp16_L2 << <_rwpk._grid_dims, _rwpk._block_dims,
                0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src, (__half*)_rwpk._dst,
                    _rwpk._calc_pitch_src, _rwpk._calc_pitch_dst, _rwpk._calc_proc_dims);
            break;

        case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L3:
            decx::reduce::GPUK::cu_warp_reduce_sum2D_h_fp16_L3 << <_rwpk._grid_dims, _rwpk._block_dims,
                0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src, (__half*)_rwpk._dst,
                    _rwpk._calc_pitch_src, _rwpk._calc_pitch_dst, _rwpk._calc_proc_dims);
            break;
        default:
            break;
        }
    }
}




void decx::reduce::reduce_sum2D_h_u8_i32_Async(decx::reduce::cuda_reduce2D_1way_configs<uint8_t>* _configs, decx::cuda_stream* S)
{
    using namespace std;
    const std::vector<decx::reduce::RWPK_2D>& _rwpks = _configs->get_rwpks();
    decx::reduce::RWPK_2D _rwpk;

    _rwpk = _rwpks[0];
    decx::reduce::GPUK::cu_warp_reduce_sum2D_h_u8_i32 << <_rwpk._grid_dims, _rwpk._block_dims,
        0, S->get_raw_stream_ref() >> > ((int4*)_rwpk._src,       (int32_t*)_rwpk._dst, 
                                            _rwpk._calc_pitch_src,     _rwpk._calc_pitch_dst,      _rwpk._calc_proc_dims);
        
    for (int i = 1; i < _rwpks.size(); ++i) {
        _rwpk = _rwpks[i];
        decx::reduce::GPUK::cu_warp_reduce_sum2D_h_int32 << <_rwpk._grid_dims, _rwpk._block_dims,
            0, S->get_raw_stream_ref() >> > ((int4*)_rwpk._src,       (int32_t*)_rwpk._dst, 
                                                _rwpk._calc_pitch_src,     _rwpk._calc_pitch_dst,      _rwpk._calc_proc_dims);
    }
}




void decx::reduce::reduce_sum2D_v_fp32_Async(decx::reduce::cuda_reduce2D_1way_configs<float>* _configs, decx::cuda_stream* S)
{
    const std::vector<decx::reduce::RWPK_2D>& _rwpks = _configs->get_rwpks();
    decx::reduce::RWPK_2D _rwpk;

    for (int i = 0; i < _rwpks.size(); ++i) 
    {
        _rwpk = _rwpks[i];
        
        decx::reduce::GPUK::cu_warp_reduce_sum2D_v_fp32 << <_rwpk._grid_dims, _rwpk._block_dims,
            0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src,       (float4*)_rwpk._dst, 
                                             _rwpk._calc_pitch_src,     _rwpk._calc_pitch_dst,      _rwpk._calc_proc_dims);
    }
}



void decx::reduce::reduce_sum2D_v_fp64_Async(decx::reduce::cuda_reduce2D_1way_configs<double>* _configs, decx::cuda_stream* S)
{
    const std::vector<decx::reduce::RWPK_2D>& _rwpks = _configs->get_rwpks();
    decx::reduce::RWPK_2D _rwpk;

    for (int i = 0; i < _rwpks.size(); ++i)
    {
        _rwpk = _rwpks[i];

        decx::reduce::GPUK::cu_warp_reduce_sum2D_v_fp64 << <_rwpk._grid_dims, _rwpk._block_dims,
            0, S->get_raw_stream_ref() >> > ((double2*)_rwpk._src, (double2*)_rwpk._dst,
                _rwpk._calc_pitch_src, _rwpk._calc_pitch_dst, _rwpk._calc_proc_dims);
    }
}



void decx::reduce::reduce_sum2D_v_fp16_Async(decx::reduce::cuda_reduce2D_1way_configs<de::Half>* _configs, decx::cuda_stream* S,
    const uint32_t _fp16_accu)
{
    const std::vector<decx::reduce::RWPK_2D>& _rwpks = _configs->get_rwpks();
    decx::reduce::RWPK_2D _rwpk;

    const bool _load_lane_changed = _fp16_accu == decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1;
    if (_load_lane_changed) {
        _rwpk = _rwpks[0];
        decx::reduce::GPUK::cu_warp_reduce_sum2D_v_fp16_L1 << <_rwpk._grid_dims, _rwpk._block_dims,
            0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src, (float4*)_rwpk._dst,
                _rwpk._calc_pitch_src, _rwpk._calc_pitch_dst, _rwpk._calc_proc_dims);
    }

    for (int i = _load_lane_changed ? 1 : 0; i < _rwpks.size(); ++i) {
        _rwpk = _rwpks[i];

        switch (_fp16_accu)
        {
        case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1:
            decx::reduce::GPUK::cu_warp_reduce_sum2D_v_fp32 << <_rwpk._grid_dims, _rwpk._block_dims,
            0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src,       (float4*)_rwpk._dst, 
                                             _rwpk._calc_pitch_src,     _rwpk._calc_pitch_dst,      _rwpk._calc_proc_dims);
            break;

        case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L2:
            decx::reduce::GPUK::cu_warp_reduce_sum2D_v_fp16_L2 << <_rwpk._grid_dims, _rwpk._block_dims,
            0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src,       (float4*)_rwpk._dst, 
                                             _rwpk._calc_pitch_src,     _rwpk._calc_pitch_dst,      _rwpk._calc_proc_dims);
            break;

        case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L3:
            decx::reduce::GPUK::cu_warp_reduce_sum2D_v_fp16_L3 << <_rwpk._grid_dims, _rwpk._block_dims,
            0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src,       (float4*)_rwpk._dst, 
                                             _rwpk._calc_pitch_src,     _rwpk._calc_pitch_dst,      _rwpk._calc_proc_dims);
            break;

        default:
            break;
        }
        
    }
}



void decx::reduce::reduce_sum2D_v_u8_i32_Async(decx::reduce::cuda_reduce2D_1way_configs<uint8_t>* _configs, decx::cuda_stream* S)
{
    const std::vector<decx::reduce::RWPK_2D>& _rwpks = _configs->get_rwpks();
    decx::reduce::RWPK_2D _rwpk;

    _rwpk = _rwpks[0];
    decx::reduce::GPUK::cu_warp_reduce_sum2D_v_u8_i32 << <_rwpk._grid_dims, _rwpk._block_dims,
        0, S->get_raw_stream_ref() >> > ((int4*)_rwpk._src, (int4*)_rwpk._dst,
            _rwpk._calc_pitch_src, _rwpk._calc_pitch_dst, _rwpk._calc_proc_dims);


    for (int i = 1; i < _rwpks.size(); ++i) {
        _rwpk = _rwpks[i];
        decx::reduce::GPUK::cu_warp_reduce_sum2D_v_fp32 << <_rwpk._grid_dims, _rwpk._block_dims,
            0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src, (float4*)_rwpk._dst,
                _rwpk._calc_pitch_src, _rwpk._calc_pitch_dst, _rwpk._calc_proc_dims);
    }
}

// ------------------------------------------------------- FULL -------------------------------------------------------

const void* decx::reduce::reduce_sum2D_full_fp32_Async(decx::reduce::cuda_reduce1D_configs<float>*   _kp_configs, 
                                                const void*                                          src_ptr, 
                                                const uint2                                          proc_dims,
                                                decx::cuda_stream*                                   S,
                                                const bool                                           _more_than_flatten)
{
    decx::reduce::RWPK_2D rwpk_flatten = _kp_configs->get_rwpk_flatten();
    
    decx::reduce::GPUK::cu_warp_reduce_sum2D_flatten_fp32 << <rwpk_flatten._grid_dims, rwpk_flatten._block_dims,
        0, S->get_raw_stream_ref() >> > ((float4*)src_ptr, (float*)_kp_configs->get_src(), rwpk_flatten._calc_pitch_src, rwpk_flatten._calc_proc_dims);

    if (_more_than_flatten) {
        decx::reduce::cuda_reduce1D_sum_fp32_caller_Async(_kp_configs, S);

        return _kp_configs->get_dst();
    }
    else {
        return _kp_configs->get_src();
    }
}




const void* decx::reduce::reduce_sum2D_full_i32_Async(decx::reduce::cuda_reduce1D_configs<int32_t>* _kp_configs, 
                                                const void*                                 src_ptr, 
                                                const uint2                                 proc_dims,
                                                decx::cuda_stream*                          S,
                                                const bool                                  _more_than_flatten)
{
    decx::reduce::RWPK_2D rwpk_flatten = _kp_configs->get_rwpk_flatten();
    
    decx::reduce::GPUK::cu_warp_reduce_sum2D_flatten_i32 << <rwpk_flatten._grid_dims, rwpk_flatten._block_dims,
        0, S->get_raw_stream_ref() >> > ((int4*)src_ptr, (int32_t*)_kp_configs->get_src(), rwpk_flatten._calc_pitch_src, rwpk_flatten._calc_proc_dims);

    if (_more_than_flatten) {
        decx::reduce::cuda_reduce1D_sum_i32_caller_Async(_kp_configs, S);

        return _kp_configs->get_dst();
    }
    else {
        return _kp_configs->get_src();
    }
}


const void* decx::reduce::reduce_sum2D_full_fp64_Async(decx::reduce::cuda_reduce1D_configs<double>* _kp_configs, 
                                                       const void*                                 src_ptr, 
                                                       const uint2                                 proc_dims,
                                                       decx::cuda_stream*                          S,
                                                       const bool                                  _more_than_flatten)
{
    decx::reduce::RWPK_2D rwpk_flatten = _kp_configs->get_rwpk_flatten();
    
    decx::reduce::GPUK::cu_warp_reduce_sum2D_flatten_fp64 << <rwpk_flatten._grid_dims, rwpk_flatten._block_dims,
        0, S->get_raw_stream_ref() >> > ((double2*)src_ptr, (double*)_kp_configs->get_src(), rwpk_flatten._calc_pitch_src, rwpk_flatten._calc_proc_dims);

    if (_more_than_flatten) {
        decx::reduce::cuda_reduce1D_sum_fp64_caller_Async(_kp_configs, S);

        return _kp_configs->get_dst();
    }
    else {
        return _kp_configs->get_src();
    }
}



const void* decx::reduce::reduce_sum2D_full_fp16_fp32_Async(decx::reduce::cuda_reduce1D_configs<float>* _kp_configs, 
                                                            const void*                                 src_ptr, 
                                                            const uint2                                 proc_dims,
                                                            decx::cuda_stream*                          S,
                                                            const bool                                  _more_than_flatten)
{
    decx::reduce::RWPK_2D rwpk_flatten = _kp_configs->get_rwpk_flatten();
    
    decx::reduce::GPUK::cu_warp_reduce_sum2D_flatten_fp16_L1 << <rwpk_flatten._grid_dims, rwpk_flatten._block_dims,
        0, S->get_raw_stream_ref() >> > ((float4*)src_ptr, (float*)_kp_configs->get_src(), rwpk_flatten._calc_pitch_src, rwpk_flatten._calc_proc_dims);
    
    if (_more_than_flatten) {
        decx::reduce::cuda_reduce1D_sum_fp32_caller_Async(_kp_configs, S);

        return _kp_configs->get_dst();
    }
    else {
        return _kp_configs->get_src();
    }
}




const void* decx::reduce::reduce_sum2D_full_fp16_Async(decx::reduce::cuda_reduce1D_configs<de::Half>*   _kp_configs, 
                                                       const void*                                      src_ptr, 
                                                       const uint2                                      proc_dims,
                                                       decx::cuda_stream*                               S,
                                                       const bool                                       _more_than_flatten,
                                                       const uint32_t                                   _fp16_accu)
{
    decx::reduce::RWPK_2D rwpk_flatten = _kp_configs->get_rwpk_flatten();
    
    switch (_fp16_accu)
    {
    case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L2:
        decx::reduce::GPUK::cu_warp_reduce_sum2D_flatten_fp16_L2 << <rwpk_flatten._grid_dims, rwpk_flatten._block_dims,
            0, S->get_raw_stream_ref() >> > ((float4*)src_ptr, (__half*)_kp_configs->get_src(), rwpk_flatten._calc_pitch_src, rwpk_flatten._calc_proc_dims);
        break;

    case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L3:
        decx::reduce::GPUK::cu_warp_reduce_sum2D_flatten_fp16_L3 << <rwpk_flatten._grid_dims, rwpk_flatten._block_dims,
            0, S->get_raw_stream_ref() >> > ((float4*)src_ptr, (__half*)_kp_configs->get_src(), rwpk_flatten._calc_pitch_src, rwpk_flatten._calc_proc_dims);
        break;
    default:
        break;
    }
    
    if (_more_than_flatten) {
        decx::reduce::cuda_reduce1D_sum_fp16_caller_Async(_kp_configs, S, _fp16_accu);

        return _kp_configs->get_dst();
    }
    else {
        return _kp_configs->get_src();
    }
}



const void* decx::reduce::reduce_sum2D_full_u8_i32_Async(decx::reduce::cuda_reduce1D_configs<int32_t>*  _kp_configs, 
                                                         const void*                                    src_ptr, 
                                                         const uint2                                    proc_dims,
                                                         decx::cuda_stream*                             S,
                                                         const bool                                     _more_than_flatten)
{
    decx::reduce::RWPK_2D rwpk_flatten = _kp_configs->get_rwpk_flatten();
    
    decx::reduce::GPUK::cu_warp_reduce_sum2D_flatten_u8_i32 << <rwpk_flatten._grid_dims, rwpk_flatten._block_dims,
        0, S->get_raw_stream_ref() >> > ((int4*)src_ptr, (int32_t*)_kp_configs->get_src(), rwpk_flatten._calc_pitch_src, rwpk_flatten._calc_proc_dims);

    if (_more_than_flatten) {
        decx::reduce::cuda_reduce1D_sum_i32_caller_Async(_kp_configs, S);

        return _kp_configs->get_dst();
    }
    else {
        return _kp_configs->get_src();
    }
}