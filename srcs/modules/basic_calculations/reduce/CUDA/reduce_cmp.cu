/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
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

    for (int i = 0; i < _rwpks.size() - 1; ++i) {
        _rwpk = _rwpks[i];
        
        decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp32<_is_max> << <_rwpk._grid_dims, _rwpk._block_dims,
            0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src,       (float*)_rwpk._dst, 
                                             _rwpk._calc_pitch_src,     _rwpk._calc_pitch_dst,      _rwpk._calc_proc_dims);
    }
    // cout all the parameters
    _rwpk = _rwpks[_rwpks.size() - 1];

    decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp32_transp<_is_max> << <_rwpk._grid_dims, _rwpk._block_dims,
        0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src,       (float*)_rwpk._dst, 
                                         _rwpk._calc_pitch_src,     _rwpk._calc_pitch_dst,      _rwpk._calc_proc_dims);
}

template void decx::reduce::reduce_cmp2D_h_fp32_Async<true>(decx::reduce::cuda_reduce2D_1way_configs<float>* _configs, decx::cuda_stream* S);
template void decx::reduce::reduce_cmp2D_h_fp32_Async<false>(decx::reduce::cuda_reduce2D_1way_configs<float>* _configs, decx::cuda_stream* S);




template <bool _is_max>
void decx::reduce::reduce_cmp2D_v_fp32_Async(decx::reduce::cuda_reduce2D_1way_configs<float>* _configs, decx::cuda_stream* S)
{
    /*const float4* read_ptr = NULL;
    float4* write_ptr = NULL;

    uint32_t grid_y = decx::utils::ceil<uint32_t>(_configs->get_actual_proc_dims().y, _REDUCE2D_BLOCK_DIM_Y_);
    const uint32_t grid_x = decx::utils::ceil<uint32_t>(_configs->get_actual_proc_dims().x, _REDUCE2D_BLOCK_DIM_X_ * 4);

    uint2 _proc_dims_v4 = _configs->get_proc_dims_v();

    const uint32_t Wsrc_v4 = _configs->get_proc_dims_v().x;
    const uint32_t Wdst_v4 = Wsrc_v4;

    while (true)
    {
        read_ptr = (float4*)_configs->get_leading_ptr();
        write_ptr = (float4*)_configs->get_lagging_ptr();

        _configs->reverse_MIF_states();

        decx::reduce::GPUK::cu_warp_reduce_cmp2D_v_fp32<_is_max> << <dim3(grid_x, grid_y), dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_),
            0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr, Wsrc_v4, Wdst_v4, _proc_dims_v4);

        if (grid_y == 1) {
            break;
        }

        _proc_dims_v4.y = grid_y;
        grid_y = decx::utils::ceil<uint32_t>(_proc_dims_v4.y, _REDUCE2D_BLOCK_DIM_Y_);
    }*/

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

    for (int i = 0; i < _rwpks.size() - 1; ++i) {
        _rwpk = _rwpks[i];
        decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp16<_is_max> << <_rwpk._grid_dims, _rwpk._block_dims,
            0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src,       (__half*)_rwpk._dst, 
                                                _rwpk._calc_pitch_src,     _rwpk._calc_pitch_dst,      _rwpk._calc_proc_dims);
    }

    _rwpk = _rwpks[_rwpks.size() - 1];
    decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp16_transp<_is_max> << <_rwpk._grid_dims, _rwpk._block_dims,
        0, S->get_raw_stream_ref() >> > ((float4*)_rwpk._src,       (__half*)_rwpk._dst, 
                                            _rwpk._calc_pitch_src,     _rwpk._calc_pitch_dst,      _rwpk._calc_proc_dims);
}

template void decx::reduce::reduce_cmp2D_h_fp16_Async<true>(decx::reduce::cuda_reduce2D_1way_configs<de::Half>*, decx::cuda_stream*);
template void decx::reduce::reduce_cmp2D_h_fp16_Async<false>(decx::reduce::cuda_reduce2D_1way_configs<de::Half>*, decx::cuda_stream*);




template <bool _is_max>
void decx::reduce::reduce_cmp2D_h_u8_Async(decx::reduce::cuda_reduce2D_1way_configs<uint8_t>* _configs, decx::cuda_stream* S)
{
    const std::vector<decx::reduce::RWPK_2D>& _rwpks = _configs->get_rwpks();
    decx::reduce::RWPK_2D _rwpk;

    for (int i = 0; i < _rwpks.size() - 1; ++i) {
        _rwpk = _rwpks[i];

        decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_u8<_is_max> << <_rwpk._grid_dims, _rwpk._block_dims,
            0, S->get_raw_stream_ref() >> > ((int4*)_rwpk._src, (uint8_t*)_rwpk._dst,
                _rwpk._calc_pitch_src, _rwpk._calc_pitch_dst, _rwpk._calc_proc_dims);
    }

    _rwpk = _rwpks[_rwpks.size() - 1];

    decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_u8_transp<_is_max> << <_rwpk._grid_dims, _rwpk._block_dims,
        0, S->get_raw_stream_ref() >> > ((int4*)_rwpk._src, (uint8_t*)_rwpk._dst,
            _rwpk._calc_pitch_src, _rwpk._calc_pitch_dst, _rwpk._calc_proc_dims);
}

template void decx::reduce::reduce_cmp2D_h_u8_Async<true>(decx::reduce::cuda_reduce2D_1way_configs<uint8_t>*, decx::cuda_stream*);
template void decx::reduce::reduce_cmp2D_h_u8_Async<false>(decx::reduce::cuda_reduce2D_1way_configs<uint8_t>*, decx::cuda_stream*);



template <bool _is_max, bool _src_from_device>
void decx::reduce::reduce_cmp2D_full_fp32_Async(decx::reduce::cuda_reduce1D_configs<float>* _kp_configs, const uint2 proc_dims, 
    const uint32_t _pitch_src_v4, decx::cuda_stream* S)
{
    //const uint2 proc_dims_v4 = make_uint2(decx::utils::ceil<uint32_t>(proc_dims.x, 4), proc_dims.y);
    //const dim3 _flatten_K_grid = dim3(decx::utils::ceil<uint32_t>(proc_dims.x, _REDUCE2D_BLOCK_DIM_X_ * 4),
    //                                  decx::utils::ceil<uint32_t>(proc_dims.y, _REDUCE2D_BLOCK_DIM_Y_));

    //const float4* read_ptr = NULL;
    //float* write_ptr = NULL;

    //if (_src_from_device) {
    //    read_ptr = (float4*)_kp_configs->get_dev_src().ptr;
    //    write_ptr = (float*)_kp_configs->get_dev_tmp1().ptr;
    //}
    //else {
    //    read_ptr = (float4*)_kp_configs->get_leading_MIF().mem;
    //    write_ptr = (float*)_kp_configs->get_lagging_MIF().mem;
    //}

    //decx::reduce::GPUK::cu_warp_reduce_cmp2D_1D_fp32<_is_max> << <_flatten_K_grid, dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_),
    //    0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr, _pitch_src_v4, proc_dims, _kp_configs->get_fill_val());

    //_kp_configs->inverse_mutex_MIF_states();

    //// 1D-layout array processing
    //uint64_t grid_len = _flatten_K_grid.x * _flatten_K_grid.y;
    //uint64_t proc_len_v1, proc_len_v;

    //if (grid_len > 1)
    //{
    //    proc_len_v1 = grid_len;
    //    proc_len_v = decx::utils::ceil<uint64_t>(proc_len_v1, 4);
    //    grid_len = decx::utils::ceil<uint64_t>(proc_len_v, _REDUCE1D_BLOCK_DIM_);

    //    while (true)
    //    {
    //        read_ptr = (float4*)_kp_configs->get_leading_MIF().mem;
    //        write_ptr = (float*)_kp_configs->get_lagging_MIF().mem;

    //        decx::reduce::GPUK::cu_block_reduce_cmp1D_fp32<_is_max> << <grid_len, _REDUCE1D_BLOCK_DIM_,
    //            0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr, proc_len_v, proc_len_v1, _kp_configs->get_fill_val());
    //        
    //        _kp_configs->inverse_mutex_MIF_states();

    //        if (grid_len == 1) {
    //            break;
    //        }

    //        proc_len_v1 = grid_len;
    //        proc_len_v = decx::utils::ceil<uint64_t>(proc_len_v1, 4);
    //        grid_len = decx::utils::ceil<uint64_t>(proc_len_v, _REDUCE1D_BLOCK_DIM_);
    //    }
    //}
}

template void decx::reduce::reduce_cmp2D_full_fp32_Async<true, true>(decx::reduce::cuda_reduce1D_configs<float>*, const uint2, const uint32_t, decx::cuda_stream*); 
template void decx::reduce::reduce_cmp2D_full_fp32_Async<true, false>(decx::reduce::cuda_reduce1D_configs<float>*, const uint2, const uint32_t, decx::cuda_stream*);
template void decx::reduce::reduce_cmp2D_full_fp32_Async<false, true>(decx::reduce::cuda_reduce1D_configs<float>*, const uint2, const uint32_t, decx::cuda_stream*);
template void decx::reduce::reduce_cmp2D_full_fp32_Async<false, false>(decx::reduce::cuda_reduce1D_configs<float>*, const uint2, const uint32_t, decx::cuda_stream*);