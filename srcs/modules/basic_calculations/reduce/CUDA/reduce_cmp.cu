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


template <bool _src_from_device, bool _is_max>
void decx::reduce::cuda_reduce1D_cmp_fp32_caller_Async(decx::reduce::cuda_reduce1D_configs<float>* _kp_configs, decx::cuda_stream* S)
{
    uint64_t proc_len_v = _kp_configs->get_proc_len() / 4;
    uint64_t proc_len_v1 = _kp_configs->get_actual_len();
    uint64_t grid_len = decx::utils::ceil<uint64_t>(proc_len_v, _REDUCE1D_BLOCK_DIM_);

    const float4* read_ptr = NULL;
    float* write_ptr = NULL;

    if (_src_from_device) {
        read_ptr = (float4*)_kp_configs->get_dev_src().ptr;
        write_ptr = (float*)_kp_configs->get_dev_tmp1().ptr;
    }
    else {
        read_ptr = (float4*)_kp_configs->get_leading_MIF().mem;
        write_ptr = (float*)_kp_configs->get_lagging_MIF().mem;
    }

    decx::reduce::GPUK::cu_block_reduce_cmp1D_fp32<_is_max> << <grid_len, _REDUCE1D_BLOCK_DIM_,
        0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr, proc_len_v, proc_len_v1, _kp_configs->get_fill_val());

    _kp_configs->inverse_mutex_MIF_states();

    if (grid_len > 1)
    {
        proc_len_v1 = grid_len;
        proc_len_v = decx::utils::ceil<uint64_t>(proc_len_v1, 4);
        grid_len = decx::utils::ceil<uint64_t>(proc_len_v, _REDUCE1D_BLOCK_DIM_);

        while (true)
        {
            read_ptr = (float4*)_kp_configs->get_leading_MIF().mem;
            write_ptr = (float*)_kp_configs->get_lagging_MIF().mem;

            decx::reduce::GPUK::cu_block_reduce_cmp1D_fp32<_is_max> << <grid_len, _REDUCE1D_BLOCK_DIM_,
                0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr, proc_len_v, proc_len_v1, _kp_configs->get_fill_val());

            _kp_configs->inverse_mutex_MIF_states();

            if (grid_len == 1) {
                break;
            }

            proc_len_v1 = grid_len;
            proc_len_v = decx::utils::ceil<uint64_t>(proc_len_v1, 4);
            grid_len = decx::utils::ceil<uint64_t>(proc_len_v, _REDUCE1D_BLOCK_DIM_);
        }
    }
}

template void decx::reduce::cuda_reduce1D_cmp_fp32_caller_Async<true, true> (decx::reduce::cuda_reduce1D_configs<float>*    _kp_configs, decx::cuda_stream* S);
template void decx::reduce::cuda_reduce1D_cmp_fp32_caller_Async<false, true>(decx::reduce::cuda_reduce1D_configs<float>*    _kp_configs, decx::cuda_stream* S);
template void decx::reduce::cuda_reduce1D_cmp_fp32_caller_Async<true, false>(decx::reduce::cuda_reduce1D_configs<float>* _kp_configs, decx::cuda_stream* S);
template void decx::reduce::cuda_reduce1D_cmp_fp32_caller_Async<false, false>(decx::reduce::cuda_reduce1D_configs<float>* _kp_configs, decx::cuda_stream* S);



template <bool _src_from_device, bool _is_max>
void decx::reduce::cuda_reduce1D_cmp_fp64_caller_Async(decx::reduce::cuda_reduce1D_configs<double>* _kp_configs, decx::cuda_stream* S)
{
    uint64_t proc_len_v = _kp_configs->get_proc_len() / 2;
    uint64_t proc_len_v1 = _kp_configs->get_actual_len();
    uint64_t grid_len = decx::utils::ceil<uint64_t>(proc_len_v, _REDUCE1D_BLOCK_DIM_);

    const double2* read_ptr = NULL;
    double* write_ptr = NULL;

    if (_src_from_device) {
        read_ptr = (double2*)_kp_configs->get_dev_src().ptr;
        write_ptr = (double*)_kp_configs->get_dev_tmp1().ptr;
    }
    else {
        read_ptr = (double2*)_kp_configs->get_leading_MIF().mem;
        write_ptr = (double*)_kp_configs->get_lagging_MIF().mem;
    }

    decx::reduce::GPUK::cu_block_reduce_cmp1D_fp64<_is_max> << <grid_len, _REDUCE1D_BLOCK_DIM_,
        0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr, proc_len_v, proc_len_v1, _kp_configs->get_fill_val());

    _kp_configs->inverse_mutex_MIF_states();

    if (grid_len > 1)
    {
        proc_len_v1 = grid_len;
        proc_len_v = decx::utils::ceil<uint64_t>(proc_len_v1, 2);
        grid_len = decx::utils::ceil<uint64_t>(proc_len_v, _REDUCE1D_BLOCK_DIM_);

        while (true)
        {
            read_ptr = (double2*)_kp_configs->get_leading_MIF().mem;
            write_ptr = (double*)_kp_configs->get_lagging_MIF().mem;

            decx::reduce::GPUK::cu_block_reduce_cmp1D_fp64<_is_max> << <grid_len, _REDUCE1D_BLOCK_DIM_,
                0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr, proc_len_v, proc_len_v1, _kp_configs->get_fill_val());

            _kp_configs->inverse_mutex_MIF_states();

            if (grid_len == 1) {
                break;
            }

            proc_len_v1 = grid_len;
            proc_len_v = decx::utils::ceil<uint64_t>(proc_len_v1, 2);
            grid_len = decx::utils::ceil<uint64_t>(proc_len_v, _REDUCE1D_BLOCK_DIM_);
        }
    }
}

template void decx::reduce::cuda_reduce1D_cmp_fp64_caller_Async<true, true>(decx::reduce::cuda_reduce1D_configs<double>* _kp_configs, decx::cuda_stream* S);
template void decx::reduce::cuda_reduce1D_cmp_fp64_caller_Async<false, true>(decx::reduce::cuda_reduce1D_configs<double>* _kp_configs, decx::cuda_stream* S);
template void decx::reduce::cuda_reduce1D_cmp_fp64_caller_Async<true, false>(decx::reduce::cuda_reduce1D_configs<double>* _kp_configs, decx::cuda_stream* S);
template void decx::reduce::cuda_reduce1D_cmp_fp64_caller_Async<false, false>(decx::reduce::cuda_reduce1D_configs<double>* _kp_configs, decx::cuda_stream* S);




template <bool _src_from_device, bool _is_max>
void decx::reduce::cuda_reduce1D_cmp_fp16_caller_Async(decx::reduce::cuda_reduce1D_configs<de::Half>* _kp_configs, decx::cuda_stream* S)
{
    uint64_t proc_len_v = _kp_configs->get_proc_len() / 8;
    uint64_t proc_len_v1 = _kp_configs->get_actual_len();
    uint64_t grid_len = decx::utils::ceil<uint64_t>(proc_len_v, _REDUCE1D_BLOCK_DIM_);

    const de::Half fill_val = _kp_configs->get_fill_val();

    const float4* read_ptr = NULL;
    __half* write_ptr = NULL;

    if (_src_from_device) {
        read_ptr = (float4*)_kp_configs->get_dev_src().ptr;
        write_ptr = (__half*)_kp_configs->get_dev_tmp1().ptr;
    }
    else {
        read_ptr = (float4*)_kp_configs->get_leading_MIF().mem;
        write_ptr = (__half*)_kp_configs->get_lagging_MIF().mem;
    }

    decx::reduce::GPUK::cu_block_reduce_cmp1D_fp16<_is_max> << <grid_len, _REDUCE1D_BLOCK_DIM_,
        0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr, proc_len_v, proc_len_v1, *((__half*)&fill_val));

    _kp_configs->inverse_mutex_MIF_states();

    if (grid_len > 1)
    {
        proc_len_v1 = grid_len;
        proc_len_v = decx::utils::ceil<uint64_t>(proc_len_v1, 8);
        grid_len = decx::utils::ceil<uint64_t>(proc_len_v, _REDUCE1D_BLOCK_DIM_);

        while (true)
        {
            read_ptr = (float4*)_kp_configs->get_leading_MIF().mem;
            write_ptr = (__half*)_kp_configs->get_lagging_MIF().mem;

            decx::reduce::GPUK::cu_block_reduce_cmp1D_fp16<_is_max> << <grid_len, _REDUCE1D_BLOCK_DIM_,
                0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr, proc_len_v, proc_len_v1, *((__half*)&fill_val));

            _kp_configs->inverse_mutex_MIF_states();

            if (grid_len == 1) {
                break;
            }

            proc_len_v1 = grid_len;
            proc_len_v = decx::utils::ceil<uint64_t>(proc_len_v1, 8);
            grid_len = decx::utils::ceil<uint64_t>(proc_len_v, _REDUCE1D_BLOCK_DIM_);
        }
    }
}

template void decx::reduce::cuda_reduce1D_cmp_fp16_caller_Async<true, true> (decx::reduce::cuda_reduce1D_configs<de::Half>*    _kp_configs, decx::cuda_stream* S);
template void decx::reduce::cuda_reduce1D_cmp_fp16_caller_Async<false, true>(decx::reduce::cuda_reduce1D_configs<de::Half>*    _kp_configs, decx::cuda_stream* S);
template void decx::reduce::cuda_reduce1D_cmp_fp16_caller_Async<true, false>(decx::reduce::cuda_reduce1D_configs<de::Half>* _kp_configs, decx::cuda_stream* S);
template void decx::reduce::cuda_reduce1D_cmp_fp16_caller_Async<false, false>(decx::reduce::cuda_reduce1D_configs<de::Half>* _kp_configs, decx::cuda_stream* S);




template <bool _src_from_device, bool _is_max>
void decx::reduce::cuda_reduce1D_cmp_u8_caller_Async(decx::reduce::cuda_reduce1D_configs<uint8_t>* _kp_configs, decx::cuda_stream* S)
{
    uint64_t proc_len_v16 = _kp_configs->get_proc_len() / 16;
    uint64_t proc_len_v1 = _kp_configs->get_actual_len();
    uint64_t grid_len = decx::utils::ceil<uint64_t>(proc_len_v16, _REDUCE1D_BLOCK_DIM_);

    const float4* read_ptr = NULL;
    uint8_t* write_ptr = NULL;

    if (_src_from_device) {
        read_ptr = (float4*)_kp_configs->get_dev_src().ptr;
        write_ptr = (uint8_t*)_kp_configs->get_dev_tmp1().ptr;
    }
    else {
        read_ptr = (float4*)_kp_configs->get_leading_MIF().mem;
        write_ptr = (uint8_t*)_kp_configs->get_lagging_MIF().mem;
    }

    decx::reduce::GPUK::cu_block_reduce_cmp1D_u8<_is_max> << <grid_len, _REDUCE1D_BLOCK_DIM_,
        0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr, proc_len_v16, proc_len_v1, _kp_configs->get_fill_val());

    _kp_configs->inverse_mutex_MIF_states();

    if (grid_len > 1)
    {
        proc_len_v1 = grid_len;
        proc_len_v16 = decx::utils::ceil<uint64_t>(proc_len_v1, 16);
        grid_len = decx::utils::ceil<uint64_t>(proc_len_v16, _REDUCE1D_BLOCK_DIM_);

        while (true)
        {
            read_ptr = (float4*)_kp_configs->get_leading_MIF().mem;
            write_ptr = (uint8_t*)_kp_configs->get_lagging_MIF().mem;

            decx::reduce::GPUK::cu_block_reduce_cmp1D_u8<_is_max> << <grid_len, _REDUCE1D_BLOCK_DIM_,
                0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr, proc_len_v16, proc_len_v1, _kp_configs->get_fill_val());

            _kp_configs->inverse_mutex_MIF_states();

            if (grid_len == 1) {
                break;
            }

            proc_len_v1 = grid_len;
            proc_len_v16 = decx::utils::ceil<uint64_t>(proc_len_v1, 16);
            grid_len = decx::utils::ceil<uint64_t>(proc_len_v16, _REDUCE1D_BLOCK_DIM_);
        }
    }
}


template void decx::reduce::cuda_reduce1D_cmp_u8_caller_Async<true, true>(decx::reduce::cuda_reduce1D_configs<uint8_t>* _kp_configs, decx::cuda_stream* S);
template void decx::reduce::cuda_reduce1D_cmp_u8_caller_Async<false, true>(decx::reduce::cuda_reduce1D_configs<uint8_t>* _kp_configs, decx::cuda_stream* S);
template void decx::reduce::cuda_reduce1D_cmp_u8_caller_Async<true, false>(decx::reduce::cuda_reduce1D_configs<uint8_t>* _kp_configs, decx::cuda_stream* S);
template void decx::reduce::cuda_reduce1D_cmp_u8_caller_Async<false, false>(decx::reduce::cuda_reduce1D_configs<uint8_t>* _kp_configs, decx::cuda_stream* S);



template <bool _is_max>
void decx::reduce::reduce_cmp2D_h_fp32_Async(decx::reduce::cuda_reduce2D_1way_configs<float>* _configs, decx::cuda_stream* S)
{
    const float4* read_ptr = NULL;
    float* write_ptr = NULL;

    uint32_t grid_x = decx::utils::ceil<uint32_t>(_configs->get_proc_dims_v().x, _REDUCE2D_BLOCK_DIM_X_);
    const uint32_t grid_y = decx::utils::ceil<uint32_t>(_configs->get_actual_proc_dims().y, _REDUCE2D_BLOCK_DIM_Y_);

    uint2 proc_dims_actual = _configs->get_actual_proc_dims();

    uint32_t Wdsrc_v4_varient = _configs->get_dtmp1()._dims.x / 4;
    uint32_t Wddst_v1_varient = decx::utils::ceil<uint32_t>(grid_x, 4) * 4;

    for (int i = 0; i < _configs->get_kernel_call_times() - 1; ++i)
    {
        read_ptr = (float4*)_configs->get_leading_ptr();
        write_ptr = (float*)_configs->get_lagging_ptr();

        _configs->reverse_MIF_states();

        decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp32<_is_max> << <dim3(grid_x, grid_y), dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_),
            0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr, Wdsrc_v4_varient, Wddst_v1_varient, proc_dims_actual);

        Wdsrc_v4_varient = decx::utils::ceil<uint32_t>(grid_x, 4);
        proc_dims_actual.x = grid_x;

        grid_x = decx::utils::ceil<uint32_t>(decx::utils::ceil<uint32_t>(proc_dims_actual.x, 4), _REDUCE2D_BLOCK_DIM_X_);
        Wddst_v1_varient = decx::utils::ceil<uint32_t>(grid_x, 4) * 4;
    }

    read_ptr = (float4*)_configs->get_leading_ptr();
    write_ptr = (float*)_configs->get_lagging_ptr();

    _configs->reverse_MIF_states();

    decx::reduce::GPUK::cu_warp_reduce_cmp2D_h_fp32_transp<_is_max> << <dim3(grid_x, grid_y), dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_),
        0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr, Wdsrc_v4_varient, proc_dims_actual.y, proc_dims_actual);
}

template void decx::reduce::reduce_cmp2D_h_fp32_Async<true>(decx::reduce::cuda_reduce2D_1way_configs<float>* _configs, decx::cuda_stream* S);
template void decx::reduce::reduce_cmp2D_h_fp32_Async<false>(decx::reduce::cuda_reduce2D_1way_configs<float>* _configs, decx::cuda_stream* S);




template <bool _is_max>
void decx::reduce::reduce_cmp2D_v_fp32_Async(decx::reduce::cuda_reduce2D_1way_configs<float>* _configs, decx::cuda_stream* S)
{
    const float4* read_ptr = NULL;
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
    }
}

template void decx::reduce::reduce_cmp2D_v_fp32_Async<true>(decx::reduce::cuda_reduce2D_1way_configs<float>*, decx::cuda_stream*); 
template void decx::reduce::reduce_cmp2D_v_fp32_Async<false>(decx::reduce::cuda_reduce2D_1way_configs<float>*, decx::cuda_stream*);





template <bool _is_max, bool _src_from_device>
void decx::reduce::reduce_cmp2D_full_fp32_Async(decx::reduce::cuda_reduce1D_configs<float>* _kp_configs, const uint2 proc_dims, 
    const uint32_t _pitch_src_v4, decx::cuda_stream* S)
{
    const uint2 proc_dims_v4 = make_uint2(decx::utils::ceil<uint32_t>(proc_dims.x, 4), proc_dims.y);
    const dim3 _flatten_K_grid = dim3(decx::utils::ceil<uint32_t>(proc_dims.x, _REDUCE2D_BLOCK_DIM_X_ * 4),
                                      decx::utils::ceil<uint32_t>(proc_dims.y, _REDUCE2D_BLOCK_DIM_Y_));

    const float4* read_ptr = NULL;
    float* write_ptr = NULL;

    if (_src_from_device) {
        read_ptr = (float4*)_kp_configs->get_dev_src().ptr;
        write_ptr = (float*)_kp_configs->get_dev_tmp1().ptr;
    }
    else {
        read_ptr = (float4*)_kp_configs->get_leading_MIF().mem;
        write_ptr = (float*)_kp_configs->get_lagging_MIF().mem;
    }

    decx::reduce::GPUK::cu_warp_reduce_cmp2D_1D_fp32<_is_max> << <_flatten_K_grid, dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_),
        0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr, _pitch_src_v4, proc_dims, _kp_configs->get_fill_val());

    _kp_configs->inverse_mutex_MIF_states();

    // 1D-layout array processing
    uint64_t grid_len = _flatten_K_grid.x * _flatten_K_grid.y;
    uint64_t proc_len_v1, proc_len_v;

    if (grid_len > 1)
    {
        proc_len_v1 = grid_len;
        proc_len_v = decx::utils::ceil<uint64_t>(proc_len_v1, 4);
        grid_len = decx::utils::ceil<uint64_t>(proc_len_v, _REDUCE1D_BLOCK_DIM_);

        while (true)
        {
            read_ptr = (float4*)_kp_configs->get_leading_MIF().mem;
            write_ptr = (float*)_kp_configs->get_lagging_MIF().mem;

            decx::reduce::GPUK::cu_block_reduce_cmp1D_fp32<_is_max> << <grid_len, _REDUCE1D_BLOCK_DIM_,
                0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr, proc_len_v, proc_len_v1, _kp_configs->get_fill_val());
            
            _kp_configs->inverse_mutex_MIF_states();

            if (grid_len == 1) {
                break;
            }

            proc_len_v1 = grid_len;
            proc_len_v = decx::utils::ceil<uint64_t>(proc_len_v1, 4);
            grid_len = decx::utils::ceil<uint64_t>(proc_len_v, _REDUCE1D_BLOCK_DIM_);
        }
    }
}

template void decx::reduce::reduce_cmp2D_full_fp32_Async<true, true>(decx::reduce::cuda_reduce1D_configs<float>*, const uint2, const uint32_t, decx::cuda_stream*); 
template void decx::reduce::reduce_cmp2D_full_fp32_Async<true, false>(decx::reduce::cuda_reduce1D_configs<float>*, const uint2, const uint32_t, decx::cuda_stream*);
template void decx::reduce::reduce_cmp2D_full_fp32_Async<false, true>(decx::reduce::cuda_reduce1D_configs<float>*, const uint2, const uint32_t, decx::cuda_stream*);
template void decx::reduce::reduce_cmp2D_full_fp32_Async<false, false>(decx::reduce::cuda_reduce1D_configs<float>*, const uint2, const uint32_t, decx::cuda_stream*);