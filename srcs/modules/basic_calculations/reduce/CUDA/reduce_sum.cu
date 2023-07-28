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



template <bool _src_from_device>
void decx::reduce::cuda_reduce1D_sum_fp32_caller_Async(decx::reduce::cuda_reduce1D_configs<float>* _kp_configs, decx::cuda_stream* S)
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

    decx::reduce::GPUK::cu_block_reduce_sum1D_fp32 << <grid_len, _REDUCE1D_BLOCK_DIM_,
        0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr, proc_len_v, proc_len_v1);

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

            decx::reduce::GPUK::cu_block_reduce_sum1D_fp32 << <grid_len, _REDUCE1D_BLOCK_DIM_,
                0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr, proc_len_v, proc_len_v1);

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


template void decx::reduce::cuda_reduce1D_sum_fp32_caller_Async<true> (decx::reduce::cuda_reduce1D_configs<float>*    _kp_configs, decx::cuda_stream* S);
template void decx::reduce::cuda_reduce1D_sum_fp32_caller_Async<false>(decx::reduce::cuda_reduce1D_configs<float>*    _kp_configs, decx::cuda_stream* S);



template <bool _src_from_device>
void decx::reduce::cuda_reduce1D_sum_u8_i32_caller_Async(decx::reduce::cuda_reduce1D_configs<uint8_t>* _kp_configs, decx::cuda_stream* S)
{
    uint64_t proc_len_v = _kp_configs->get_proc_len() / 16;
    uint64_t proc_len_v1 = _kp_configs->get_actual_len();
    uint64_t grid_len = decx::utils::ceil<uint64_t>(proc_len_v, _REDUCE1D_BLOCK_DIM_);

    const int4* read_ptr = NULL;
    int32_t* write_ptr = NULL;

    if (_src_from_device) {
        read_ptr = (int4*)_kp_configs->get_dev_src().ptr;
        write_ptr = (int32_t*)_kp_configs->get_dev_tmp1().ptr;
    }
    else {
        read_ptr = (int4*)_kp_configs->get_leading_MIF().mem;
        write_ptr = (int32_t*)_kp_configs->get_lagging_MIF().mem;
    }

    decx::reduce::GPUK::cu_block_reduce_sum1D_u8_i32 << <grid_len, _REDUCE1D_BLOCK_DIM_,
        0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr, proc_len_v, proc_len_v1);

    _kp_configs->inverse_mutex_MIF_states();

    if (grid_len > 1) 
    {
        proc_len_v1 = grid_len;
        proc_len_v = decx::utils::ceil<uint64_t>(proc_len_v1, 4);
        grid_len = decx::utils::ceil<uint64_t>(proc_len_v, _REDUCE1D_BLOCK_DIM_);

        while (true)
        {
            read_ptr = (int4*)_kp_configs->get_leading_MIF().mem;
            write_ptr = (int32_t*)_kp_configs->get_lagging_MIF().mem;

            decx::reduce::GPUK::cu_block_reduce_sum1D_int32 << <grid_len, _REDUCE1D_BLOCK_DIM_,
                0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr, proc_len_v, proc_len_v1);

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


template void decx::reduce::cuda_reduce1D_sum_u8_i32_caller_Async<true> (decx::reduce::cuda_reduce1D_configs<uint8_t>*  _kp_configs, decx::cuda_stream* S);
template void decx::reduce::cuda_reduce1D_sum_u8_i32_caller_Async<false>(decx::reduce::cuda_reduce1D_configs<uint8_t>*  _kp_configs, decx::cuda_stream* S);



template <bool _src_from_device>
void decx::reduce::cuda_reduce1D_sum_fp16_fp32_caller_Async(decx::reduce::cuda_reduce1D_configs<de::Half>* _kp_configs, decx::cuda_stream* S)
{
    uint64_t proc_len_v = _kp_configs->get_proc_len() / 8;
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

    decx::reduce::GPUK::cu_block_reduce_sum1D_fp16_fp32 << <grid_len, _REDUCE1D_BLOCK_DIM_,
        0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr, proc_len_v, proc_len_v1);

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

            decx::reduce::GPUK::cu_block_reduce_sum1D_fp32 << <grid_len, _REDUCE1D_BLOCK_DIM_,
                0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr, proc_len_v, proc_len_v1);

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

template void decx::reduce::cuda_reduce1D_sum_fp16_fp32_caller_Async<true> (decx::reduce::cuda_reduce1D_configs<de::Half>* _kp_configs, decx::cuda_stream* S);
template void decx::reduce::cuda_reduce1D_sum_fp16_fp32_caller_Async<false>(decx::reduce::cuda_reduce1D_configs<de::Half>* _kp_configs, decx::cuda_stream* S);