/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _FFT1D_RADIX_5_KERNEL_CUH_
#define _FFT1D_RADIX_5_KERNEL_CUH_

#include "../../../../core/basic.h"
#include "../../../../classes/classes_util.h"
#include "../../fft_utils.h"


namespace decx {
    namespace signal {
        namespace GPUK {
            __global__
            void cu_FFT1D_R5_R2C_first(const float* __restrict src, float2* __restrict dst, const size_t B_ops_num);



            __global__
            void cu_IFFT1D_R5_C2C_first(const float2* __restrict src, float2* __restrict dst, const size_t B_ops_num);


            __global__
            void cu_FFT1D_R5_C2C_first(const float2* __restrict src, float2* __restrict dst, const size_t B_ops_num);



            __global__
            void cu_IFFT1D_R5_C2R_once(const float2* __restrict src, float* __restrict dst, const size_t B_ops_num);



            __global__
            void cu_FFT1D_R5_C2C(const float2* src, float2* dst, const size_t B_ops_num, const uint warp_proc_len);



            __global__
            void cu_IFFT1D_R5_C2R_last(const float2* src, float* dst, const size_t B_ops_num, const uint warp_proc_len);


            __global__
            /*
            * @param B_ops_num : in Vec4
            * @param warp_proc_len : element
            */
            void cu_FFT1D_R5_C2C_vec4(const float4* src, float4* dst, const size_t B_ops_num, const uint warp_proc_len);



            __global__
            /*
            * @param B_ops_num : in Vec4
            * @param warp_proc_len : element
            */
            void cu_IFFT1D_R5_C2R_vec4_last(const float4* src, float4* dst, const size_t B_ops_num, const uint warp_proc_len);
        }
    }
}



namespace decx
{
    namespace signal
    {
        static void CUDA_FFT1D_R5_R2C_first_caller(const float* src, const de::CPf* dst, const size_t signal_len, decx::cuda_stream* S);

        template <bool is_IFFT>
        static void CUDA_FFT1D_R5_C2C_first_caller(const de::CPf* src, const de::CPf* dst, const size_t signal_len, decx::cuda_stream* S);


        static void CUDA_IFFT1D_R5_C2R_once_caller(const de::CPf* src, const de::CPf* dst, const size_t signal_len, decx::cuda_stream* S);


        static void CUDA_FFT1D_R5_C2C_caller(decx::alloc::MIF<de::CPf>* MIF_0, decx::alloc::MIF<de::CPf>* MIF_1, const size_t signal_len, 
            const uint warp_proc_len, decx::cuda_stream* S);


        static void CUDA_IFFT1D_R5_C2R_last_caller(decx::alloc::MIF<de::CPf>* MIF_0, decx::alloc::MIF<de::CPf>* MIF_1, const size_t signal_len,
            const uint warp_proc_len, decx::cuda_stream* S);
    }
}


static void decx::signal::CUDA_FFT1D_R5_R2C_first_caller(const float* src, const de::CPf* dst, const size_t signal_len, decx::cuda_stream* S)
{
    const size_t total_B_ops_num = signal_len / 5;
    decx::signal::GPUK::cu_FFT1D_R5_R2C_first << <decx::utils::ceil<size_t>(total_B_ops_num, _CUDA_FFT1D_BLOCK_SIZE),
                              _CUDA_FFT1D_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
            src, (float2*)dst, total_B_ops_num);
}


template <bool is_IFFT>
static void decx::signal::CUDA_FFT1D_R5_C2C_first_caller(const de::CPf* src, const de::CPf* dst, const size_t signal_len, decx::cuda_stream* S)
{
    const size_t total_B_ops_num = signal_len / 5;
    if (is_IFFT) {
        decx::signal::GPUK::cu_IFFT1D_R5_C2C_first << <decx::utils::ceil<size_t>(total_B_ops_num, _CUDA_FFT1D_BLOCK_SIZE),
            _CUDA_FFT1D_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                (float2*)src, (float2*)dst, total_B_ops_num);
    }
    else {
        decx::signal::GPUK::cu_FFT1D_R5_C2C_first << <decx::utils::ceil<size_t>(total_B_ops_num, _CUDA_FFT1D_BLOCK_SIZE),
            _CUDA_FFT1D_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (
                (float2*)src, (float2*)dst, total_B_ops_num);
    }
}



static void
decx::signal::CUDA_IFFT1D_R5_C2R_once_caller(const de::CPf* src, const de::CPf* dst, const size_t signal_len, decx::cuda_stream* S)
{
    const size_t total_B_ops_num = signal_len / 5;

    decx::signal::GPUK::cu_IFFT1D_R5_C2R_once << <decx::utils::ceil<size_t>(total_B_ops_num, _CUDA_FFT1D_BLOCK_SIZE),
        _CUDA_FFT1D_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > ((float2*)src, (float*)dst, total_B_ops_num);
}



static void 
decx::signal::CUDA_FFT1D_R5_C2C_caller(decx::alloc::MIF<de::CPf>* MIF_0, decx::alloc::MIF<de::CPf>* MIF_1, const size_t signal_len, 
    const uint warp_proc_len, decx::cuda_stream* S)
{
    if ((warp_proc_len / 5) % 4) {
        const size_t total_B_ops_num = signal_len / 5;
        float2* read_ptr = NULL, * write_ptr = NULL;
        if (MIF_0->leading) {
            read_ptr = reinterpret_cast<float2*>(MIF_0->mem);          write_ptr = reinterpret_cast<float2*>(MIF_1->mem);
            decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_1, MIF_0);
        }
        else {
            read_ptr = reinterpret_cast<float2*>(MIF_1->mem);          write_ptr = reinterpret_cast<float2*>(MIF_0->mem);
            decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_0, MIF_1);
        }
        decx::signal::GPUK::cu_FFT1D_R5_C2C << <decx::utils::ceil<size_t>(total_B_ops_num, _CUDA_FFT1D_BLOCK_SIZE), _CUDA_FFT1D_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> >
            (read_ptr, write_ptr, total_B_ops_num, warp_proc_len);
    }
    else {
        const size_t total_B_ops_num = signal_len / 5 / 4;
        float4* read_ptr = NULL, * write_ptr = NULL;
        if (MIF_0->leading) {
            read_ptr = reinterpret_cast<float4*>(MIF_0->mem);          write_ptr = reinterpret_cast<float4*>(MIF_1->mem);
            decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_1, MIF_0);
        }
        else {
            read_ptr = reinterpret_cast<float4*>(MIF_1->mem);          write_ptr = reinterpret_cast<float4*>(MIF_0->mem);
            decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_0, MIF_1);
        }
        decx::signal::GPUK::cu_FFT1D_R5_C2C_vec4 << <decx::utils::ceil<size_t>(total_B_ops_num, _CUDA_FFT1D_BLOCK_SIZE), _CUDA_FFT1D_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> >
            (read_ptr, write_ptr, total_B_ops_num, warp_proc_len);
    }
}



static void
decx::signal::CUDA_IFFT1D_R5_C2R_last_caller(decx::alloc::MIF<de::CPf>* MIF_0, decx::alloc::MIF<de::CPf>* MIF_1, const size_t signal_len,
    const uint warp_proc_len, decx::cuda_stream* S)
{
    const size_t total_B_ops_num = signal_len / 5;
    if (total_B_ops_num % 4) {
        float2* read_ptr = NULL;
        float* write_ptr = NULL;
        if (MIF_0->leading) {
            read_ptr = reinterpret_cast<float2*>(MIF_0->mem);
            write_ptr = reinterpret_cast<float*>(MIF_1->mem);
            decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_1, MIF_0);
        }
        else {
            read_ptr = reinterpret_cast<float2*>(MIF_1->mem);
            write_ptr = reinterpret_cast<float*>(MIF_0->mem);
            decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_0, MIF_1);
        }
        decx::signal::GPUK::cu_IFFT1D_R5_C2R_last << <decx::utils::ceil<size_t>(total_B_ops_num, _CUDA_FFT1D_BLOCK_SIZE),
            _CUDA_FFT1D_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr, total_B_ops_num, warp_proc_len);
    }
    else {
        float4* read_ptr = NULL, * write_ptr = NULL;
        if (MIF_0->leading) {
            read_ptr = reinterpret_cast<float4*>(MIF_0->mem);
            write_ptr = reinterpret_cast<float4*>(MIF_1->mem);
            decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_1, MIF_0);
        }
        else {
            read_ptr = reinterpret_cast<float4*>(MIF_1->mem);
            write_ptr = reinterpret_cast<float4*>(MIF_0->mem);
            decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_0, MIF_1);
        }
        decx::signal::GPUK::cu_IFFT1D_R5_C2R_vec4_last << <decx::utils::ceil<size_t>(total_B_ops_num / 4, _CUDA_FFT1D_BLOCK_SIZE),
            _CUDA_FFT1D_BLOCK_SIZE, 0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr, total_B_ops_num / 4, warp_proc_len);
    }
}



#endif