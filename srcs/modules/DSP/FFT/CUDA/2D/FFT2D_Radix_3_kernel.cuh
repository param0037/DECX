/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _GPU_FFT2D_RADIX_3_KERNEL_H_
#define _GPU_FFT2D_RADIX_3_KERNEL_H_


#include "../../../../core/basic.h"
#include "../../../../classes/classes_util.h"
#include "../../fft_utils.h"


namespace decx
{
    namespace signal {
        namespace GPUK {
            __global__
            void cu_FFT2D_R3_R2C_first(const float* src, float2* dst,
                const uint B_ops_num, const uint pitchsrc, const uint pitchdst, const uint procH);



            __global__
            void cu_FFT2D_R3_C2C_first(const float2* src, float2* dst,
                const uint B_ops_num, const uint pitch, const uint procH);


            __global__
            void cu_IFFT2D_R3_C2R_once(const float2* src, float* dst,
                const uint B_ops_num, const uint pitch, const uint procH);


            __global__
            void cu_IFFT2D_R3_C2C_first(const float2* src, float2* dst,
                const uint B_ops_num, const uint pitch, const uint procH);



            __global__
            void cu_FFT2D_R3_C2C(const float2* src, float2* dst,
                const uint B_ops_num, const uint warp_proc_len,
                const uint pitch, const uint procH);


            __global__
            void cu_IFFT2D_R3_C2R_last(const float2* src, float* dst,
                const uint B_ops_num, const uint warp_proc_len,
                const uint pitch, const uint procH);


            __global__
            /*
            * @param B_ops_num : in Vec4
            * @param warp_proc_len : element
            * @param pitch : in float4
            */
            void cu_FFT2D_R3_C2C_vec4(const float4* src, float4* dst,
                const uint B_ops_num, const uint warp_proc_len,
                const uint pitch, const uint procH);



            __global__
            /*
            * @param B_ops_num : in Vec4
            * @param warp_proc_len : element
            * @param pitch : in float4
            */
            void cu_IFFT2D_R3_C2R_vec4_last(const float4* src, float4* dst,
                const uint B_ops_num, const uint warp_proc_len,
                const uint pitch, const uint procH);
        }
    }
}



namespace decx
{
    namespace signal
    {
        namespace gpu {
            static void CUDA_FFT2D_R3_R2C_first_caller(decx::alloc::MIF<de::CPf>* MIF_0, decx::alloc::MIF<de::CPf>* MIF_1, 
                const uint signal_len, const uint pitchsrc, const uint pitchdst, const uint procH, decx::cuda_stream* S);


            static void dev_CUDA_FFT2D_R3_R2C_first_caller(float* src, decx::alloc::MIF<de::CPf>* MIF_0,
                const uint signal_len, const uint pitchsrc, const uint pitchdst, const uint procH, decx::cuda_stream* S);


            template <bool is_IFFT>
            static void CUDA_FFT2D_R3_C2C_first_caller(decx::alloc::MIF<de::CPf>* MIF_0, decx::alloc::MIF<de::CPf>* MIF_1, 
                const uint signal_len, const uint pitch, const uint procH, decx::cuda_stream* S);


            template <bool is_IFFT>
            static void dev_CUDA_FFT2D_R3_C2C_first_caller(de::CPf* src, decx::alloc::MIF<de::CPf>* MIF_0,
                const uint signal_len, const uint pitch, const uint procH, decx::cuda_stream* S);


            static void CUDA_IFFT2D_R3_C2R_once_caller(decx::alloc::MIF<de::CPf>* MIF_0, decx::alloc::MIF<de::CPf>* MIF_1,
                const uint signal_len, const uint pitch, const uint procH, decx::cuda_stream* S);


            static void CUDA_FFT2D_R3_C2C_caller(decx::alloc::MIF<de::CPf>* MIF_0, decx::alloc::MIF<de::CPf>* MIF_1, 
                const uint signal_len, const uint pitch, const uint procH, const uint warp_proc_len, decx::cuda_stream* S);


            static void CUDA_IFFT2D_R3_C2R_last_caller(decx::alloc::MIF<de::CPf>* MIF_0, decx::alloc::MIF<de::CPf>* MIF_1,
                const uint signal_len, const uint pitch, const uint procH, const uint warp_proc_len, decx::cuda_stream* S);
        }
    }
}



static void 
decx::signal::gpu::CUDA_FFT2D_R3_R2C_first_caller(decx::alloc::MIF<de::CPf>* MIF_0, 
                                                  decx::alloc::MIF<de::CPf>* MIF_1, 
                                                  const uint signal_len,
                                                  const uint pitchsrc,       // in float
                                                  const uint pitchdst,       // in de::CPf
                                                  const uint procH, 
                                                  decx::cuda_stream* S)
{
    const uint total_B_ops_num = signal_len / 3;

    dim3 thread(FFT2D_BLOCK_SIZE, FFT2D_BLOCK_SIZE);
    
    float* read_ptr = NULL;
    float2* write_ptr = NULL;
    if (MIF_0->leading) {
        read_ptr = reinterpret_cast<float*>(MIF_0->mem);        write_ptr = reinterpret_cast<float2*>(MIF_1->mem);
        decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_1, MIF_0);
    }
    else {
        read_ptr = reinterpret_cast<float*>(MIF_1->mem);        write_ptr = reinterpret_cast<float2*>(MIF_0->mem);
        decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_0, MIF_1);
    }
    dim3 grid(decx::utils::ceil<uint>(procH, FFT2D_BLOCK_SIZE), 
            decx::utils::ceil<uint>(total_B_ops_num, FFT2D_BLOCK_SIZE));
    decx::signal::GPUK::cu_FFT2D_R3_R2C_first << <grid, thread, 0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr,
        total_B_ops_num, pitchsrc, pitchdst, procH);
}




static void 
decx::signal::gpu::dev_CUDA_FFT2D_R3_R2C_first_caller(float* src, 
                                                  decx::alloc::MIF<de::CPf>* MIF_0, 
                                                  const uint signal_len,
                                                  const uint pitchsrc,       // in float
                                                  const uint pitchdst,       // in de::CPf
                                                  const uint procH, 
                                                  decx::cuda_stream* S)
{
    const uint total_B_ops_num = signal_len / 3;

    dim3 thread(FFT2D_BLOCK_SIZE, FFT2D_BLOCK_SIZE);
    
    MIF_0->leading = true;
    
    dim3 grid(decx::utils::ceil<uint>(procH, FFT2D_BLOCK_SIZE), 
            decx::utils::ceil<uint>(total_B_ops_num, FFT2D_BLOCK_SIZE));
    decx::signal::GPUK::cu_FFT2D_R3_R2C_first << <grid, thread, 0, S->get_raw_stream_ref() >> > (src, reinterpret_cast<float2*>(MIF_0->mem),
        total_B_ops_num, pitchsrc, pitchdst, procH);
}




template <bool is_IFFT> static void 
decx::signal::gpu::CUDA_FFT2D_R3_C2C_first_caller(decx::alloc::MIF<de::CPf>* MIF_0,
                                                  decx::alloc::MIF<de::CPf>* MIF_1,
                                                  const uint signal_len,
                                                  const uint pitch,         // in de::CPf
                                                  const uint procH,         // in de::CPf
                                                  decx::cuda_stream* S)
{
    const size_t total_B_ops_num = signal_len / 3;
    dim3 thread(FFT2D_BLOCK_SIZE, FFT2D_BLOCK_SIZE);

    float2* read_ptr = NULL, * write_ptr = NULL;
    dim3 grid(decx::utils::ceil<uint>(procH, FFT2D_BLOCK_SIZE),
                decx::utils::ceil<uint>(total_B_ops_num, FFT2D_BLOCK_SIZE));
    if (MIF_0->leading) {
        read_ptr = reinterpret_cast<float2*>(MIF_0->mem);       write_ptr = reinterpret_cast<float2*>(MIF_1->mem);
        decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_1, MIF_0);
    }
    else{
        read_ptr = reinterpret_cast<float2*>(MIF_1->mem);       write_ptr = reinterpret_cast<float2*>(MIF_0->mem);
        decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_0, MIF_1);
    }

    if (is_IFFT) {
        decx::signal::GPUK::cu_IFFT2D_R3_C2C_first << <grid, thread, 0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr,
            total_B_ops_num, pitch, procH);
    }
    else {
        decx::signal::GPUK::cu_FFT2D_R3_C2C_first << <grid, thread, 0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr,
            total_B_ops_num, pitch, procH);
    }
}




template <bool is_IFFT> static void 
decx::signal::gpu::dev_CUDA_FFT2D_R3_C2C_first_caller(de::CPf* src,
                                                  decx::alloc::MIF<de::CPf>* MIF_0,
                                                  const uint signal_len,
                                                  const uint pitch,         // in de::CPf
                                                  const uint procH,         // in de::CPf
                                                  decx::cuda_stream* S)
{
    const size_t total_B_ops_num = signal_len / 3;
    dim3 thread(FFT2D_BLOCK_SIZE, FFT2D_BLOCK_SIZE);

    float2* read_ptr = NULL, * write_ptr = NULL;
    dim3 grid(decx::utils::ceil<uint>(procH, FFT2D_BLOCK_SIZE),
                decx::utils::ceil<uint>(total_B_ops_num, FFT2D_BLOCK_SIZE));
    
    MIF_0->leading = true;

    if (is_IFFT) {
        decx::signal::GPUK::cu_IFFT2D_R3_C2C_first << <grid, thread, 0, S->get_raw_stream_ref() >> > (
            reinterpret_cast<float2*>(src), reinterpret_cast<float2*>(MIF_0->mem),
            total_B_ops_num, pitch, procH);
    }
    else {
        decx::signal::GPUK::cu_FFT2D_R3_C2C_first << <grid, thread, 0, S->get_raw_stream_ref() >> > (
            reinterpret_cast<float2*>(src), reinterpret_cast<float2*>(MIF_0->mem),
            total_B_ops_num, pitch, procH);
    }
}




static void 
decx::signal::gpu::CUDA_IFFT2D_R3_C2R_once_caller(decx::alloc::MIF<de::CPf>*MIF_0, 
                                                  decx::alloc::MIF<de::CPf>*MIF_1,
                                                  const uint signal_len,
                                                  const uint pitch,            // in de::CPf
                                                  const uint procH,            // in de::CPf
                                                  decx::cuda_stream* S)
{
    const size_t total_B_ops_num = signal_len / 3;
    dim3 thread(FFT2D_BLOCK_SIZE, FFT2D_BLOCK_SIZE);
    
    dim3 grid(decx::utils::ceil<uint>(procH, FFT2D_BLOCK_SIZE),
        decx::utils::ceil<uint>(total_B_ops_num, FFT2D_BLOCK_SIZE));

    float2* read_ptr = NULL;
    float* write_ptr = NULL;
    if (MIF_0->leading) {
        read_ptr = reinterpret_cast<float2*>(MIF_0->mem);          write_ptr = reinterpret_cast<float*>(MIF_1->mem);
        decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_1, MIF_0);
    }
    else {
        read_ptr = reinterpret_cast<float2*>(MIF_1->mem);          write_ptr = reinterpret_cast<float*>(MIF_0->mem);
        decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_0, MIF_1);
    }

    decx::signal::GPUK::cu_IFFT2D_R3_C2R_once << <grid, thread, 0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr,
        total_B_ops_num, pitch, procH);
}



static void 
decx::signal::gpu::CUDA_FFT2D_R3_C2C_caller(decx::alloc::MIF<de::CPf>*MIF_0, 
                                            decx::alloc::MIF<de::CPf>*MIF_1,
                                            const uint signal_len,
                                            const uint pitch,            // in de::CPf
                                            const uint procH,            // in de::CPf
                                            const uint warp_proc_len, 
                                            decx::cuda_stream* S)
{
    const size_t total_B_ops_num = signal_len / 3;
    dim3 thread(FFT2D_BLOCK_SIZE, FFT2D_BLOCK_SIZE);
    
    if (total_B_ops_num % 4) {
        dim3 grid(decx::utils::ceil<uint>(procH, FFT2D_BLOCK_SIZE), 
              decx::utils::ceil<uint>(total_B_ops_num, FFT2D_BLOCK_SIZE));

        float2* read_ptr = NULL, *write_ptr = NULL;
        if (MIF_0->leading) {
            read_ptr = reinterpret_cast<float2*>(MIF_0->mem);          write_ptr = reinterpret_cast<float2*>(MIF_1->mem);
            decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_1, MIF_0);
        }
        else {
            read_ptr = reinterpret_cast<float2*>(MIF_1->mem);          write_ptr = reinterpret_cast<float2*>(MIF_0->mem);
            decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_0, MIF_1);
        }

        decx::signal::GPUK::cu_FFT2D_R3_C2C << <grid, thread, 0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr,
            total_B_ops_num, warp_proc_len, pitch, procH);
    }
    else {
        dim3 grid(decx::utils::ceil<uint>(procH, FFT2D_BLOCK_SIZE),
            decx::utils::ceil<uint>(total_B_ops_num / 4, FFT2D_BLOCK_SIZE));

        float4* read_ptr = NULL, * write_ptr = NULL;
        if (MIF_0->leading) {
            read_ptr = reinterpret_cast<float4*>(MIF_0->mem);          write_ptr = reinterpret_cast<float4*>(MIF_1->mem);
            decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_1, MIF_0);
        }
        else {
            read_ptr = reinterpret_cast<float4*>(MIF_1->mem);          write_ptr = reinterpret_cast<float4*>(MIF_0->mem);
            decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_0, MIF_1);
        }

        decx::signal::GPUK::cu_FFT2D_R3_C2C_vec4 << < grid, thread, 0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr,
            total_B_ops_num / 4, warp_proc_len, pitch / 2, procH);
    }
}


static void 
decx::signal::gpu::CUDA_IFFT2D_R3_C2R_last_caller(decx::alloc::MIF<de::CPf>*MIF_0, 
                                                  decx::alloc::MIF<de::CPf>*MIF_1,
                                                  const uint signal_len,
                                                  const uint pitch,            // in de::CPf
                                                  const uint procH,            // in de::CPf
                                                  const uint warp_proc_len, 
                                                  decx::cuda_stream* S)
{
    const size_t total_B_ops_num = signal_len / 3;
    dim3 thread(FFT2D_BLOCK_SIZE, FFT2D_BLOCK_SIZE);
    
    if (total_B_ops_num % 4) {
        dim3 grid(decx::utils::ceil<uint>(procH, FFT2D_BLOCK_SIZE), 
              decx::utils::ceil<uint>(total_B_ops_num, FFT2D_BLOCK_SIZE));

        float2* read_ptr = NULL;
        float* write_ptr = NULL;
        if (MIF_0->leading) {
            read_ptr = reinterpret_cast<float2*>(MIF_0->mem);          write_ptr = reinterpret_cast<float*>(MIF_1->mem);
            decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_1, MIF_0);
        }
        else {
            read_ptr = reinterpret_cast<float2*>(MIF_1->mem);          write_ptr = reinterpret_cast<float*>(MIF_0->mem);
            decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_0, MIF_1);
        }

        decx::signal::GPUK::cu_IFFT2D_R3_C2R_last << <grid, thread, 0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr,
            total_B_ops_num, warp_proc_len, pitch, procH);
    }
    else {
        dim3 grid(decx::utils::ceil<uint>(procH, FFT2D_BLOCK_SIZE),
            decx::utils::ceil<uint>(total_B_ops_num / 4, FFT2D_BLOCK_SIZE));

        float4* read_ptr = NULL, * write_ptr = NULL;
        if (MIF_0->leading) {
            read_ptr = reinterpret_cast<float4*>(MIF_0->mem);          write_ptr = reinterpret_cast<float4*>(MIF_1->mem);
            decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_1, MIF_0);
        }
        else {
            read_ptr = reinterpret_cast<float4*>(MIF_1->mem);          write_ptr = reinterpret_cast<float4*>(MIF_0->mem);
            decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_0, MIF_1);
        }

        decx::signal::GPUK::cu_IFFT2D_R3_C2R_vec4_last << < grid, thread, 0, S->get_raw_stream_ref() >> > (read_ptr, write_ptr,
            total_B_ops_num / 4, warp_proc_len, pitch / 2, procH);
    }
}


#endif