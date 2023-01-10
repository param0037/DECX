/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _GPU_FFT2D_UTILS_KERNEL_H_
#define _GPU_FFT2D_UTILS_KERNEL_H_


#include "../../../../basic_process/transpose/CUDA/transpose.cuh"
#include "../../../../classes/classes_util.h"


namespace decx
{
    namespace signal {
        namespace utils {
            namespace gpu 
            {
                static void _FFT2D_transpose_C2C_Async(decx::alloc::MIF<de::CPf>* MIF_0, decx::alloc::MIF<de::CPf>* MIF_1,
                    const uint pitchsrc, const uint pitchdst, const uint2 proc_dim_dst, decx::cuda_stream* S);


                static void _FFT2D_transpose_C2C_div_Async(decx::alloc::MIF<de::CPf>* MIF_0, decx::alloc::MIF<de::CPf>* MIF_1,
                    const uint pitchsrc, const uint pitchdst, const uint2 proc_dim_dst, const uint signal_len, decx::cuda_stream* S);


                static void _IFFT2D_transpose_R2R_Async(decx::alloc::MIF<de::CPf>* MIF_0, decx::alloc::MIF<de::CPf>* MIF_1,
                    const uint pitchsrc, const uint pitchdst, const uint2 proc_dim_dst, decx::cuda_stream* S);
            }
        }
    }
}



static void 
decx::signal::utils::gpu::_FFT2D_transpose_C2C_Async(decx::alloc::MIF<de::CPf>* MIF_0, decx::alloc::MIF<de::CPf>* MIF_1,
    const uint pitchsrc, const uint pitchdst, const uint2 proc_dim_dst, decx::cuda_stream* S)
{
    dim3 transp_thread_0(_CUDA_TRANSPOSE_4X4_D_BLOCK_SIZE_, _CUDA_TRANSPOSE_4X4_D_BLOCK_SIZE_);
    dim3 transp_grid_0(decx::utils::ceil<uint>(proc_dim_dst.y, _CUDA_TRANSPOSE_4X4_D_BLOCK_SIZE_ * 4),
                       decx::utils::ceil<uint>(proc_dim_dst.x, _CUDA_TRANSPOSE_4X4_D_BLOCK_SIZE_ * 4));

    double2* read_ptr = NULL, * write_ptr = NULL;

    if (MIF_0->leading) {
        read_ptr = reinterpret_cast<double2*>(MIF_0->mem);
        write_ptr = reinterpret_cast<double2*>(MIF_1->mem);
        decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_1, MIF_0);
    }
    else {
        read_ptr = reinterpret_cast<double2*>(MIF_1->mem);
        write_ptr = reinterpret_cast<double2*>(MIF_0->mem);
        decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_0, MIF_1);
    }
    cu_transpose_vec4x4d << <transp_grid_0, transp_thread_0, 0, S->get_raw_stream_ref() >> > (
        read_ptr, write_ptr, pitchsrc, pitchdst, proc_dim_dst);
}



static void 
decx::signal::utils::gpu::_FFT2D_transpose_C2C_div_Async(decx::alloc::MIF<de::CPf>* MIF_0, decx::alloc::MIF<de::CPf>* MIF_1,
    const uint pitchsrc, const uint pitchdst, const uint2 proc_dim_dst, const uint signal_len, decx::cuda_stream* S)
{
    dim3 transp_thread_0(_CUDA_TRANSPOSE_4X4_D_BLOCK_SIZE_, _CUDA_TRANSPOSE_4X4_D_BLOCK_SIZE_);
    dim3 transp_grid_0(decx::utils::ceil<uint>(proc_dim_dst.y, _CUDA_TRANSPOSE_4X4_D_BLOCK_SIZE_ * 4),
                       decx::utils::ceil<uint>(proc_dim_dst.x, _CUDA_TRANSPOSE_4X4_D_BLOCK_SIZE_ * 4));

    double2* read_ptr = NULL, * write_ptr = NULL;

    if (MIF_0->leading) {
        read_ptr = reinterpret_cast<double2*>(MIF_0->mem);
        write_ptr = reinterpret_cast<double2*>(MIF_1->mem);
        decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_1, MIF_0);
    }
    else {
        read_ptr = reinterpret_cast<double2*>(MIF_1->mem);
        write_ptr = reinterpret_cast<double2*>(MIF_0->mem);
        decx::utils::set_mutex_memory_state<de::CPf, de::CPf>(MIF_0, MIF_1);
    }
    cu_transpose_vec4x4d_and_divide << <transp_grid_0, transp_thread_0, 0, S->get_raw_stream_ref() >> > (
        read_ptr, write_ptr, pitchsrc, pitchdst, (float)signal_len, proc_dim_dst);
}

// cu_transpose_vec4x4f


static void
decx::signal::utils::gpu::_IFFT2D_transpose_R2R_Async(decx::alloc::MIF<de::CPf>* MIF_0, decx::alloc::MIF<de::CPf>* MIF_1,
    const uint pitchsrc, const uint pitchdst, const uint2 proc_dim_dst, decx::cuda_stream* S)
{
    dim3 transp_thread_0(_CUDA_TRANSPOSE_4X4_D_BLOCK_SIZE_, _CUDA_TRANSPOSE_4X4_D_BLOCK_SIZE_);
    dim3 transp_grid_0(decx::utils::ceil<uint>(proc_dim_dst.y, _CUDA_TRANSPOSE_4X4_D_BLOCK_SIZE_ * 4),
        decx::utils::ceil<uint>(proc_dim_dst.x, _CUDA_TRANSPOSE_4X4_D_BLOCK_SIZE_ * 4));

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
    cu_transpose_vec4x4f << <transp_grid_0, transp_thread_0, 0, S->get_raw_stream_ref() >> > (
        read_ptr, write_ptr, pitchsrc, pitchdst, proc_dim_dst);
}


#endif