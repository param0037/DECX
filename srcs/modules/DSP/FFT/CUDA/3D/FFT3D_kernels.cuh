/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _FFT3D_KERNELS_CUH_
#define _FFT3D_KERNELS_CUH_


#include "../../../../core/utils/decx_cuda_vectypes_ops.cuh"
#include "../../../../core/utils/decx_cuda_math_functions.cuh"
#include "../../../CUDA_cpf32.cuh"
#include "../../FFT_commons.h"
#include "../2D/FFT2D_config.cuh"


namespace decx
{
namespace dsp {
namespace fft 
{
    namespace GPUK 
    {
        // ------------------------------------------------ Radix-2 ------------------------------------------------
        
        template <bool _div> __global__ void
        cu_FFT3_R2_1st_C2C_cplxf(const float4* __restrict src, float4* __restrict dst, const uint32_t _signal_len,
            const uint2 _signal_pitch, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2, const uint32_t _paral);


        __global__ void
        cu_FFT3_R2_C2C_cplxf(const float4* __restrict, float4* __restrict dst, const decx::dsp::fft::FKI_4_2DK _kernel_info,
            const uint32_t signal_pitch, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2, const uint32_t _paral);


        // ------------------------------------------------ Radix-3 ------------------------------------------------
        
        template <bool _div> __global__ void
        cu_FFT3_R3_1st_C2C_cplxf(const float4* __restrict src, float4* __restrict dst, const uint32_t _signal_len,
            const uint2 _signal_pitch, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2, const uint32_t _paral);


        __global__ void
        cu_FFT3_R3_C2C_cplxf(const float4* __restrict, float4* __restrict dst, const decx::dsp::fft::FKI_4_2DK _kernel_info,
            const uint32_t signal_pitch, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2, const uint32_t _paral);

        // ------------------------------------------------ Radix-4 ------------------------------------------------

        template <bool _div> __global__ void 
        cu_FFT3_R4_1st_C2C_cplxf(const float4* __restrict src, float4* __restrict dst, const uint32_t _signal_len, 
            const uint2 _signal_pitch, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2, 
            const uint32_t _paral, const uint64_t _div_length = 0);


        __global__ void
        cu_FFT3_R4_C2C_cplxf(const float4* __restrict, float4* __restrict dst, const decx::dsp::fft::FKI_4_2DK _kernel_info, 
            const uint32_t signal_pitch, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2, const uint32_t _paral);


        // ------------------------------------------------ Radix-5 ------------------------------------------------
        
        template <bool _div> __global__ void 
        cu_FFT3_R5_1st_C2C_cplxf(const float4* __restrict src, float4* __restrict dst, const uint32_t _signal_len, 
            const uint2 _signal_pitch, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2, const uint32_t _paral);


        __global__ void
        cu_FFT3_R5_C2C_cplxf(const float4* __restrict, float4* __restrict dst, const decx::dsp::fft::FKI_4_2DK _kernel_info, 
            const uint32_t signal_pitch, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2, const uint32_t _paral);
    }

    template <bool _div>
    static void FFT3D_1st_C2C_caller_cplxf(const float4* __restrict src, float4* __restrict dst,
        const uint8_t _radix, const uint32_t _signal_len, const uint2 _signal_pitch, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2,
        const uint32_t _paral, decx::cuda_stream* S);


    static void FFT3D_C2C_caller_cplxf(const float4* __restrict src, float4* __restrict dst, const uint8_t _radix, 
        const decx::dsp::fft::FKI_4_2DK* _kernel_info, const uint32_t _signal_pitch, const uint32_t _pitchsrc_v2, const uint32_t _putchdst_v2, 
        const uint32_t _paral, decx::cuda_stream* S);
}
}
}



template <bool _div> static void
decx::dsp::fft::FFT3D_1st_C2C_caller_cplxf(const float4* __restrict     src, 
                                           float4* __restrict           dst,
                                           const uint8_t                _radix, 
                                           const uint32_t               _signal_len,
                                           const uint2                  _signal_pitch,
                                           const uint32_t               _pitchsrc_v2, 
                                           const uint32_t               _pitchdst_v2,
                                           const uint32_t               _paral,
                                           decx::cuda_stream*           S)
{
    dim3 _grid(decx::utils::ceil<uint32_t>(min(_pitchsrc_v2, _pitchdst_v2), _FFT2D_BLOCK_X_),
               decx::utils::ceil<uint32_t>((_signal_len / _radix) * _paral, _FFT2D_BLOCK_Y_));

    dim3 _block(_FFT2D_BLOCK_X_, _FFT2D_BLOCK_Y_);

    switch (_radix)
    {
    case 2:
        decx::dsp::fft::GPUK::cu_FFT3_R2_1st_C2C_cplxf<_div> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _signal_pitch, _pitchsrc_v2, _pitchdst_v2, _paral);
        break;

    case 3:
        decx::dsp::fft::GPUK::cu_FFT3_R3_1st_C2C_cplxf<_div> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _signal_pitch, _pitchsrc_v2, _pitchdst_v2, _paral);
        break;

    case 4:
        decx::dsp::fft::GPUK::cu_FFT3_R4_1st_C2C_cplxf<_div> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _signal_pitch, _pitchsrc_v2, _pitchdst_v2, _paral);
        break;

    case 5:
        decx::dsp::fft::GPUK::cu_FFT3_R5_1st_C2C_cplxf<_div> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _signal_pitch, _pitchsrc_v2, _pitchdst_v2, _paral);
        break;
    default:
        break;
    }
}


static void 
decx::dsp::fft::FFT3D_C2C_caller_cplxf(const float4* __restrict src,
                                   float4* __restrict dst, 
                                   const uint8_t _radix,
                                   const decx::dsp::fft::FKI_4_2DK* _kernel_info, 
                                   const uint32_t _signal_pitch, 
                                   const uint32_t _pitchsrc_v2, 
                                   const uint32_t _pitchdst_v2,
                                   const uint32_t _paral, 
                                   decx::cuda_stream* S)
{
    dim3 _grid(decx::utils::ceil<uint32_t>(min(_pitchsrc_v2, _pitchdst_v2), _FFT2D_BLOCK_X_),
        decx::utils::ceil<uint32_t>((_kernel_info->_signal_len / _radix) * _paral, _FFT2D_BLOCK_Y_));
    dim3 _block(_FFT2D_BLOCK_X_, _FFT2D_BLOCK_Y_);

    switch (_radix)
    {
    case 2:
        decx::dsp::fft::GPUK::cu_FFT3_R2_C2C_cplxf << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _signal_pitch, _pitchsrc_v2, _pitchdst_v2, _paral);
        break;

    case 3:
        decx::dsp::fft::GPUK::cu_FFT3_R3_C2C_cplxf << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _signal_pitch, _pitchsrc_v2, _pitchdst_v2, _paral);
        break;

    case 4:
        decx::dsp::fft::GPUK::cu_FFT3_R4_C2C_cplxf << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _signal_pitch, _pitchsrc_v2, _pitchdst_v2, _paral);
        break;

    case 5:
        decx::dsp::fft::GPUK::cu_FFT3_R5_C2C_cplxf << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _signal_pitch, _pitchsrc_v2, _pitchdst_v2, _paral);
        break;
    default:
        break;
    }
}


#endif