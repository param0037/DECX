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


#ifndef _FFT2D_KERNELS_CUH_
#define _FFT2D_KERNELS_CUH_

#include "../../../../core/utils/decx_cuda_vectypes_ops.cuh"
#include "../../../../core/utils/decx_cuda_math_functions.cuh"
#include "../../../CUDA_cpf32.cuh"
#include "../../FFT_commons.h"
#include "FFT2D_config.cuh"


namespace decx
{
namespace dsp {
namespace fft 
{
    namespace GPUK 
    {
        // ------------------------------------------------ Radix-2 ------------------------------------------------
        __global__ void cu_FFT2_R2_1st_R2C_cplxf(const float2* __restrict src, float4* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc_v2, const uint32_t _putchdst_v2);


        __global__ void cu_FFT2_R2_1st_R2C_uc8_cplxf(const ushort* __restrict src, float4* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc_v2, const uint32_t _putchdst_v2);


        __global__ void cu_FFT2_R2_1st_R2C_uc8_cplxd(const uint8_t* __restrict src, double2* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc_v1, const uint32_t _putchdst_v1);


        template <bool _div> __global__ void
            cu_FFT2_R2_1st_C2C_cplxf(const float4* __restrict src, float4* __restrict dst,
                const uint32_t _signal_len, const uint32_t _pitchsrc_v2, const uint32_t _putchdst_v2, const uint64_t _div_length = 0);


        template <bool _conj> __global__ void
            cu_FFT2_R2_C2C_cplxf(const float4* __restrict, float4* __restrict dst,
                const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2);


        __global__ void
        cu_FFT2_R2_C2R_cplxf_u8(const float4* __restrict, uchar2* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2);


        __global__ void
        cu_FFT2_R2_C2R_cplxd_u8(const double2* __restrict, uint8_t* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v1, const uint32_t _pitchdst_v1);


        __global__ void
        cu_FFT2_R2_C2R_cplxf_fp32(const float4* __restrict, float2* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2);


        // ------------------------------------------------ Radix-3 ------------------------------------------------
        __global__ void cu_FFT2_R3_1st_R2C_cplxf(const float2* __restrict src, float4* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc_v2, const uint32_t _putchdst_v2);


        __global__ void cu_FFT2_R3_1st_R2C_uc8_cplxf(const ushort* __restrict src, float4* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc_v2, const uint32_t _putchdst_v2);


        __global__ void cu_FFT2_R3_1st_R2C_uc8_cplxd(const uint8_t* __restrict src, double2* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc_v1, const uint32_t _putchdst_v1);


        template <bool _div> __global__ void
            cu_FFT2_R3_1st_C2C_cplxf(const float4* __restrict src, float4* __restrict dst,
                const uint32_t _signal_len, const uint32_t _pitchsrc_v2, const uint32_t _putchdst_v2, const uint64_t _div_length = 0);


        template <bool _conj> __global__ void
            cu_FFT2_R3_C2C_cplxf(const float4* __restrict, float4* __restrict dst,
                const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2);


        __global__ void
        cu_FFT2_R3_C2R_cplxf_u8(const float4* __restrict, uchar2* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2);


        __global__ void
        cu_FFT2_R3_C2R_cplxd_u8(const double2* __restrict, uint8_t* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v1, const uint32_t _pitchdst_v1);



        __global__ void
        cu_FFT2_R3_C2R_cplxf_fp32(const float4* __restrict, float2* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2);

        // ------------------------------------------------ Radix-4 ------------------------------------------------
        __global__ void cu_FFT2_R4_1st_R2C_cplxf(const float2* __restrict src, float4* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc_v2, const uint32_t _putchdst_v2);


        __global__ void cu_FFT2_R4_1st_R2C_uc8_cplxf(const ushort* __restrict src, float4* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc_v2, const uint32_t _putchdst_v2);


        __global__ void cu_FFT2_R4_1st_R2C_uc8_cplxd(const uint8_t* __restrict src, double2* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc_v1, const uint32_t _putchdst_v1);


        template <bool _div> __global__ void 
        cu_FFT2_R4_1st_C2C_cplxf(const float4* __restrict src, float4* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc_v2, const uint32_t _putchdst_v2, const uint64_t _div_length = 0);


        template <bool _conj> __global__ void
        cu_FFT2_R4_C2C_cplxf(const float4* __restrict, float4* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2);


        __global__ void
        cu_FFT2_R4_C2R_cplxf_u8(const float4* __restrict, uchar2* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2);


        __global__ void
        cu_FFT2_R4_C2R_cplxd_u8(const double2* __restrict, uint8_t* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2);



        __global__ void
        cu_FFT2_R4_C2R_cplxf_fp32(const float4* __restrict, float2* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2);


        // ------------------------------------------------ Radix-5 ------------------------------------------------
        __global__ void cu_FFT2_R5_1st_R2C_cplxf(const float2* __restrict src, float4* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2);


        __global__ void cu_FFT2_R5_1st_R2C_uc8_cplxf(const ushort* __restrict src, float4* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2);

        __global__ void cu_FFT2_R5_1st_R2C_uc8_cplxd(const uint8_t* __restrict src, double2* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc_v1, const uint32_t _pitchdst_v1);

        template <bool _div> __global__ void 
        cu_FFT2_R5_1st_C2C_cplxf(const float4* __restrict src, float4* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc_v2, const uint32_t _putchdst_v2, const uint64_t _div_length = 0);


        template <bool _conj> __global__ void
        cu_FFT2_R5_C2C_cplxf(const float4* __restrict, float4* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2);


        __global__ void
        cu_FFT2_R5_C2R_cplxf_u8(const float4* __restrict, uchar2* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2);

        __global__ void
        cu_FFT2_R5_C2R_cplxd_u8(const double2* __restrict, uint8_t* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v1, const uint32_t _pitchdst_v1);

        __global__ void
        cu_FFT2_R5_C2R_cplxf_fp32(const float4* __restrict, float2* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2);
    }
    
    static void FFT2D_1st_R2C_caller_cplxf(const float2* __restrict src, float4* __restrict dst,
        const uint8_t _radix, const uint32_t _signal_len, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2,
        decx::cuda_stream* S);


    static void FFT2D_1st_R2C_caller_uc8_cplxf(const ushort* __restrict src, float4* __restrict dst,
        const uint8_t _radix, const uint32_t _signal_len, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2,
        decx::cuda_stream* S);


    static void FFT2D_1st_R2C_caller_uc8_cplxd(const uint8_t* __restrict src, double2* __restrict dst,
        const uint8_t _radix, const uint32_t _signal_len, const uint32_t _pitchsrc_v1, const uint32_t _pitchdst_v1,
        decx::cuda_stream* S);


    template <bool _div>
    static void FFT2D_1st_C2C_caller_cplxf(const float4* __restrict src, float4* __restrict dst,
        const uint8_t _radix, const uint32_t _signal_len, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2,
        decx::cuda_stream* S, const uint64_t _div_length = 0);


    template <bool _conj>
    static void FFT2D_C2C_caller_cplxf(const float4* __restrict src, float4* __restrict dst,
        const uint8_t _radix, const decx::dsp::fft::FKI_4_2DK* _kernel_info, const uint32_t _pitchsrc_v2, const uint32_t _putchdst_v2,
        decx::cuda_stream* S);


    static void IFFT2D_C2R_caller_cplxf_u8(const float4* __restrict src, uchar2* __restrict dst,
        const uint8_t _radix, const decx::dsp::fft::FKI_4_2DK* _kernel_info, const uint32_t _pitchsrc_v2, const uint32_t _putchdst_v2,
        decx::cuda_stream* S);


    static void IFFT2D_C2R_caller_cplxd_u8(const double2* __restrict src, uint8_t* __restrict dst,
        const uint8_t _radix, const decx::dsp::fft::FKI_4_2DK* _kernel_info, const uint32_t _pitchsrc_v1, const uint32_t _putchdst_v1,
        decx::cuda_stream* S);


    static void IFFT2D_C2R_caller_cplxf_fp32(const float4* __restrict src, float2* __restrict dst,
        const uint8_t _radix, const decx::dsp::fft::FKI_4_2DK* _kernel_info, const uint32_t _pitchsrc_v2, const uint32_t _putchdst_v2,
        decx::cuda_stream* S);
}
}
}


static void 
decx::dsp::fft::FFT2D_1st_R2C_caller_cplxf(const float2* __restrict src, 
                                           float4* __restrict dst,
                                           const uint8_t _radix, 
                                           const uint32_t _signal_len, 
                                           const uint32_t _pitchsrc_v2, 
                                           const uint32_t _pitchdst_v2,
                                           decx::cuda_stream* S)
{
    dim3 _grid(decx::utils::ceil<uint32_t>(min(_pitchsrc_v2, _pitchdst_v2), _FFT2D_BLOCK_X_),
               decx::utils::ceil<uint32_t>(_signal_len / _radix, _FFT2D_BLOCK_Y_));
    dim3 _block(_FFT2D_BLOCK_X_, _FFT2D_BLOCK_Y_);

    switch (_radix)
    {
    case 2:
        decx::dsp::fft::GPUK::cu_FFT2_R2_1st_R2C_cplxf << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc_v2, _pitchdst_v2);
        break;

    case 3:
        decx::dsp::fft::GPUK::cu_FFT2_R3_1st_R2C_cplxf << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc_v2, _pitchdst_v2);
        break;

    case 4:
        decx::dsp::fft::GPUK::cu_FFT2_R4_1st_R2C_cplxf << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc_v2, _pitchdst_v2);
        break;

    case 5:
        decx::dsp::fft::GPUK::cu_FFT2_R5_1st_R2C_cplxf << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc_v2, _pitchdst_v2);
        break;
    default:
        break;
    }
}



static void 
decx::dsp::fft::FFT2D_1st_R2C_caller_uc8_cplxf(const ushort* __restrict src, 
                                           float4* __restrict dst,
                                           const uint8_t _radix, 
                                           const uint32_t _signal_len, 
                                           const uint32_t _pitchsrc_v2, 
                                           const uint32_t _pitchdst_v2,
                                           decx::cuda_stream* S)
{
    dim3 _grid(decx::utils::ceil<uint32_t>(min(_pitchsrc_v2, _pitchdst_v2), _FFT2D_BLOCK_X_),
               decx::utils::ceil<uint32_t>(_signal_len / _radix, _FFT2D_BLOCK_Y_));
    dim3 _block(_FFT2D_BLOCK_X_, _FFT2D_BLOCK_Y_);

    switch (_radix)
    {
    case 2:
        decx::dsp::fft::GPUK::cu_FFT2_R2_1st_R2C_uc8_cplxf << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc_v2, _pitchdst_v2);
        break;

    case 3:
        decx::dsp::fft::GPUK::cu_FFT2_R3_1st_R2C_uc8_cplxf << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc_v2, _pitchdst_v2);
        break;

    case 4:
        decx::dsp::fft::GPUK::cu_FFT2_R4_1st_R2C_uc8_cplxf << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc_v2, _pitchdst_v2);
        break;

    case 5:
        decx::dsp::fft::GPUK::cu_FFT2_R5_1st_R2C_uc8_cplxf << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc_v2, _pitchdst_v2);
        break;
    default:
        break;
    }
}




static void decx::dsp::fft::
FFT2D_1st_R2C_caller_uc8_cplxd(const uint8_t* __restrict src, 
                               double2* __restrict dst,
                               const uint8_t _radix, 
                               const uint32_t _signal_len, 
                               const uint32_t _pitchsrc_v1, 
                               const uint32_t _pitchdst_v1,
                               decx::cuda_stream* S)
{
    dim3 _grid(decx::utils::ceil<uint32_t>(min(_pitchsrc_v1, _pitchdst_v1), _FFT2D_BLOCK_X_),
               decx::utils::ceil<uint32_t>(_signal_len / _radix, _FFT2D_BLOCK_Y_));
    dim3 _block(_FFT2D_BLOCK_X_, _FFT2D_BLOCK_Y_);

    switch (_radix)
    {
    case 2:
        decx::dsp::fft::GPUK::cu_FFT2_R2_1st_R2C_uc8_cplxd << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc_v1, _pitchdst_v1);
        break;

    case 3:
        decx::dsp::fft::GPUK::cu_FFT2_R3_1st_R2C_uc8_cplxd << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc_v1, _pitchdst_v1);
        break;

    case 4:
        decx::dsp::fft::GPUK::cu_FFT2_R4_1st_R2C_uc8_cplxd << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc_v1, _pitchdst_v1);
        break;

    case 5:
        decx::dsp::fft::GPUK::cu_FFT2_R5_1st_R2C_uc8_cplxd << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc_v1, _pitchdst_v1);
        break;
    default:
        break;
    }
}


template <bool _div> static void 
decx::dsp::fft::FFT2D_1st_C2C_caller_cplxf(const float4* __restrict src, 
                                           float4* __restrict dst,
                                           const uint8_t _radix, 
                                           const uint32_t _signal_len, 
                                           const uint32_t _pitchsrc_v2, 
                                           const uint32_t _pitchdst_v2,
                                           decx::cuda_stream* S,
                                           const uint64_t _div_len)
{
    dim3 _grid(decx::utils::ceil<uint32_t>(min(_pitchsrc_v2, _pitchdst_v2), _FFT2D_BLOCK_X_),
               decx::utils::ceil<uint32_t>(_signal_len / _radix, _FFT2D_BLOCK_Y_));
    dim3 _block(_FFT2D_BLOCK_X_, _FFT2D_BLOCK_Y_);

    switch (_radix)
    {
    case 2:
        decx::dsp::fft::GPUK::cu_FFT2_R2_1st_C2C_cplxf<_div> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc_v2, _pitchdst_v2, _div_len);
        break;

    case 3:
        decx::dsp::fft::GPUK::cu_FFT2_R3_1st_C2C_cplxf<_div> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc_v2, _pitchdst_v2, _div_len);
        break;

    case 4:
        decx::dsp::fft::GPUK::cu_FFT2_R4_1st_C2C_cplxf<_div> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc_v2, _pitchdst_v2, _div_len);
        break;

    case 5:
        decx::dsp::fft::GPUK::cu_FFT2_R5_1st_C2C_cplxf<_div> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc_v2, _pitchdst_v2, _div_len);
        break;
    default:
        break;
    }
}


template <bool _conj> static void 
decx::dsp::fft::FFT2D_C2C_caller_cplxf(const float4* __restrict src, 
                                       float4* __restrict dst,
                                       const uint8_t _radix, 
                                       const decx::dsp::fft::FKI_4_2DK* _kernel_info,
                                       const uint32_t _pitchsrc_v2, 
                                       const uint32_t _pitchdst_v2,
                                       decx::cuda_stream* S)
{
    dim3 _grid(decx::utils::ceil<uint32_t>(min(_pitchsrc_v2, _pitchdst_v2), _FFT2D_BLOCK_X_),
               decx::utils::ceil<uint32_t>(_kernel_info->_signal_len / _radix, _FFT2D_BLOCK_Y_));
    dim3 _block(_FFT2D_BLOCK_X_, _FFT2D_BLOCK_Y_);

    switch (_radix)
    {
    case 2:
        decx::dsp::fft::GPUK::cu_FFT2_R2_C2C_cplxf<_conj> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc_v2, _pitchdst_v2);
        break;

    case 3:
        decx::dsp::fft::GPUK::cu_FFT2_R3_C2C_cplxf<_conj> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc_v2, _pitchdst_v2);
        break;

    case 4:
        decx::dsp::fft::GPUK::cu_FFT2_R4_C2C_cplxf<_conj> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc_v2, _pitchdst_v2);
        break;

    case 5:
        decx::dsp::fft::GPUK::cu_FFT2_R5_C2C_cplxf<_conj> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc_v2, _pitchdst_v2);
        break;
    default:
        break;
    }
}


static void
decx::dsp::fft::IFFT2D_C2R_caller_cplxf_u8(const float4* __restrict src, 
                                       uchar2* __restrict dst,
                                       const uint8_t _radix, 
                                       const decx::dsp::fft::FKI_4_2DK* _kernel_info,
                                       const uint32_t _pitchsrc_v2, 
                                       const uint32_t _pitchdst_v2,
                                       decx::cuda_stream* S)
{
    dim3 _grid(decx::utils::ceil<uint32_t>(min(_pitchsrc_v2, _pitchdst_v2), _FFT2D_BLOCK_X_),
               decx::utils::ceil<uint32_t>(_kernel_info->_signal_len / _radix, _FFT2D_BLOCK_Y_));
    dim3 _block(_FFT2D_BLOCK_X_, _FFT2D_BLOCK_Y_);

    switch (_radix)
    {
    case 2:
        decx::dsp::fft::GPUK::cu_FFT2_R2_C2R_cplxf_u8 << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc_v2, _pitchdst_v2);
        break;

    case 3:
        decx::dsp::fft::GPUK::cu_FFT2_R3_C2R_cplxf_u8 << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc_v2, _pitchdst_v2);
        break;

    case 4:
        decx::dsp::fft::GPUK::cu_FFT2_R4_C2R_cplxf_u8 << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc_v2, _pitchdst_v2);
        break;

    case 5:
        decx::dsp::fft::GPUK::cu_FFT2_R5_C2R_cplxf_u8 << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc_v2, _pitchdst_v2);
        break;
    default:
        break;
    }
}



static void
decx::dsp::fft::IFFT2D_C2R_caller_cplxd_u8(const double2* __restrict src, 
                                       uint8_t* __restrict dst,
                                       const uint8_t _radix, 
                                       const decx::dsp::fft::FKI_4_2DK* _kernel_info,
                                       const uint32_t _pitchsrc_v1, 
                                       const uint32_t _pitchdst_v1,
                                       decx::cuda_stream* S)
{
    dim3 _grid(decx::utils::ceil<uint32_t>(min(_pitchsrc_v1, _pitchdst_v1), _FFT2D_BLOCK_X_),
               decx::utils::ceil<uint32_t>(_kernel_info->_signal_len / _radix, _FFT2D_BLOCK_Y_));
    dim3 _block(_FFT2D_BLOCK_X_, _FFT2D_BLOCK_Y_);

    switch (_radix)
    {
    case 2:
        decx::dsp::fft::GPUK::cu_FFT2_R2_C2R_cplxd_u8 << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc_v1, _pitchdst_v1);
        break;

    case 3:
        decx::dsp::fft::GPUK::cu_FFT2_R3_C2R_cplxd_u8 << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc_v1, _pitchdst_v1);
        break;

    case 4:
        decx::dsp::fft::GPUK::cu_FFT2_R4_C2R_cplxd_u8 << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc_v1, _pitchdst_v1);
        break;

    case 5:
        decx::dsp::fft::GPUK::cu_FFT2_R5_C2R_cplxd_u8 << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc_v1, _pitchdst_v1);
        break;
    default:
        break;
    }
}



static void
decx::dsp::fft::IFFT2D_C2R_caller_cplxf_fp32(const float4* __restrict src, 
                                       float2* __restrict dst,
                                       const uint8_t _radix, 
                                       const decx::dsp::fft::FKI_4_2DK* _kernel_info,
                                       const uint32_t _pitchsrc_v2, 
                                       const uint32_t _pitchdst_v2,
                                       decx::cuda_stream* S)
{
    dim3 _grid(decx::utils::ceil<uint32_t>(min(_pitchsrc_v2, _pitchdst_v2), _FFT2D_BLOCK_X_),
               decx::utils::ceil<uint32_t>(_kernel_info->_signal_len / _radix, _FFT2D_BLOCK_Y_));
    dim3 _block(_FFT2D_BLOCK_X_, _FFT2D_BLOCK_Y_);

    switch (_radix)
    {
    case 2:
        decx::dsp::fft::GPUK::cu_FFT2_R2_C2R_cplxf_fp32 << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc_v2, _pitchdst_v2);
        break;

    case 3:
        decx::dsp::fft::GPUK::cu_FFT2_R3_C2R_cplxf_fp32 << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc_v2, _pitchdst_v2);
        break;

    case 4:
        decx::dsp::fft::GPUK::cu_FFT2_R4_C2R_cplxf_fp32 << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc_v2, _pitchdst_v2);
        break;

    case 5:
        decx::dsp::fft::GPUK::cu_FFT2_R5_C2R_cplxf_fp32 << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc_v2, _pitchdst_v2);
        break;
    default:
        break;
    }
}



#endif
