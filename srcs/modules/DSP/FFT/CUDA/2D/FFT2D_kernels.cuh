/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
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
        cu_FFT2_R2_C2R_cplxf_fp32(const float4* __restrict, float2* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2);


        // ------------------------------------------------ Radix-3 ------------------------------------------------
        __global__ void cu_FFT2_R3_1st_R2C_cplxf(const float2* __restrict src, float4* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc_v2, const uint32_t _putchdst_v2);


        __global__ void cu_FFT2_R3_1st_R2C_uc8_cplxf(const ushort* __restrict src, float4* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc_v2, const uint32_t _putchdst_v2);


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
        cu_FFT2_R3_C2R_cplxf_fp32(const float4* __restrict, float2* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2);

        // ------------------------------------------------ Radix-4 ------------------------------------------------
        __global__ void cu_FFT2_R4_1st_R2C_cplxf(const float2* __restrict src, float4* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc_v2, const uint32_t _putchdst_v2);


        __global__ void cu_FFT2_R4_1st_R2C_uc8_cplxf(const ushort* __restrict src, float4* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc_v2, const uint32_t _putchdst_v2);


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
        cu_FFT2_R4_C2R_cplxf_fp32(const float4* __restrict, float2* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2);


        // ------------------------------------------------ Radix-5 ------------------------------------------------
        __global__ void cu_FFT2_R5_1st_R2C_cplxf(const float2* __restrict src, float4* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2);


        __global__ void cu_FFT2_R5_1st_R2C_uc8_cplxf(const ushort* __restrict src, float4* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2);


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
        cu_FFT2_R5_C2R_cplxf_fp32(const float4* __restrict, float2* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2);
    }
    
    static void FFT2D_1st_R2C_caller_cplxf(const float2* __restrict src, float4* __restrict dst,
        const uint8_t _radix, const uint32_t _signal_len, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2,
        decx::cuda_stream* S);


    static void FFT2D_1st_R2C_caller_uc8_cplxf(const ushort* __restrict src, float4* __restrict dst,
        const uint8_t _radix, const uint32_t _signal_len, const uint32_t _pitchsrc_v2, const uint32_t _pitchdst_v2,
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


namespace decx
{
namespace dsp {
namespace fft {
    namespace GPUK 
    {
        //// ------------------------------------------------ Radix-2 ------------------------------------------------
        //__global__ void cu_FFT2_R2_1st_R2C_cplxd(const double2* __restrict src, double4* __restrict dst,
        //    const uint32_t _signal_len, const uint32_t _pitchsrc_v1, const uint32_t _putchdst_v1);


        //__global__ void cu_FFT2_R2_1st_R2C_uc8_cplxd(const ushort* __restrict src, double4* __restrict dst,
        //    const uint32_t _signal_len, const uint32_t _pitchsrc_v1, const uint32_t _putchdst_v1);


        //template <bool _div> __global__ void
        //    cu_FFT2_R2_1st_C2C_cplxd(const double4* __restrict src, double4* __restrict dst,
        //        const uint32_t _signal_len, const uint32_t _pitchsrc_v1, const uint32_t _putchdst_v1, const uint64_t _div_length = 0);


        //template <bool _conj> __global__ void
        //    cu_FFT2_R2_C2C_cplxd(const double4* __restrict, double4* __restrict dst,
        //        const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v1, const uint32_t _pitchdst_v1);


        //__global__ void
        //cu_FFT2_R2_C2R_cplxd_u8(const double4* __restrict, uchar2* __restrict dst,
        //    const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v1, const uint32_t _pitchdst_v1);


        //__global__ void
        //cu_FFT2_R2_C2R_cplxd_fp32(const double4* __restrict, double2* __restrict dst,
        //    const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v1, const uint32_t _pitchdst_v1);


        //// ------------------------------------------------ Radix-3 ------------------------------------------------
        //__global__ void cu_FFT2_R3_1st_R2C_cplxd(const double2* __restrict src, double4* __restrict dst,
        //    const uint32_t _signal_len, const uint32_t _pitchsrc_v1, const uint32_t _putchdst_v1);


        //__global__ void cu_FFT2_R3_1st_R2C_uc8_cplxd(const ushort* __restrict src, double4* __restrict dst,
        //    const uint32_t _signal_len, const uint32_t _pitchsrc_v1, const uint32_t _putchdst_v1);


        //template <bool _div> __global__ void
        //    cu_FFT2_R3_1st_C2C_cplxd(const double4* __restrict src, double4* __restrict dst,
        //        const uint32_t _signal_len, const uint32_t _pitchsrc_v1, const uint32_t _putchdst_v1, const uint64_t _div_length = 0);


        //template <bool _conj> __global__ void
        //    cu_FFT2_R3_C2C_cplxd(const double4* __restrict, double4* __restrict dst,
        //        const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v1, const uint32_t _pitchdst_v1);


        //__global__ void
        //cu_FFT2_R3_C2R_cplxd_u8(const double4* __restrict, uchar2* __restrict dst,
        //    const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v1, const uint32_t _pitchdst_v1);


        //__global__ void
        //cu_FFT2_R3_C2R_cplxd_fp32(const double4* __restrict, double2* __restrict dst,
        //    const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v1, const uint32_t _pitchdst_v1);

        //// ------------------------------------------------ Radix-4 ------------------------------------------------
        //__global__ void cu_FFT2_R4_1st_R2C_cplxd(const double2* __restrict src, double4* __restrict dst,
        //    const uint32_t _signal_len, const uint32_t _pitchsrc_v1, const uint32_t _putchdst_v1);


        //__global__ void cu_FFT2_R4_1st_R2C_uc8_cplxd(const ushort* __restrict src, double4* __restrict dst,
        //    const uint32_t _signal_len, const uint32_t _pitchsrc_v1, const uint32_t _putchdst_v1);


        //template <bool _div> __global__ void 
        //cu_FFT2_R4_1st_C2C_cplxd(const double4* __restrict src, double4* __restrict dst,
        //    const uint32_t _signal_len, const uint32_t _pitchsrc_v1, const uint32_t _putchdst_v1, const uint64_t _div_length = 0);


        //template <bool _conj> __global__ void
        //cu_FFT2_R4_C2C_cplxd(const double4* __restrict, double4* __restrict dst,
        //    const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v1, const uint32_t _pitchdst_v1);


        //__global__ void
        //cu_FFT2_R4_C2R_cplxd_u8(const double4* __restrict, uchar2* __restrict dst,
        //    const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v1, const uint32_t _pitchdst_v1);


        //__global__ void
        //cu_FFT2_R4_C2R_cplxd_fp32(const double4* __restrict, double2* __restrict dst,
        //    const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v1, const uint32_t _pitchdst_v1);


        //// ------------------------------------------------ Radix-5 ------------------------------------------------
        //__global__ void cu_FFT2_R5_1st_R2C_cplxd(const double2* __restrict src, double4* __restrict dst,
        //    const uint32_t _signal_len, const uint32_t _pitchsrc_v1, const uint32_t _pitchdst_v1);


        //__global__ void cu_FFT2_R5_1st_R2C_uc8_cplxd(const ushort* __restrict src, double4* __restrict dst,
        //    const uint32_t _signal_len, const uint32_t _pitchsrc_v1, const uint32_t _pitchdst_v1);


        //template <bool _div> __global__ void 
        //cu_FFT2_R5_1st_C2C_cplxd(const double4* __restrict src, double4* __restrict dst,
        //    const uint32_t _signal_len, const uint32_t _pitchsrc_v1, const uint32_t _putchdst_v1, const uint64_t _div_length = 0);


        //template <bool _conj> __global__ void
        //cu_FFT2_R5_C2C_cplxd(const double4* __restrict, double4* __restrict dst,
        //    const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v1, const uint32_t _pitchdst_v1);


        //__global__ void
        //cu_FFT2_R5_C2R_cplxd_u8(const double4* __restrict, uchar2* __restrict dst,
        //    const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v1, const uint32_t _pitchdst_v1);


        //__global__ void
        //cu_FFT2_R5_C2R_cplxd_fp32(const double4* __restrict, double2* __restrict dst,
        //    const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc_v1, const uint32_t _pitchdst_v1);
    }
    }
}
}



#endif