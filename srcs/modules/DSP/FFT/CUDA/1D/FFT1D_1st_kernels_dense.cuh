/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _FFT1D_1st_KERNELS_DENSE_CUH_
#define _FFT1D_1st_KERNELS_DENSE_CUH_ 


#include "../../../../core/basic.h"
#include "../../../../core/utils/decx_cuda_vectypes_ops.cuh"
#include "../../../../core/utils/decx_cuda_math_functions.cuh"
#include "../../FFT_commons.h"
#include "../2D/FFT2D_config.cuh"
#include "../2D/FFT2D_kernels.cuh"


namespace decx
{
namespace dsp {
namespace fft {
    namespace GPUK 
    {
        // R2C 1st
        __global__ void cu_FFT2D_R5_1st_cplxf_R2C_dense(const float* __restrict src, float2* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc, const uint32_t _pitchdst);


        __global__ void cu_FFT2D_R4_1st_cplxf_R2C_dense(const float* __restrict src, float2* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc, const uint32_t _pitchdst);


        __global__ void cu_FFT2D_R3_1st_cplxf_R2C_dense(const float* __restrict src, float2* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc, const uint32_t _pitchdst);


        __global__ void cu_FFT2D_R2_1st_cplxf_R2C_dense(const float* __restrict src, float2* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc, const uint32_t _pitchdst);


        // C2C 1st
        template <bool _div>
        __global__ void cu_FFT2D_R5_1st_cplxf_C2C_dense(const float2* __restrict src, float2* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc, const uint32_t _pitchdst, const uint64_t _div_len);


        template <bool _div>
        __global__ void cu_FFT2D_R4_1st_cplxf_C2C_dense(const float2* __restrict src, float2* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc, const uint32_t _pitchdst, const uint64_t _div_len);


        template <bool _div>
        __global__ void cu_FFT2D_R3_1st_cplxf_C2C_dense(const float2* __restrict src, float2* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc, const uint32_t _pitchdst, const uint64_t _div_len);


        template <bool _div>
        __global__ void cu_FFT2D_R2_1st_cplxf_C2C_dense(const float2* __restrict src, float2* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc, const uint32_t _pitchdst, const uint64_t _div_len);

        // C2C end
        template <bool _conj>
        __global__ void cu_FFT2D_R5_end_cplxf_C2C_dense(const float2* __restrict src, float2* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc, const uint32_t _pitchdst);


        template <bool _conj>
        __global__ void cu_FFT2D_R4_end_cplxf_C2C_dense(const float2* __restrict src, float2* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc, const uint32_t _pitchdst);


        template <bool _conj>
        __global__ void cu_FFT2D_R3_end_cplxf_C2C_dense(const float2* __restrict src, float2* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc, const uint32_t _pitchdst);


        template <bool _conj>
        __global__ void cu_FFT2D_R2_end_cplxf_C2C_dense(const float2* __restrict src, float2* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc, const uint32_t _pitchdst);


        // C2R end
        __global__ void cu_FFT2D_R5_end_cplxf_C2R_dense(const float2* __restrict src, float* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc, const uint32_t _pitchdst);


        __global__ void cu_FFT2D_R4_end_cplxf_C2R_dense(const float2* __restrict src, float* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc, const uint32_t _pitchdst);


        __global__ void cu_FFT2D_R3_end_cplxf_C2R_dense(const float2* __restrict src, float* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc, const uint32_t _pitchdst);


        __global__ void cu_FFT2D_R2_end_cplxf_C2R_dense(const float2* __restrict src, float* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc, const uint32_t _pitchdst);
    }


    static void FFT1D_1st_R2C_caller_cplxf_dense(const float* __restrict src, float2* __restrict dst,
        const uint8_t _radix, const uint32_t _signal_len, const uint32_t _pitchsrc, const uint32_t _pitchdst,
        decx::cuda_stream* S);


    template <bool _div>
    static void FFT1D_1st_C2C_caller_cplxf_dense(const float2* __restrict src, float2* __restrict dst,
        const uint8_t _radix, const uint32_t _signal_len_fractioned, const uint64_t _signal_len_total, const uint32_t _pitchsrc, const uint32_t _pitchdst,
        decx::cuda_stream* S);


    template <bool _conj>
    static void FFT1D_end_C2C_caller_cplxf_dense(const float2* __restrict src, float2* __restrict dst,
        const uint8_t _radix, const decx::dsp::fft::FKI_4_2DK* _kernel_info, const uint32_t _pitchsrc, const uint32_t _pitchdst,
        decx::cuda_stream* S);


    static void FFT1D_end_C2R_caller_cplxf_dense(const float2* __restrict src, float* __restrict dst,
        const uint8_t _radix, const decx::dsp::fft::FKI_4_2DK* _kernel_info, const uint32_t _pitchsrc, const uint32_t _pitchdst,
        decx::cuda_stream* S);
}
}
}


static void 
decx::dsp::fft::FFT1D_1st_R2C_caller_cplxf_dense(const float* __restrict src, 
                                                 float2* __restrict dst,
                                                 const uint8_t _radix, 
                                                 const uint32_t _signal_len, 
                                                 const uint32_t _pitchsrc, 
                                                 const uint32_t _pitchdst,
                                                 decx::cuda_stream* S)
{
    dim3 _grid(decx::utils::ceil<uint32_t>(min(_pitchsrc, _pitchdst), _FFT2D_BLOCK_X_),
        decx::utils::ceil<uint32_t>(_signal_len / _radix, _FFT2D_BLOCK_Y_));
    dim3 _block(_FFT2D_BLOCK_X_, _FFT2D_BLOCK_Y_);

    switch (_radix)
    {
    case 5:
        decx::dsp::fft::GPUK::cu_FFT2D_R5_1st_cplxf_R2C_dense << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc, _pitchdst);
        break;

    case 4:
        decx::dsp::fft::GPUK::cu_FFT2D_R4_1st_cplxf_R2C_dense << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc, _pitchdst);
        break;

    case 3:
        decx::dsp::fft::GPUK::cu_FFT2D_R3_1st_cplxf_R2C_dense << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc, _pitchdst);
        break;

    case 2:
        decx::dsp::fft::GPUK::cu_FFT2D_R2_1st_cplxf_R2C_dense << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc, _pitchdst);
        break;
    default:
        break;
    }
}



template <bool _div> static void 
decx::dsp::fft::FFT1D_1st_C2C_caller_cplxf_dense(const float2* __restrict src, 
                                                 float2* __restrict dst,
                                                 const uint8_t _radix, 
                                                 const uint32_t _signal_len_fractioned, 
                                                 const uint64_t _signal_len_total,
                                                 const uint32_t _pitchsrc, 
                                                 const uint32_t _pitchdst,
                                                 decx::cuda_stream* S)
{
    dim3 _grid(decx::utils::ceil<uint32_t>(min(_pitchsrc, _pitchdst), _FFT2D_BLOCK_X_),
        decx::utils::ceil<uint32_t>(_signal_len_fractioned / _radix, _FFT2D_BLOCK_Y_));
    dim3 _block(_FFT2D_BLOCK_X_, _FFT2D_BLOCK_Y_);

    switch (_radix)
    {
    case 5:
        decx::dsp::fft::GPUK::cu_FFT2D_R5_1st_cplxf_C2C_dense<_div> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len_fractioned, _pitchsrc, _pitchdst, _signal_len_total);
        break;

    case 4:
        decx::dsp::fft::GPUK::cu_FFT2D_R4_1st_cplxf_C2C_dense<_div> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len_fractioned, _pitchsrc, _pitchdst, _signal_len_total);
        break;

    case 3:
        decx::dsp::fft::GPUK::cu_FFT2D_R3_1st_cplxf_C2C_dense<_div> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len_fractioned, _pitchsrc, _pitchdst, _signal_len_total);
        break;

    case 2:
        decx::dsp::fft::GPUK::cu_FFT2D_R2_1st_cplxf_C2C_dense<_div> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len_fractioned, _pitchsrc, _pitchdst, _signal_len_total);
        break;
    default:
        break;
    }
}


template <bool _conj> static void
decx::dsp::fft::FFT1D_end_C2C_caller_cplxf_dense(const float2* __restrict src,
                                                 float2* __restrict dst,
                                                 const uint8_t _radix, 
                                                 const decx::dsp::fft::FKI_4_2DK* _kernel_info, 
                                                 const uint32_t _pitchsrc, 
                                                 const uint32_t _pitchdst,
                                                 decx::cuda_stream* S)
{
    dim3 _grid(decx::utils::ceil<uint32_t>(min(_pitchsrc, _pitchdst), _FFT2D_BLOCK_X_),
        decx::utils::ceil<uint32_t>(_kernel_info->_signal_len / _radix, _FFT2D_BLOCK_Y_));
    dim3 _block(_FFT2D_BLOCK_X_, _FFT2D_BLOCK_Y_);

    switch (_radix)
    {
    case 5:
        decx::dsp::fft::GPUK::cu_FFT2D_R5_end_cplxf_C2C_dense<_conj> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc, _pitchdst);
        break;

    case 4:
        decx::dsp::fft::GPUK::cu_FFT2D_R4_end_cplxf_C2C_dense<_conj> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc, _pitchdst);
        break;

    case 3:
        decx::dsp::fft::GPUK::cu_FFT2D_R3_end_cplxf_C2C_dense<_conj> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc, _pitchdst);
        break;

    case 2:
        decx::dsp::fft::GPUK::cu_FFT2D_R2_end_cplxf_C2C_dense<_conj> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc, _pitchdst);
        break;
    default:
        break;
    }
}



static void
decx::dsp::fft::FFT1D_end_C2R_caller_cplxf_dense(const float2* __restrict src,
                                                 float* __restrict dst,
                                                 const uint8_t _radix, 
                                                 const decx::dsp::fft::FKI_4_2DK* _kernel_info, 
                                                 const uint32_t _pitchsrc, 
                                                 const uint32_t _pitchdst,
                                                 decx::cuda_stream* S)
{
    dim3 _grid(decx::utils::ceil<uint32_t>(min(_pitchsrc, _pitchdst), _FFT2D_BLOCK_X_),
        decx::utils::ceil<uint32_t>(_kernel_info->_signal_len / _radix, _FFT2D_BLOCK_Y_));
    dim3 _block(_FFT2D_BLOCK_X_, _FFT2D_BLOCK_Y_);

    switch (_radix)
    {
    case 5:
        decx::dsp::fft::GPUK::cu_FFT2D_R5_end_cplxf_C2R_dense << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc, _pitchdst);
        break;

    case 4:
        decx::dsp::fft::GPUK::cu_FFT2D_R4_end_cplxf_C2R_dense << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc, _pitchdst);
        break;

    case 3:
        decx::dsp::fft::GPUK::cu_FFT2D_R3_end_cplxf_C2R_dense << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc, _pitchdst);
        break;

    case 2:
        decx::dsp::fft::GPUK::cu_FFT2D_R2_end_cplxf_C2R_dense << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc, _pitchdst);
        break;
    default:
        break;
    }
}



namespace decx
{
namespace dsp {
    namespace fft 
    {
        static void FFT1D_1st_R2C_caller_cplxf_caller(const void* __restrict src, void* __restrict dst,
            const uint8_t _radix, const uint32_t _signal_len, const uint32_t _pitchsrc, const uint32_t _pitchdst,
            decx::cuda_stream* S);


        template <bool _div>
        static void FFT1D_1st_C2C_caller_cplxf_caller(const void* __restrict src, void* __restrict dst,
            const uint8_t _radix, const uint32_t _signal_len_fractioned, const uint64_t _signal_len_total, const uint32_t _pitchsrc, const uint32_t _pitchdst,
            decx::cuda_stream* S);


        template <bool _conj>
        static void FFT1D_end_C2C_caller_cplxf_caller(const void* __restrict src, void* __restrict dst,
            const uint8_t _radix, const decx::dsp::fft::FKI_4_2DK* _kernel_info, const uint32_t _pitchsrc, const uint32_t _pitchdst,
            decx::cuda_stream* S);


        static void IFFT1D_end_C2R_caller_cplxf_caller(const void* __restrict src, void* __restrict dst,
            const uint8_t _radix, const decx::dsp::fft::FKI_4_2DK* _kernel_info, const uint32_t _pitchsrc, const uint32_t _pitchdst,
            decx::cuda_stream* S);
    }
}
}


template <bool _conj> static void 
decx::dsp::fft::FFT1D_end_C2C_caller_cplxf_caller(const void* __restrict src, 
                                                  void* __restrict dst,
                                                  const uint8_t _radix, 
                                                  const decx::dsp::fft::FKI_4_2DK* _kernel_info, 
                                                  const uint32_t _pitchsrc, 
                                                  const uint32_t _pitchdst,
                                                  decx::cuda_stream* S)
{
    // Call vectorized kernel if possible
    if (_pitchdst % 2) {        // The pitchdst is odd, call unvectorized kernel
        decx::dsp::fft::FFT1D_end_C2C_caller_cplxf_dense<_conj>((const float2*)src, (float2*)dst,
            _radix, _kernel_info, _pitchsrc, _pitchdst, S);
    }
    else {  // The pitchdst is even, call vectorized kernel
        decx::dsp::fft::FFT2D_C2C_caller_cplxf<_conj>((const float4*)src, (float4*)dst,
            _radix, _kernel_info, _pitchsrc / 2, _pitchdst / 2, S);
    }
}



static void 
decx::dsp::fft::FFT1D_1st_R2C_caller_cplxf_caller(const void* __restrict src, 
                                                  void* __restrict dst,
                                                  const uint8_t _radix, 
                                                  const uint32_t _signal_len, 
                                                  const uint32_t _pitchsrc, 
                                                  const uint32_t _pitchdst,
                                                  decx::cuda_stream* S)
{
    // Call vectorized kernel if possible
    if (_pitchsrc % 2) {        // The pitchdst is odd, call unvectorized kernel
        decx::dsp::fft::FFT1D_1st_R2C_caller_cplxf_dense((const float*)src, (float2*)dst,
            _radix, _signal_len, _pitchsrc, _pitchdst, S);
    }
    else {  // The pitchdst is even, call vectorized kernel
        decx::dsp::fft::FFT2D_1st_R2C_caller_cplxf((const float2*)src, (float4*)dst,
            _radix, _signal_len, _pitchsrc / 2, _pitchdst / 2, S);
    }
}



template <bool _div> static void 
decx::dsp::fft::FFT1D_1st_C2C_caller_cplxf_caller(const void* __restrict src, 
                                                  void* __restrict dst,
                                                  const uint8_t _radix, 
                                                  const uint32_t _signal_len_fractioned, 
                                                  const uint64_t _signal_len_total,
                                                  const uint32_t _pitchsrc, 
                                                  const uint32_t _pitchdst,
                                                  decx::cuda_stream* S)
{
    // Call vectorized kernel if possible
    if (_pitchsrc % 2) {        // The pitchdst is odd, call unvectorized kernel
        decx::dsp::fft::FFT1D_1st_C2C_caller_cplxf_dense<_div>((const float2*)src, (float2*)dst,
            _radix, _signal_len_fractioned, _signal_len_total, _pitchsrc, _pitchdst, S);
    }
    else {  // The pitchdst is even, call vectorized kernel
        decx::dsp::fft::FFT2D_1st_C2C_caller_cplxf<_div>((const float4*)src, (float4*)dst,
            _radix, _signal_len_fractioned, _pitchsrc / 2, _pitchdst / 2, S, _signal_len_total);
    }
}



static void 
decx::dsp::fft::IFFT1D_end_C2R_caller_cplxf_caller(const void* __restrict src, 
                                                  void* __restrict dst,
                                                  const uint8_t _radix, 
                                                  const decx::dsp::fft::FKI_4_2DK* _kernel_info, 
                                                  const uint32_t _pitchsrc, 
                                                  const uint32_t _pitchdst,
                                                  decx::cuda_stream* S)
{
    // Call vectorized kernel if possible
    if (_pitchdst % 2) {        // The pitchdst is odd, call unvectorized kernel
        decx::dsp::fft::FFT1D_end_C2R_caller_cplxf_dense((const float2*)src, (float*)dst,
            _radix, _kernel_info, _pitchsrc, _pitchdst, S);
    }
    else {  // The pitchdst is even, call vectorized kernel
        decx::dsp::fft::IFFT2D_C2R_caller_cplxf_fp32((const float4*)src, (float2*)dst,
            _radix, _kernel_info, _pitchsrc / 2, _pitchdst / 2, S);
    }
}


// ---------------------------------------------------------- double ----------------------------------------------------------


namespace decx
{
namespace dsp {
namespace fft {
    namespace GPUK 
    {
        // R2C 1st
        __global__ void cu_FFT2D_R5_1st_cplxd_R2C_dense(const double* __restrict src, double2* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc, const uint32_t _pitchdst);


        __global__ void cu_FFT2D_R4_1st_cplxd_R2C_dense(const double* __restrict src, double2* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc, const uint32_t _pitchdst);


        __global__ void cu_FFT2D_R3_1st_cplxd_R2C_dense(const double* __restrict src, double2* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc, const uint32_t _pitchdst);


        __global__ void cu_FFT2D_R2_1st_cplxd_R2C_dense(const double* __restrict src, double2* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc, const uint32_t _pitchdst);


        // C2C 1st
        template <bool _div>
        __global__ void cu_FFT2D_R5_1st_cplxd_C2C_dense(const double2* __restrict src, double2* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc, const uint32_t _pitchdst, const uint64_t _div_len);


        template <bool _div>
        __global__ void cu_FFT2D_R4_1st_cplxd_C2C_dense(const double2* __restrict src, double2* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc, const uint32_t _pitchdst, const uint64_t _div_len);


        template <bool _div>
        __global__ void cu_FFT2D_R3_1st_cplxd_C2C_dense(const double2* __restrict src, double2* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc, const uint32_t _pitchdst, const uint64_t _div_len);


        template <bool _div>
        __global__ void cu_FFT2D_R2_1st_cplxd_C2C_dense(const double2* __restrict src, double2* __restrict dst,
            const uint32_t _signal_len, const uint32_t _pitchsrc, const uint32_t _pitchdst, const uint64_t _div_len);

        // C2C end
        template <bool _conj>
        __global__ void cu_FFT2D_R5_end_cplxd_C2C_dense(const double2* __restrict src, double2* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc, const uint32_t _pitchdst);


        template <bool _conj>
        __global__ void cu_FFT2D_R4_end_cplxd_C2C_dense(const double2* __restrict src, double2* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc, const uint32_t _pitchdst);


        template <bool _conj>
        __global__ void cu_FFT2D_R3_end_cplxd_C2C_dense(const double2* __restrict src, double2* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc, const uint32_t _pitchdst);


        template <bool _conj>
        __global__ void cu_FFT2D_R2_end_cplxd_C2C_dense(const double2* __restrict src, double2* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc, const uint32_t _pitchdst);


        // C2R end
        __global__ void cu_FFT2D_R5_end_cplxd_C2R_dense(const double2* __restrict src, double* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc, const uint32_t _pitchdst);


        __global__ void cu_FFT2D_R4_end_cplxd_C2R_dense(const double2* __restrict src, double* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc, const uint32_t _pitchdst);


        __global__ void cu_FFT2D_R3_end_cplxd_C2R_dense(const double2* __restrict src, double* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc, const uint32_t _pitchdst);


        __global__ void cu_FFT2D_R2_end_cplxd_C2R_dense(const double2* __restrict src, double* __restrict dst,
            const decx::dsp::fft::FKI_4_2DK _kernel_info, const uint32_t _pitchsrc, const uint32_t _pitchdst);
    }

    static void FFT1D_1st_R2C_caller_cplxd_dense(const double* __restrict src, double2* __restrict dst,
        const uint8_t _radix, const uint32_t _signal_len, const uint32_t _pitchsrc, const uint32_t _pitchdst,
        decx::cuda_stream* S);


    template <bool _div>
    static void FFT2D_1st_C2C_caller_cplxd(const double2* __restrict src, double2* __restrict dst,
        const uint8_t _radix, const uint32_t _signal_len_fractioned, const uint32_t _pitchsrc, 
        const uint32_t _pitchdst, decx::cuda_stream* S, const uint64_t _signal_len_total = 0);


    template <bool _conj>
    static void FFT2D_C2C_caller_cplxd(const double2* __restrict src, double2* __restrict dst,
        const uint8_t _radix, const decx::dsp::fft::FKI_4_2DK* _kernel_info, const uint32_t _pitchsrc_v1,
        const uint32_t _pitchdst_v1, decx::cuda_stream* S);


    static void FFT2D_end_C2R_caller_cplxd(const double2* __restrict src, double* __restrict dst,
        const uint8_t _radix, const decx::dsp::fft::FKI_4_2DK* _kernel_info,
        const uint32_t _pitchsrc, const uint32_t _pitchdst, decx::cuda_stream* S);
}
}
}


// Copy to the buffer with aligned pitch (ensuring coalsecing memory access)
static void 
decx::dsp::fft::FFT1D_1st_R2C_caller_cplxd_dense(const double* __restrict src, 
                                                 double2* __restrict dst,
                                                 const uint8_t _radix, 
                                                 const uint32_t _signal_len, 
                                                 const uint32_t _pitchsrc, 
                                                 const uint32_t _pitchdst,
                                                 decx::cuda_stream* S)
{
    dim3 _grid(decx::utils::ceil<uint32_t>(min(_pitchsrc, _pitchdst), _FFT2D_BLOCK_X_),
        decx::utils::ceil<uint32_t>(_signal_len / _radix, _FFT2D_BLOCK_Y_));
    dim3 _block(_FFT2D_BLOCK_X_, _FFT2D_BLOCK_Y_);

    switch (_radix)
    {
    case 5:
        decx::dsp::fft::GPUK::cu_FFT2D_R5_1st_cplxd_R2C_dense << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc, _pitchdst);
        break;

    case 4:
        decx::dsp::fft::GPUK::cu_FFT2D_R4_1st_cplxd_R2C_dense << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc, _pitchdst);
        break;

    case 3:
        decx::dsp::fft::GPUK::cu_FFT2D_R3_1st_cplxd_R2C_dense << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc, _pitchdst);
        break;

    case 2:
        decx::dsp::fft::GPUK::cu_FFT2D_R2_1st_cplxd_R2C_dense << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len, _pitchsrc, _pitchdst);
        break;
    default:
        break;
    }
}



template <bool _div> static void 
decx::dsp::fft::FFT2D_1st_C2C_caller_cplxd(const double2* __restrict src, 
                                           double2* __restrict dst,
                                           const uint8_t _radix, 
                                           const uint32_t _signal_len_fractioned, 
                                           const uint32_t _pitchsrc, 
                                           const uint32_t _pitchdst,
                                           decx::cuda_stream* S,
                                           const uint64_t _signal_len_total)
{
    dim3 _grid(decx::utils::ceil<uint32_t>(min(_pitchsrc, _pitchdst), _FFT2D_BLOCK_X_),
        decx::utils::ceil<uint32_t>(_signal_len_fractioned / _radix, _FFT2D_BLOCK_Y_));
    dim3 _block(_FFT2D_BLOCK_X_, _FFT2D_BLOCK_Y_);

    switch (_radix)
    {
    case 5:
        decx::dsp::fft::GPUK::cu_FFT2D_R5_1st_cplxd_C2C_dense<_div> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len_fractioned, _pitchsrc, _pitchdst, _signal_len_total);
        break;

    case 4:
        decx::dsp::fft::GPUK::cu_FFT2D_R4_1st_cplxd_C2C_dense<_div> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len_fractioned, _pitchsrc, _pitchdst, _signal_len_total);
        break;

    case 3:
        decx::dsp::fft::GPUK::cu_FFT2D_R3_1st_cplxd_C2C_dense<_div> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len_fractioned, _pitchsrc, _pitchdst, _signal_len_total);
        break;

    case 2:
        decx::dsp::fft::GPUK::cu_FFT2D_R2_1st_cplxd_C2C_dense<_div> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, _signal_len_fractioned, _pitchsrc, _pitchdst, _signal_len_total);
        break;
    default:
        break;
    }
}


template <bool _conj> static void 
decx::dsp::fft::FFT2D_C2C_caller_cplxd(const double2* __restrict src,
                                       double2* __restrict dst,
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
    case 5:
        decx::dsp::fft::GPUK::cu_FFT2D_R5_end_cplxd_C2C_dense<_conj> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc_v1, _pitchdst_v1);
        break;

    case 4:
        decx::dsp::fft::GPUK::cu_FFT2D_R4_end_cplxd_C2C_dense<_conj> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc_v1, _pitchdst_v1);
        break;

    case 3:
        decx::dsp::fft::GPUK::cu_FFT2D_R3_end_cplxd_C2C_dense<_conj> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc_v1, _pitchdst_v1);
        break;

    case 2:
        decx::dsp::fft::GPUK::cu_FFT2D_R2_end_cplxd_C2C_dense<_conj> << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc_v1, _pitchdst_v1);
        break;
    default:
        break;
    }
}



static void
decx::dsp::fft::FFT2D_end_C2R_caller_cplxd(const double2* __restrict src,
                                           double* __restrict dst,
                                           const uint8_t _radix, 
                                           const decx::dsp::fft::FKI_4_2DK* _kernel_info, 
                                           const uint32_t _pitchsrc, 
                                           const uint32_t _pitchdst,
                                           decx::cuda_stream* S)
{
    dim3 _grid(decx::utils::ceil<uint32_t>(min(_pitchsrc, _pitchdst), _FFT2D_BLOCK_X_),
        decx::utils::ceil<uint32_t>(_kernel_info->_signal_len / _radix, _FFT2D_BLOCK_Y_));
    dim3 _block(_FFT2D_BLOCK_X_, _FFT2D_BLOCK_Y_);

    switch (_radix)
    {
    case 5:
        decx::dsp::fft::GPUK::cu_FFT2D_R5_end_cplxd_C2R_dense << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc, _pitchdst);
        break;

    case 4:
        decx::dsp::fft::GPUK::cu_FFT2D_R4_end_cplxd_C2R_dense << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc, _pitchdst);
        break;

    case 3:
        decx::dsp::fft::GPUK::cu_FFT2D_R3_end_cplxd_C2R_dense << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc, _pitchdst);
        break;

    case 2:
        decx::dsp::fft::GPUK::cu_FFT2D_R2_end_cplxd_C2R_dense << <_grid, _block, 0, S->get_raw_stream_ref() >> > (
            src, dst, *_kernel_info, _pitchsrc, _pitchdst);
        break;
    default:
        break;
    }
}



#endif
