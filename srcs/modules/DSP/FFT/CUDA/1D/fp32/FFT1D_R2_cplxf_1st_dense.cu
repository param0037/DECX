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


#include "../FFT1D_1st_kernels_dense.cuh"



// [32 * 2, 8] = [64, 8]
__global__ void 
decx::dsp::fft::GPUK::cu_FFT2D_R2_1st_cplxf_R2C_dense(const float* __restrict src,
                                                      float2* __restrict dst,
                                                      const uint32_t _signal_len,
                                                      const uint32_t _pitchsrc_v2,
                                                      const uint32_t _pitchdst_v2)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _signal_len / 2;

    float recv[2];
    decx::utils::_cuda_vec64 res;

    uint32_t _FFT_domain_dex = tidy;

    if (tidy < _Bops_num && tidx < _pitchsrc_v2)
    {
#pragma unroll 2
        for (uint8_t i = 0; i < 2; ++i) {
            recv[i] = src[_FFT_domain_dex * _pitchsrc_v2 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    _FFT_domain_dex = tidy * 2;

    if (tidy < _Bops_num && tidx < _pitchdst_v2)
    {
        res._cplxf32.image = 0.f;
        res._cplxf32.real = __fadd_rn(recv[0], recv[1]);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf2;
        ++_FFT_domain_dex;

        res._cplxf32.real = __fsub_rn(recv[0], recv[1]);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf2;
    }
}


template<bool _div> __global__ void 
decx::dsp::fft::GPUK::cu_FFT2D_R2_1st_cplxf_C2C_dense(const float2* __restrict src,
                                               float2* __restrict dst,
                                               const uint32_t _signal_len,
                                               const uint32_t _pitchsrc,
                                               const uint32_t _pitchdst,
                                               const uint64_t _div_length)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const float _numer = __ull2float_rn(_div_length ? _div_length : _signal_len);
    const uint32_t _Bops_num = _signal_len / 2;

    decx::utils::_cuda_vec64 recv[2];
    decx::utils::_cuda_vec64 res;

    uint32_t _FFT_domain_dex = tidy;

    if (tidy < _Bops_num && tidx < _pitchsrc)
    {
#pragma unroll 2
        for (uint8_t i = 0; i < 2; ++i) {
            recv[i]._vf2 = src[_FFT_domain_dex * _pitchsrc + tidx];
            if (_div) { recv[i]._vf2 = decx::utils::cuda::__float_div2_1(recv[i]._vf2, _numer); }
            _FFT_domain_dex += _Bops_num;
        }
    }

    _FFT_domain_dex = tidy * 2;

    if (tidy < _Bops_num && tidx < _pitchdst)
    {
        res._vf2 = decx::utils::cuda::__float_add2(recv[0]._vf2, recv[1]._vf2);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
        ++_FFT_domain_dex;

        res._vf2 = decx::utils::cuda::__float_sub2(recv[0]._vf2, recv[1]._vf2);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
    }
}


template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R2_1st_cplxf_C2C_dense<true>(const float2* __restrict, float2* __restrict,
    const uint32_t, const uint32_t, const uint32_t, const uint64_t);

template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R2_1st_cplxf_C2C_dense<false>(const float2* __restrict, float2* __restrict,
    const uint32_t, const uint32_t, const uint32_t, const uint64_t);



template <bool _conj> __global__ void 
decx::dsp::fft::GPUK::cu_FFT2D_R2_end_cplxf_C2C_dense(const float2* __restrict src,
                                                      float2* __restrict dst,
                                                      const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                                      const uint32_t _pitchsrc_v2,
                                                      const uint32_t _pitchdst_v2)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 2;

    decx::utils::_cuda_vec64 recv[2];
    decx::utils::_cuda_vec64 res;

    uint32_t _FFT_domain_dex, warp_loc_id;

    decx::utils::_cuda_vec64 W;

    warp_loc_id = tidy % _kernel_info._store_pitch;

    if (tidy < _Bops_num && tidx < _pitchdst_v2)
    {
        _FFT_domain_dex = tidy;
#pragma unroll 2
        for (uint8_t i = 0; i < 2; ++i) {
            recv[i]._vf2 = src[_FFT_domain_dex * _pitchsrc_v2 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    W._cplxf32.construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[1]._cplxf32 = decx::dsp::fft::GPUK::_complex_mul_fp32(recv[1]._cplxf32, W._cplxf32);

    _FFT_domain_dex = (tidy / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + warp_loc_id;

    if (tidy < _Bops_num && tidx < _pitchdst_v2)
    {
        res._vf2 = decx::utils::cuda::__float_add2(recv[0]._vf2, recv[1]._vf2);
        if (_conj) { res._cplxf32 = decx::dsp::fft::GPUK::_complex_conjugate_fp32(res._cplxf32); }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf2;
        _FFT_domain_dex += _kernel_info._store_pitch;

        res._vf2 = decx::utils::cuda::__float_sub2(recv[0]._vf2, recv[1]._vf2);
        if (_conj) { res._cplxf32 = decx::dsp::fft::GPUK::_complex_conjugate_fp32(res._cplxf32); }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf2;
    }
}

template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R2_end_cplxf_C2C_dense<true>(const float2* __restrict, float2* __restrict,
    const decx::dsp::fft::FKI_4_2DK, const uint32_t, const uint32_t);

template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R2_end_cplxf_C2C_dense<false>(const float2* __restrict, float2* __restrict,
    const decx::dsp::fft::FKI_4_2DK, const uint32_t, const uint32_t);



__global__ void 
decx::dsp::fft::GPUK::cu_FFT2D_R2_end_cplxf_C2R_dense(const float2* __restrict src,
                                              float* __restrict dst,
                                              const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                              const uint32_t _pitchsrc_v2,
                                              const uint32_t _pitchdst_v2)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 2;

    decx::utils::_cuda_vec64 recv[2], tmp;
    float res;
    de::CPf W;

    uint32_t _FFT_domain_dex = tidy;
    uint32_t _warp_loc_id = tidy % _kernel_info._store_pitch;

    if (tidy < _Bops_num && tidx < _pitchsrc_v2)
    {
#pragma unroll 2
        for (uint8_t i = 0; i < 2; ++i) {
            recv[i]._vf2 = src[_FFT_domain_dex * _pitchsrc_v2 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    W.construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)_warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[1]._cplxf32 = decx::dsp::fft::GPUK::_complex_mul_fp32(recv[1]._cplxf32, W);

    _FFT_domain_dex = (tidy / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + _warp_loc_id;

    if (tidy < _Bops_num && tidx < _pitchdst_v2)
    {
        res = __fadd_rn(recv[0]._vf2.x, recv[1]._vf2.x);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res;
        _FFT_domain_dex += _kernel_info._store_pitch;

        res = __fsub_rn(recv[0]._vf2.x, recv[1]._vf2.x);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res;
    }
}
