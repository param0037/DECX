/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "FFT1D_1st_kernels_dense.cuh"


__global__ void 
decx::dsp::fft::GPUK::cu_FFT2D_R5_1st_cplxf_R2C_dense(const float* __restrict src, 
                                                      float2* __restrict dst,
                                                      const uint32_t _signal_len, 
                                                      const uint32_t _pitchsrc, 
                                                      const uint32_t _pitchdst)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _signal_len / 5;

    float recv[5];
    decx::utils::_cuda_vec64 res;

    uint32_t _FFT_domain_dex = tidy;

    if (tidy < _Bops_num && tidx < _pitchsrc)
    {
#pragma unroll 5
        for (uint8_t i = 0; i < 5; ++i) {
            recv[i] = src[_FFT_domain_dex * _pitchsrc + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    _FFT_domain_dex = tidy * 5;

    if (tidy < _Bops_num && tidx < _pitchdst)
    {
        res._vf2 = decx::utils::vec2_set1_fp32(0.f);

        res._cplxf32.real = __fadd_rn(__fadd_rn(recv[0], recv[1]),
                                                __fadd_rn(recv[2], recv[3]));
        res._cplxf32.real = __fadd_rn(res._cplxf32.real, recv[4]);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
        ++_FFT_domain_dex;

        res._cplxf32.real = __fmaf_rn(0.309017, recv[1], recv[0]);
        res._cplxf32.image = __fmul_rn(recv[1], 0.9510565);
        res._cplxf32.real = __fmaf_rn(-0.809017, recv[2], res._cplxf32.real);
        res._cplxf32.image = __fmaf_rn(0.5877853, recv[2], res._cplxf32.image);
        res._cplxf32.real = __fmaf_rn(-0.809017, recv[3], res._cplxf32.real);
        res._cplxf32.image = __fmaf_rn(-0.5877853, recv[3], res._cplxf32.image);
        res._cplxf32.real = __fmaf_rn(0.309017, recv[4], res._cplxf32.real);
        res._cplxf32.image = __fmaf_rn(-0.9510565, recv[4], res._cplxf32.image);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
        ++_FFT_domain_dex;

        res._cplxf32.real = __fmaf_rn(-0.809017, recv[1], recv[0]);
        res._cplxf32.image = __fmul_rn(recv[1], 0.5877853);
        res._cplxf32.real = __fmaf_rn(0.309017, recv[2], res._cplxf32.real);
        res._cplxf32.image = __fmaf_rn(-0.9510565, recv[2], res._cplxf32.image);
        res._cplxf32.real = __fmaf_rn(0.309017, recv[3], res._cplxf32.real);
        res._cplxf32.image = __fmaf_rn(0.9510565, recv[3], res._cplxf32.image);
        res._cplxf32.real = __fmaf_rn(-0.809017, recv[4], res._cplxf32.real);
        res._cplxf32.image = __fmaf_rn(-0.5877853, recv[4], res._cplxf32.image);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
        ++_FFT_domain_dex;


        res._cplxf32.real = __fmaf_rn(-0.809017, recv[1], recv[0]);
        res._cplxf32.image = __fmul_rn(recv[1], -0.5877853);
        res._cplxf32.real = __fmaf_rn(0.309017, recv[2], res._cplxf32.real);
        res._cplxf32.image = __fmaf_rn(0.9510565, recv[2], res._cplxf32.image);
        res._cplxf32.real = __fmaf_rn(0.309017, recv[3], res._cplxf32.real);
        res._cplxf32.image = __fmaf_rn(-0.9510565, recv[3], res._cplxf32.image);
        res._cplxf32.real = __fmaf_rn(-0.809017, recv[4], res._cplxf32.real);
        res._cplxf32.image = __fmaf_rn(0.5877853, recv[4], res._cplxf32.image);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
        ++_FFT_domain_dex;

        res._cplxf32.real = __fmaf_rn(0.309017, recv[1], recv[0]);
        res._cplxf32.image = __fmul_rn(recv[1], -0.9510565);
        res._cplxf32.real = __fmaf_rn(-0.809017, recv[2], res._cplxf32.real);
        res._cplxf32.image = __fmaf_rn(-0.5877853, recv[2], res._cplxf32.image);
        res._cplxf32.real = __fmaf_rn(-0.809017, recv[3], res._cplxf32.real);
        res._cplxf32.image = __fmaf_rn(0.5877853, recv[3], res._cplxf32.image);
        res._cplxf32.real = __fmaf_rn(0.309017, recv[4], res._cplxf32.real);
        res._cplxf32.image = __fmaf_rn(0.9510565, recv[4], res._cplxf32.image);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
    }
}


template <bool _div> __global__ void 
decx::dsp::fft::GPUK::cu_FFT2D_R5_1st_cplxf_C2C_dense(const float2* __restrict src,
                                                      float2* __restrict dst,
                                                      const uint32_t _signal_len,
                                                      const uint32_t _pitchsrc,
                                                      const uint32_t _pitchdst,
                                                      const uint64_t _div_length)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const float _numer = __ull2float_rn(_div_length ? _div_length : _signal_len);
    const uint32_t _Bops_num = _signal_len / 5;

    decx::utils::_cuda_vec64 recv[5];
    decx::utils::_cuda_vec64 res;

    
    uint32_t _FFT_domain_dex = tidy;

    if (tidy < _Bops_num && tidx < _pitchsrc)
    {
#pragma unroll 5
        for (uint8_t i = 0; i < 5; ++i) {
            recv[i]._vf2 = src[_FFT_domain_dex * _pitchsrc + tidx];
            if (_div) { recv[i]._vf2 = decx::utils::cuda::__float_div2_1(recv[i]._vf2, _numer); }
            _FFT_domain_dex += _Bops_num;
        }
    }

    _FFT_domain_dex = tidy * 5;

    if (tidy < _Bops_num && tidx < _pitchdst)
    {
        res._cplxf32.real = __fadd_rn(__fadd_rn(recv[0]._cplxf32.real, recv[1]._cplxf32.real), 
                                      __fadd_rn(recv[2]._cplxf32.real, recv[3]._cplxf32.real));
        res._cplxf32.real = __fadd_rn(res._cplxf32.real, recv[4]._cplxf32.real);
        res._cplxf32.image = __fadd_rn(__fadd_rn(recv[0]._cplxf32.image, recv[1]._cplxf32.image), 
                                       __fadd_rn(recv[2]._cplxf32.image, recv[3]._cplxf32.image));
        res._cplxf32.image = __fadd_rn(res._cplxf32.image, recv[4]._cplxf32.image);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
        ++_FFT_domain_dex;

        res._vf2 = recv[0]._vf2;
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[1]._cplxf32, de::CPf(0.309017, 0.9510565), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[2]._cplxf32, de::CPf(-0.809017, 0.5877853), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[3]._cplxf32, de::CPf(-0.809017, -0.5877853), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[4]._cplxf32, de::CPf(0.309017, -0.9510565), res._cplxf32);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
        ++_FFT_domain_dex;

        res._vf2 = recv[0]._vf2;
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[1]._cplxf32, de::CPf(-0.809017, 0.5877853), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[2]._cplxf32, de::CPf(0.309017, -0.9510565), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[3]._cplxf32, de::CPf(0.309017, 0.9510565), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[4]._cplxf32, de::CPf(-0.809017, -0.5877853), res._cplxf32);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
        ++_FFT_domain_dex;

        res._vf2 = recv[0]._vf2;
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[1]._cplxf32, de::CPf(-0.809017, -0.5877853), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[2]._cplxf32, de::CPf(0.309017, 0.9510565), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[3]._cplxf32, de::CPf(0.309017, -0.9510565), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[4]._cplxf32, de::CPf(-0.809017, 0.5877853), res._cplxf32);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
        ++_FFT_domain_dex;

        res._vf2 = recv[0]._vf2;
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[1]._cplxf32, de::CPf(0.309017, -0.9510565), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[2]._cplxf32, de::CPf(-0.809017, -0.5877853), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[3]._cplxf32, de::CPf(-0.809017, 0.5877853), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[4]._cplxf32, de::CPf(0.309017, 0.9510565), res._cplxf32);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
    }
}

template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R5_1st_cplxf_C2C_dense<true>(const float2* __restrict, float2* __restrict,
    const uint32_t, const uint32_t, const uint32_t, const uint64_t);

template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R5_1st_cplxf_C2C_dense<false>(const float2* __restrict, float2* __restrict,
    const uint32_t, const uint32_t, const uint32_t, const uint64_t);



template <bool _conj> __global__ void 
decx::dsp::fft::GPUK::cu_FFT2D_R5_end_cplxf_C2C_dense(const float2* __restrict src,
                                           float2* __restrict dst,
                                           const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                           const uint32_t _pitchsrc,
                                           const uint32_t _pitchdst)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 5;

    decx::utils::_cuda_vec64 recv[5];
    decx::utils::_cuda_vec64 res;
    de::CPf W;

    uint32_t _FFT_domain_dex = tidy;
    const uint32_t _warp_loc_id = tidy % _kernel_info._store_pitch;

    if (tidy < _Bops_num && tidx < _pitchsrc)
    {
#pragma unroll 5
        for (uint8_t i = 0; i < 5; ++i) {
            recv[i]._vf2 = src[_FFT_domain_dex * _pitchsrc + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    const float _frac = __fdividef(_warp_loc_id, _kernel_info._warp_proc_len);
    W.dev_construct_with_phase(__fmul_rn(Two_Pi, _frac));
    recv[1]._cplxf32 = decx::dsp::fft::GPUK::_complex_mul_fp32(recv[1]._cplxf32, W);

    W.dev_construct_with_phase(__fmul_rn(Four_Pi, _frac));
    recv[2]._cplxf32 = decx::dsp::fft::GPUK::_complex_mul_fp32(recv[2]._cplxf32, W);

    W.dev_construct_with_phase(__fmul_rn(Six_Pi, _frac));
    recv[3]._cplxf32 = decx::dsp::fft::GPUK::_complex_mul_fp32(recv[3]._cplxf32, W);

    W.dev_construct_with_phase(__fmul_rn(Eight_Pi, _frac));
    recv[4]._cplxf32 = decx::dsp::fft::GPUK::_complex_mul_fp32(recv[4]._cplxf32, W);

    _FFT_domain_dex = (tidy / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + _warp_loc_id;

    if (tidy < _Bops_num && tidx < _pitchdst) 
    {
        res._cplxf32.real = __fadd_rn(__fadd_rn(recv[0]._cplxf32.real, recv[1]._cplxf32.real), 
                                      __fadd_rn(recv[2]._cplxf32.real, recv[3]._cplxf32.real));
        res._cplxf32.real = __fadd_rn(res._cplxf32.real, recv[4]._cplxf32.real);
        res._cplxf32.image = __fadd_rn(__fadd_rn(recv[0]._cplxf32.image, recv[1]._cplxf32.image), 
                                       __fadd_rn(recv[2]._cplxf32.image, recv[3]._cplxf32.image));
        res._cplxf32.image = __fadd_rn(res._cplxf32.image, recv[4]._cplxf32.image);
        if (_conj) { res._cplxf32 = decx::dsp::fft::GPUK::_complex_conjugate_fp32(res._cplxf32); }
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
        _FFT_domain_dex += _kernel_info._store_pitch;

        res._vf2 = recv[0]._vf2;
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[1]._cplxf32, de::CPf(0.309017, 0.9510565), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[2]._cplxf32, de::CPf(-0.809017, 0.5877853), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[3]._cplxf32, de::CPf(-0.809017, -0.5877853), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[4]._cplxf32, de::CPf(0.309017, -0.9510565), res._cplxf32);
        if (_conj) { res._cplxf32 = decx::dsp::fft::GPUK::_complex_conjugate_fp32(res._cplxf32); }
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
        _FFT_domain_dex += _kernel_info._store_pitch;

        res._vf2 = recv[0]._vf2;
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[1]._cplxf32, de::CPf(-0.809017, 0.5877853), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[2]._cplxf32, de::CPf(0.309017, -0.9510565), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[3]._cplxf32, de::CPf(0.309017, 0.9510565), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[4]._cplxf32, de::CPf(-0.809017, -0.5877853), res._cplxf32);
        if (_conj) { res._cplxf32 = decx::dsp::fft::GPUK::_complex_conjugate_fp32(res._cplxf32); }
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
        _FFT_domain_dex += _kernel_info._store_pitch;

        res._vf2 = recv[0]._vf2;
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[1]._cplxf32, de::CPf(-0.809017, -0.5877853), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[2]._cplxf32, de::CPf(0.309017, 0.9510565), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[3]._cplxf32, de::CPf(0.309017, -0.9510565), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[4]._cplxf32, de::CPf(-0.809017, 0.5877853), res._cplxf32);
        if (_conj) { res._cplxf32 = decx::dsp::fft::GPUK::_complex_conjugate_fp32(res._cplxf32); }
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
        _FFT_domain_dex += _kernel_info._store_pitch;

        res._vf2 = recv[0]._vf2;
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[1]._cplxf32, de::CPf(0.309017, -0.9510565), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[2]._cplxf32, de::CPf(-0.809017, -0.5877853), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[3]._cplxf32, de::CPf(-0.809017, 0.5877853), res._cplxf32);
        res._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[4]._cplxf32, de::CPf(0.309017, 0.9510565), res._cplxf32);
        if (_conj) { res._cplxf32 = decx::dsp::fft::GPUK::_complex_conjugate_fp32(res._cplxf32); }
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
    }
}

template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R5_end_cplxf_C2C_dense<true>(const float2* __restrict, float2* __restrict,
    const decx::dsp::fft::FKI_4_2DK, const uint32_t, const uint32_t);

template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R5_end_cplxf_C2C_dense<false>(const float2* __restrict, float2* __restrict,
    const decx::dsp::fft::FKI_4_2DK, const uint32_t, const uint32_t);



__global__ void 
decx::dsp::fft::GPUK::cu_FFT2D_R5_end_cplxf_C2R_dense(const float2* __restrict src,
                                              float* __restrict dst,
                                              const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                              const uint32_t _pitchsrc,
                                              const uint32_t _pitchdst)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 5;

    decx::utils::_cuda_vec64 recv[5], tmp;
    float res;
    de::CPf W;

    uint32_t _FFT_domain_dex = tidy;
    uint32_t _warp_loc_id = tidy % _kernel_info._store_pitch;

    if (tidy < _Bops_num && tidx < _pitchsrc)
    {
#pragma unroll 5
        for (uint8_t i = 0; i < 5; ++i) {
            recv[i]._vf2 = src[_FFT_domain_dex * _pitchsrc + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    const float _frac = __fdividef(_warp_loc_id, _kernel_info._warp_proc_len);
    W.dev_construct_with_phase(__fmul_rn(Two_Pi, _frac));
    recv[1]._cplxf32 = decx::dsp::fft::GPUK::_complex_mul_fp32(recv[1]._cplxf32, W);

    W.dev_construct_with_phase(__fmul_rn(Four_Pi, _frac));
    recv[2]._cplxf32 = decx::dsp::fft::GPUK::_complex_mul_fp32(recv[2]._cplxf32, W);

    W.dev_construct_with_phase(__fmul_rn(Six_Pi, _frac));
    recv[3]._cplxf32 = decx::dsp::fft::GPUK::_complex_mul_fp32(recv[3]._cplxf32, W);

    W.dev_construct_with_phase(__fmul_rn(Eight_Pi, _frac));
    recv[4]._cplxf32 = decx::dsp::fft::GPUK::_complex_mul_fp32(recv[4]._cplxf32, W);

    _FFT_domain_dex = (tidy / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + _warp_loc_id;

    if (tidy < _Bops_num && tidx < _pitchdst)
    {
        res = __fadd_rn(__fadd_rn(recv[0]._cplxf32.real, recv[1]._cplxf32.real), 
                        __fadd_rn(recv[2]._cplxf32.real, recv[3]._cplxf32.real));
        res = __fadd_rn(res, recv[4]._cplxf32.real);

        dst[_FFT_domain_dex * _pitchdst + tidx] = res;
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vf2 = recv[0]._vf2;
        tmp._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[1]._cplxf32, de::CPf(0.309017, 0.9510565), tmp._cplxf32);
        tmp._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[2]._cplxf32, de::CPf(-0.809017, 0.5877853), tmp._cplxf32);
        tmp._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[3]._cplxf32, de::CPf(-0.809017, -0.5877853), tmp._cplxf32);
        res = __fsub_rn(__fmaf_rn(recv[4]._vf2.x, 0.309017, tmp._vf2.x), __fmul_rn(recv[4]._vf2.y, -0.9510565));

        dst[_FFT_domain_dex * _pitchdst + tidx] = res;
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vf2 = recv[0]._vf2;
        tmp._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[1]._cplxf32, de::CPf(-0.809017, 0.5877853), tmp._cplxf32);
        tmp._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[2]._cplxf32, de::CPf(0.309017, -0.9510565), tmp._cplxf32);
        tmp._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[3]._cplxf32, de::CPf(0.309017, 0.9510565), tmp._cplxf32);
        res = __fsub_rn(__fmaf_rn(recv[4]._vf2.x, -0.809017, tmp._vf2.x), __fmul_rn(recv[4]._vf2.y, -0.5877853));

        dst[_FFT_domain_dex * _pitchdst + tidx] = res;
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vf2 = recv[0]._vf2;
        tmp._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[1]._cplxf32, de::CPf(-0.809017, -0.5877853), tmp._cplxf32);
        tmp._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[2]._cplxf32, de::CPf(0.309017, 0.9510565), tmp._cplxf32);
        tmp._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[3]._cplxf32, de::CPf(0.309017, -0.9510565), tmp._cplxf32);
        res = __fsub_rn(__fmaf_rn(recv[4]._vf2.x, -0.809017, tmp._vf2.x), __fmul_rn(recv[4]._vf2.y, 0.5877853));

        dst[_FFT_domain_dex * _pitchdst + tidx] = res;
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vf2 = recv[0]._vf2;
        tmp._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[1]._cplxf32, de::CPf(0.309017, -0.9510565), tmp._cplxf32);
        tmp._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[2]._cplxf32, de::CPf(-0.809017, -0.5877853), tmp._cplxf32);
        tmp._cplxf32 = decx::dsp::fft::GPUK::_complex_fma_fp32(recv[3]._cplxf32, de::CPf(-0.809017, 0.5877853), tmp._cplxf32);
        res = __fsub_rn(__fmaf_rn(recv[4]._vf2.x, 0.309017, tmp._vf2.x), __fmul_rn(recv[4]._vf2.y, 0.9510565));

        dst[_FFT_domain_dex * _pitchdst + tidx] = res;
    }
}