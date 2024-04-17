/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "../FFT2D_kernels.cuh"


// [32 * 2, 8] = [64, 8]
__global__ void 
decx::dsp::fft::GPUK::cu_FFT2_R5_1st_R2C_cplxf(const float2* __restrict src,
                                               float4* __restrict dst,
                                               const uint32_t _signal_len,
                                               const uint32_t _pitchsrc_v2,
                                               const uint32_t _pitchdst_v2)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _signal_len / 5;

    decx::utils::_cuda_vec64 recv[5];
    decx::utils::_cuda_vec128 res;

    uint32_t _FFT_domain_dex = tidy;

    if (tidy < _Bops_num && tidx < _pitchsrc_v2)
    {
#pragma unroll 5
        for (uint8_t i = 0; i < 5; ++i) {
            recv[i]._vf2 = src[_FFT_domain_dex * _pitchsrc_v2 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    _FFT_domain_dex = tidy * 5;

    if (tidy < _Bops_num && tidx < _pitchdst_v2)
    {
        res._vf = decx::utils::vec4_set1_fp32(0.f);

        for (uint8_t i = 0; i < 2; ++i){
            res._arrcplxf2[i].real = __fadd_rn(__fadd_rn(recv[0]._arrf[i], recv[1]._arrf[i]),
                                                  __fadd_rn(recv[2]._arrf[i], recv[3]._arrf[i]));
            res._arrcplxf2[i].real = __fadd_rn(res._arrcplxf2[i].real, recv[4]._arrf[i]);
        }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        for (uint8_t i = 0; i < 2; ++i) {
            res._arrcplxf2[i].real = __fmaf_rn(0.309017, recv[1]._arrf[i], recv[0]._arrf[i]);
            res._arrcplxf2[i].image = __fmul_rn(recv[1]._arrf[i], 0.9510565);
            res._arrcplxf2[i].real = __fmaf_rn(-0.809017, recv[2]._arrf[i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(0.5877853, recv[2]._arrf[i], res._arrcplxf2[i].image);
            res._arrcplxf2[i].real = __fmaf_rn(-0.809017, recv[3]._arrf[i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(-0.5877853, recv[3]._arrf[i], res._arrcplxf2[i].image);
            res._arrcplxf2[i].real = __fmaf_rn(0.309017, recv[4]._arrf[i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(-0.9510565, recv[4]._arrf[i], res._arrcplxf2[i].image);
        }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        for (uint8_t i = 0; i < 2; ++i) {
            res._arrcplxf2[i].real = __fmaf_rn(-0.809017, recv[1]._arrf[i], recv[0]._arrf[i]);
            res._arrcplxf2[i].image = __fmul_rn(recv[1]._arrf[i], 0.5877853);
            res._arrcplxf2[i].real = __fmaf_rn(0.309017, recv[2]._arrf[i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(-0.9510565, recv[2]._arrf[i], res._arrcplxf2[i].image);
            res._arrcplxf2[i].real = __fmaf_rn(0.309017, recv[3]._arrf[i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(0.9510565, recv[3]._arrf[i], res._arrcplxf2[i].image);
            res._arrcplxf2[i].real = __fmaf_rn(-0.809017, recv[4]._arrf[i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(-0.5877853, recv[4]._arrf[i], res._arrcplxf2[i].image);
        }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        for (uint8_t i = 0; i < 2; ++i) {
            res._arrcplxf2[i].real = __fmaf_rn(-0.809017, recv[1]._arrf[i], recv[0]._arrf[i]);
            res._arrcplxf2[i].image = __fmul_rn(recv[1]._arrf[i], -0.5877853);
            res._arrcplxf2[i].real = __fmaf_rn(0.309017, recv[2]._arrf[i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(0.9510565, recv[2]._arrf[i], res._arrcplxf2[i].image);
            res._arrcplxf2[i].real = __fmaf_rn(0.309017, recv[3]._arrf[i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(-0.9510565, recv[3]._arrf[i], res._arrcplxf2[i].image);
            res._arrcplxf2[i].real = __fmaf_rn(-0.809017, recv[4]._arrf[i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(0.5877853, recv[4]._arrf[i], res._arrcplxf2[i].image);
        }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        for (uint8_t i = 0; i < 2; ++i) {
            res._arrcplxf2[i].real = __fmaf_rn(0.309017, recv[1]._arrf[i], recv[0]._arrf[i]);
            res._arrcplxf2[i].image = __fmul_rn(recv[1]._arrf[i], -0.9510565);
            res._arrcplxf2[i].real = __fmaf_rn(-0.809017, recv[2]._arrf[i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(-0.5877853, recv[2]._arrf[i], res._arrcplxf2[i].image);
            res._arrcplxf2[i].real = __fmaf_rn(-0.809017, recv[3]._arrf[i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(0.5877853, recv[3]._arrf[i], res._arrcplxf2[i].image);
            res._arrcplxf2[i].real = __fmaf_rn(0.309017, recv[4]._arrf[i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(0.9510565, recv[4]._arrf[i], res._arrcplxf2[i].image);
        }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
    }
}




// [32 * 2, 8] = [64, 8]
__global__ void 
decx::dsp::fft::GPUK::cu_FFT2_R5_1st_R2C_uc8_cplxf(const ushort* __restrict src,
                                               float4* __restrict dst,
                                               const uint32_t _signal_len,
                                               const uint32_t _pitchsrc_v2,
                                               const uint32_t _pitchdst_v2)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _signal_len / 5;

    uchar recv[5][2];
    decx::utils::_cuda_vec128 res;

    uint32_t _FFT_domain_dex = tidy;

    if (tidy < _Bops_num && tidx < _pitchsrc_v2)
    {
#pragma unroll 5
        for (uint8_t i = 0; i < 5; ++i) {
            *((ushort*)recv[i]) = src[_FFT_domain_dex * _pitchsrc_v2 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    _FFT_domain_dex = tidy * 5;

    if (tidy < _Bops_num && tidx < _pitchdst_v2)
    {
        res._vf = decx::utils::vec4_set1_fp32(0.f);

        for (uint8_t i = 0; i < 2; ++i){
            res._arrcplxf2[i].real = __fadd_rn(__fadd_rn(recv[0][i], recv[1][i]),
                                                  __fadd_rn(recv[2][i], recv[3][i]));
            res._arrcplxf2[i].real = __fadd_rn(res._arrcplxf2[i].real, recv[4][i]);
        }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        for (uint8_t i = 0; i < 2; ++i) {
            res._arrcplxf2[i].real = __fmaf_rn(0.309017, recv[1][i], recv[0][i]);
            res._arrcplxf2[i].image = __fmul_rn(recv[1][i], 0.9510565);
            res._arrcplxf2[i].real = __fmaf_rn(-0.809017, recv[2][i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(0.5877853, recv[2][i], res._arrcplxf2[i].image);
            res._arrcplxf2[i].real = __fmaf_rn(-0.809017, recv[3][i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(-0.5877853, recv[3][i], res._arrcplxf2[i].image);
            res._arrcplxf2[i].real = __fmaf_rn(0.309017, recv[4][i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(-0.9510565, recv[4][i], res._arrcplxf2[i].image);
        }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        for (uint8_t i = 0; i < 2; ++i) {
            res._arrcplxf2[i].real = __fmaf_rn(-0.809017, recv[1][i], recv[0][i]);
            res._arrcplxf2[i].image = __fmul_rn(recv[1][i], 0.5877853);
            res._arrcplxf2[i].real = __fmaf_rn(0.309017, recv[2][i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(-0.9510565, recv[2][i], res._arrcplxf2[i].image);
            res._arrcplxf2[i].real = __fmaf_rn(0.309017, recv[3][i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(0.9510565, recv[3][i], res._arrcplxf2[i].image);
            res._arrcplxf2[i].real = __fmaf_rn(-0.809017, recv[4][i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(-0.5877853, recv[4][i], res._arrcplxf2[i].image);
        }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        for (uint8_t i = 0; i < 2; ++i) {
            res._arrcplxf2[i].real = __fmaf_rn(-0.809017, recv[1][i], recv[0][i]);
            res._arrcplxf2[i].image = __fmul_rn(recv[1][i], -0.5877853);
            res._arrcplxf2[i].real = __fmaf_rn(0.309017, recv[2][i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(0.9510565, recv[2][i], res._arrcplxf2[i].image);
            res._arrcplxf2[i].real = __fmaf_rn(0.309017, recv[3][i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(-0.9510565, recv[3][i], res._arrcplxf2[i].image);
            res._arrcplxf2[i].real = __fmaf_rn(-0.809017, recv[4][i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(0.5877853, recv[4][i], res._arrcplxf2[i].image);
        }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        for (uint8_t i = 0; i < 2; ++i) {
            res._arrcplxf2[i].real = __fmaf_rn(0.309017, recv[1][i], recv[0][i]);
            res._arrcplxf2[i].image = __fmul_rn(recv[1][i], -0.9510565);
            res._arrcplxf2[i].real = __fmaf_rn(-0.809017, recv[2][i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(-0.5877853, recv[2][i], res._arrcplxf2[i].image);
            res._arrcplxf2[i].real = __fmaf_rn(-0.809017, recv[3][i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(0.5877853, recv[3][i], res._arrcplxf2[i].image);
            res._arrcplxf2[i].real = __fmaf_rn(0.309017, recv[4][i], res._arrcplxf2[i].real);
            res._arrcplxf2[i].image = __fmaf_rn(0.9510565, recv[4][i], res._arrcplxf2[i].image);
        }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
    }
}





template <bool _div> __global__ void 
decx::dsp::fft::GPUK::cu_FFT2_R5_1st_C2C_cplxf(const float4* __restrict src,
                                               float4* __restrict dst,
                                               const uint32_t _signal_len,
                                               const uint32_t _pitchsrc_v2,
                                               const uint32_t _pitchdst_v2,
                                               const uint64_t _div_length)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _signal_len / 5;

    decx::utils::_cuda_vec128 recv[5];
    decx::utils::_cuda_vec128 res;

    const float _numer = __ull2float_rn(_div_length ? _div_length : _signal_len);
    uint32_t _FFT_domain_dex = tidy;

    if (tidy < _Bops_num && tidx < _pitchsrc_v2)
    {
#pragma unroll 5
        for (uint8_t i = 0; i < 5; ++i) {
            recv[i]._vf = src[_FFT_domain_dex * _pitchsrc_v2 + tidx];
            if (_div) { recv[i]._vf = decx::utils::cuda::__float_div4_1(recv[i]._vf, _numer); }
            _FFT_domain_dex += _Bops_num;
        }
    }

    _FFT_domain_dex = tidy * 5;

    if (tidy < _Bops_num && tidx < _pitchdst_v2)
    {
        for (uint8_t i = 0; i < 2; ++i) {
            res._arrcplxf2[i].real = __fadd_rn(__fadd_rn(recv[0]._arrcplxf2[i].real, recv[1]._arrcplxf2[i].real) + 
                recv[2]._arrcplxf2[i].real, __fadd_rn(recv[3]._arrcplxf2[i].real, recv[4]._arrcplxf2[i].real));
            res._arrcplxf2[i].image = __fadd_rn(__fadd_rn(recv[0]._arrcplxf2[i].image, recv[1]._arrcplxf2[i].image) +
                recv[2]._arrcplxf2[i].image, __fadd_rn(recv[3]._arrcplxf2[i].image, recv[4]._arrcplxf2[i].image));
        }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        res._vf = recv[0]._vf;
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[1]._vf, de::CPf(0.309017, 0.9510565),  res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[2]._vf, de::CPf(-0.809017, 0.5877853), res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[3]._vf, de::CPf(-0.809017, -0.5877853),res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[4]._vf, de::CPf(0.309017, -0.9510565), res._vf);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        res._vf = recv[0]._vf;
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[1]._vf, de::CPf(-0.809017, 0.5877853), res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[2]._vf, de::CPf(0.309017, -0.9510565), res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[3]._vf, de::CPf(0.309017, 0.9510565),  res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[4]._vf, de::CPf(-0.809017, -0.5877853),res._vf);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        res._vf = recv[0]._vf;
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[1]._vf, de::CPf(-0.809017, -0.5877853), res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[2]._vf, de::CPf(0.309017, 0.9510565),   res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[3]._vf, de::CPf(0.309017, -0.9510565),  res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[4]._vf, de::CPf(-0.809017, 0.5877853),  res._vf);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        res._vf = recv[0]._vf;
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[1]._vf, de::CPf(0.309017, -0.9510565),  res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[2]._vf, de::CPf(-0.809017, -0.5877853), res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[3]._vf, de::CPf(-0.809017, 0.5877853),  res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[4]._vf, de::CPf(0.309017, 0.9510565),   res._vf);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
    }
}


template __global__ void decx::dsp::fft::GPUK::cu_FFT2_R5_1st_C2C_cplxf<true>(const float4* __restrict, float4* __restrict,
    const uint32_t, const uint32_t, const uint32_t, const uint64_t);

template __global__ void decx::dsp::fft::GPUK::cu_FFT2_R5_1st_C2C_cplxf<false>(const float4* __restrict, float4* __restrict,
    const uint32_t, const uint32_t, const uint32_t, const uint64_t);



template <bool _conj> __global__ void 
decx::dsp::fft::GPUK::cu_FFT2_R5_C2C_cplxf(const float4* __restrict src,
                                           float4* __restrict dst,
                                           const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                           const uint32_t _pitchsrc_v2,
                                           const uint32_t _pitchdst_v2)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 5;

    decx::utils::_cuda_vec128 recv[5];
    decx::utils::_cuda_vec128 res;
    de::CPf W;

    uint32_t _FFT_domain_dex = tidy;
    const uint32_t _warp_loc_id = tidy % _kernel_info._store_pitch;

    if (tidy < _Bops_num && tidx < _pitchsrc_v2)
    {
#pragma unroll 5
        for (uint8_t i = 0; i < 5; ++i) {
            recv[i]._vf = src[_FFT_domain_dex * _pitchsrc_v2 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    const float _frac = __fdividef(_warp_loc_id, _kernel_info._warp_proc_len);
    W.construct_with_phase(__fmul_rn(Two_Pi, _frac));
    recv[1]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[1]._vf, W);

    W.construct_with_phase(__fmul_rn(Four_Pi, _frac));
    recv[2]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[2]._vf, W);

    W.construct_with_phase(__fmul_rn(Six_Pi, _frac));
    recv[3]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[3]._vf, W);

    W.construct_with_phase(__fmul_rn(Eight_Pi, _frac));
    recv[4]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[4]._vf, W);

    _FFT_domain_dex = (tidy / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + _warp_loc_id;

    if (tidy < _Bops_num && tidx < _pitchdst_v2) 
    {
        for (uint8_t i = 0; i < 2; ++i) {
            /*res._arrcplxf2[i].real = __fadd_rn(__fadd_rn(recv[0]._arrcplxf2[i].real, recv[1]._arrcplxf2[i].real) +
                recv[2]._arrcplxf2[i].real, __fadd_rn(recv[3]._arrcplxf2[i].real, recv[4]._arrcplxf2[i].real));
            res._arrcplxf2[i].image = __fadd_rn(__fadd_rn(recv[0]._arrcplxf2[i].image, recv[1]._arrcplxf2[i].image) +
                recv[2]._arrcplxf2[i].image, __fadd_rn(recv[3]._arrcplxf2[i].image, recv[4]._arrcplxf2[i].image));*/

            res._arrcplxf2[i].real = __fadd_rn(recv[0]._arrcplxf2[i].real, recv[1]._arrcplxf2[i].real);
            res._arrcplxf2[i].real = __fadd_rn(res._arrcplxf2[i].real, recv[2]._arrcplxf2[i].real);
            res._arrcplxf2[i].real = __fadd_rn(res._arrcplxf2[i].real, recv[3]._arrcplxf2[i].real);
            res._arrcplxf2[i].real = __fadd_rn(res._arrcplxf2[i].real, recv[4]._arrcplxf2[i].real);

            res._arrcplxf2[i].image = __fadd_rn(recv[0]._arrcplxf2[i].image, recv[1]._arrcplxf2[i].image);
            res._arrcplxf2[i].image = __fadd_rn(res._arrcplxf2[i].image, recv[2]._arrcplxf2[i].image);
            res._arrcplxf2[i].image = __fadd_rn(res._arrcplxf2[i].image, recv[3]._arrcplxf2[i].image);
            res._arrcplxf2[i].image = __fadd_rn(res._arrcplxf2[i].image, recv[4]._arrcplxf2[i].image);
        }
        if (_conj) { res = decx::dsp::fft::GPUK::_complex4_conjugate_fp32(res); }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        _FFT_domain_dex += _kernel_info._store_pitch;

        res._vf = recv[0]._vf;
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[1]._vf, de::CPf(0.309017, 0.9510565), res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[2]._vf, de::CPf(-0.809017, 0.5877853), res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[3]._vf, de::CPf(-0.809017, -0.5877853), res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[4]._vf, de::CPf(0.309017, -0.9510565), res._vf);
        if (_conj) { res = decx::dsp::fft::GPUK::_complex4_conjugate_fp32(res); }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        _FFT_domain_dex += _kernel_info._store_pitch;

        res._vf = recv[0]._vf;
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[1]._vf, de::CPf(-0.809017, 0.5877853), res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[2]._vf, de::CPf(0.309017, -0.9510565), res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[3]._vf, de::CPf(0.309017, 0.9510565), res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[4]._vf, de::CPf(-0.809017, -0.5877853), res._vf);
        if (_conj) { res = decx::dsp::fft::GPUK::_complex4_conjugate_fp32(res); }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        _FFT_domain_dex += _kernel_info._store_pitch;

        res._vf = recv[0]._vf;
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[1]._vf, de::CPf(-0.809017, -0.5877853), res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[2]._vf, de::CPf(0.309017, 0.9510565), res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[3]._vf, de::CPf(0.309017, -0.9510565), res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[4]._vf, de::CPf(-0.809017, 0.5877853), res._vf);
        if (_conj) { res = decx::dsp::fft::GPUK::_complex4_conjugate_fp32(res); }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        _FFT_domain_dex += _kernel_info._store_pitch;

        res._vf = recv[0]._vf;
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[1]._vf, de::CPf(0.309017, -0.9510565), res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[2]._vf, de::CPf(-0.809017, -0.5877853), res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[3]._vf, de::CPf(-0.809017, 0.5877853), res._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[4]._vf, de::CPf(0.309017, 0.9510565), res._vf);
        if (_conj) { res = decx::dsp::fft::GPUK::_complex4_conjugate_fp32(res); }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
    }
}

template __global__ void decx::dsp::fft::GPUK::cu_FFT2_R5_C2C_cplxf<true>(const float4* __restrict, float4* __restrict,
    const decx::dsp::fft::FKI_4_2DK, const uint32_t, const uint32_t);

template __global__ void decx::dsp::fft::GPUK::cu_FFT2_R5_C2C_cplxf<false>(const float4* __restrict, float4* __restrict,
    const decx::dsp::fft::FKI_4_2DK, const uint32_t, const uint32_t);



__global__ void 
decx::dsp::fft::GPUK::cu_FFT2_R5_C2R_cplxf_u8(const float4* __restrict src,
                                              uchar2* __restrict dst,
                                              const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                              const uint32_t _pitchsrc_v2,
                                              const uint32_t _pitchdst_v2)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 5;

    decx::utils::_cuda_vec128 recv[5], tmp;
    decx::utils::_cuda_vec64 res;
    de::CPf W;

    uint32_t _FFT_domain_dex = tidy;
    uint32_t _warp_loc_id = tidy % _kernel_info._store_pitch;

    if (tidy < _Bops_num && tidx < _pitchsrc_v2)
    {
#pragma unroll 5
        for (uint8_t i = 0; i < 5; ++i) {
            recv[i]._vf = src[_FFT_domain_dex * _pitchsrc_v2 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    const float _frac = __fdividef(_warp_loc_id, _kernel_info._warp_proc_len);
    W.construct_with_phase(__fmul_rn(Two_Pi, _frac));
    recv[1]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[1]._vf, W);

    W.construct_with_phase(__fmul_rn(Four_Pi, _frac));
    recv[2]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[2]._vf, W);

    W.construct_with_phase(__fmul_rn(Six_Pi, _frac));
    recv[3]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[3]._vf, W);

    W.construct_with_phase(__fmul_rn(Eight_Pi, _frac));
    recv[4]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[4]._vf, W);

    _FFT_domain_dex = (tidy / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + _warp_loc_id;

    if (tidy < _Bops_num && tidx < _pitchdst_v2)
    {
        for (uint8_t i = 0; i < 2; ++i) {
            res._arrf[i] = __fadd_rn(__fadd_rn(recv[0]._arrcplxf2[i].real, recv[1]._arrcplxf2[i].real) +
                recv[2]._arrcplxf2[i].real, __fadd_rn(recv[3]._arrcplxf2[i].real, recv[4]._arrcplxf2[i].real));
        }

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = make_uchar2(res._arrf[0], res._arrf[1]);
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vf = recv[0]._vf;
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[1]._vf, de::CPf(0.309017, 0.9510565), tmp._vf);
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[2]._vf, de::CPf(-0.809017, 0.5877853), tmp._vf);
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[3]._vf, de::CPf(-0.809017, -0.5877853), tmp._vf);
        res._vf2 = make_float2(__fsub_rn(fmaf(recv[4]._vf.x, 0.309017, tmp._vf.x), __fmul_rn(recv[4]._vf.y, -0.9510565)),
            __fsub_rn(fmaf(recv[4]._vf.z, 0.309017, tmp._vf.z), __fmul_rn(recv[4]._vf.w, -0.9510565)));
        
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = make_uchar2(res._arrf[0], res._arrf[1]);
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vf = recv[0]._vf;
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[1]._vf, de::CPf(-0.809017, 0.5877853), tmp._vf);
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[2]._vf, de::CPf(0.309017, -0.9510565), tmp._vf);
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[3]._vf, de::CPf(0.309017, 0.9510565), tmp._vf);
        res._vf2 = make_float2(__fsub_rn(fmaf(recv[4]._vf.x, -0.809017, tmp._vf.x), __fmul_rn(recv[4]._vf.y, -0.5877853)),
            __fsub_rn(fmaf(recv[4]._vf.z, -0.809017, tmp._vf.z), __fmul_rn(recv[4]._vf.w, -0.5877853)));

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = make_uchar2(res._arrf[0], res._arrf[1]);
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vf = recv[0]._vf;
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[1]._vf, de::CPf(-0.809017, -0.5877853), tmp._vf);
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[2]._vf, de::CPf(0.309017, 0.9510565), tmp._vf);
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[3]._vf, de::CPf(0.309017, -0.9510565), tmp._vf);
        res._vf2 = make_float2(__fsub_rn(fmaf(recv[4]._vf.x, -0.809017, tmp._vf.x), __fmul_rn(recv[4]._vf.y, 0.5877853)),
            __fsub_rn(fmaf(recv[4]._vf.z, -0.809017, tmp._vf.z), __fmul_rn(recv[4]._vf.w, 0.5877853)));

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = make_uchar2(res._arrf[0], res._arrf[1]);
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vf = recv[0]._vf;
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[1]._vf, de::CPf(0.309017, -0.9510565), tmp._vf);
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[2]._vf, de::CPf(-0.809017, -0.5877853), tmp._vf);
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[3]._vf, de::CPf(-0.809017, 0.5877853), tmp._vf);
        res._vf2 = make_float2(__fsub_rn(fmaf(recv[4]._vf.x, 0.309017, tmp._vf.x), __fmul_rn(recv[4]._vf.y, 0.9510565)),
            __fsub_rn(fmaf(recv[4]._vf.z, 0.309017, tmp._vf.z), __fmul_rn(recv[4]._vf.w, 0.9510565)));

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = make_uchar2(res._arrf[0], res._arrf[1]);
    }
}




__global__ void 
decx::dsp::fft::GPUK::cu_FFT2_R5_C2R_cplxf_fp32(const float4* __restrict src,
                                              float2* __restrict dst,
                                              const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                              const uint32_t _pitchsrc_v2,
                                              const uint32_t _pitchdst_v2)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 5;

    decx::utils::_cuda_vec128 recv[5], tmp;
    decx::utils::_cuda_vec64 res;
    de::CPf W;

    uint32_t _FFT_domain_dex = tidy;
    uint32_t _warp_loc_id = tidy % _kernel_info._store_pitch;

    if (tidy < _Bops_num && tidx < _pitchsrc_v2)
    {
#pragma unroll 5
        for (uint8_t i = 0; i < 5; ++i) {
            recv[i]._vf = src[_FFT_domain_dex * _pitchsrc_v2 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    const float _frac = __fdividef(_warp_loc_id, _kernel_info._warp_proc_len);
    W.construct_with_phase(__fmul_rn(Two_Pi, _frac));
    recv[1]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[1]._vf, W);

    W.construct_with_phase(__fmul_rn(Four_Pi, _frac));
    recv[2]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[2]._vf, W);

    W.construct_with_phase(__fmul_rn(Six_Pi, _frac));
    recv[3]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[3]._vf, W);

    W.construct_with_phase(__fmul_rn(Eight_Pi, _frac));
    recv[4]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[4]._vf, W);

    _FFT_domain_dex = (tidy / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + _warp_loc_id;

    if (tidy < _Bops_num && tidx < _pitchdst_v2)
    {
        for (uint8_t i = 0; i < 2; ++i) {
            res._arrf[i] = __fadd_rn(__fadd_rn(recv[0]._arrcplxf2[i].real, recv[1]._arrcplxf2[i].real) +
                recv[2]._arrcplxf2[i].real, __fadd_rn(recv[3]._arrcplxf2[i].real, recv[4]._arrcplxf2[i].real));
        }

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf2;
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vf = recv[0]._vf;
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[1]._vf, de::CPf(0.309017, 0.9510565), tmp._vf);
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[2]._vf, de::CPf(-0.809017, 0.5877853), tmp._vf);
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[3]._vf, de::CPf(-0.809017, -0.5877853), tmp._vf);
        res._vf2 = make_float2(__fsub_rn(fmaf(recv[4]._vf.x, 0.309017, tmp._vf.x), __fmul_rn(recv[4]._vf.y, -0.9510565)),
            __fsub_rn(fmaf(recv[4]._vf.z, 0.309017, tmp._vf.z), __fmul_rn(recv[4]._vf.w, -0.9510565)));

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf2;
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vf = recv[0]._vf;
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[1]._vf, de::CPf(-0.809017, 0.5877853), tmp._vf);
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[2]._vf, de::CPf(0.309017, -0.9510565), tmp._vf);
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[3]._vf, de::CPf(0.309017, 0.9510565), tmp._vf);
        res._vf2 = make_float2(__fsub_rn(fmaf(recv[4]._vf.x, -0.809017, tmp._vf.x), __fmul_rn(recv[4]._vf.y, -0.5877853)),
            __fsub_rn(fmaf(recv[4]._vf.z, -0.809017, tmp._vf.z), __fmul_rn(recv[4]._vf.w, -0.5877853)));

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf2;
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vf = recv[0]._vf;
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[1]._vf, de::CPf(-0.809017, -0.5877853), tmp._vf);
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[2]._vf, de::CPf(0.309017, 0.9510565), tmp._vf);
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[3]._vf, de::CPf(0.309017, -0.9510565), tmp._vf);
        res._vf2 = make_float2(__fsub_rn(fmaf(recv[4]._vf.x, -0.809017, tmp._vf.x), __fmul_rn(recv[4]._vf.y, 0.5877853)),
            __fsub_rn(fmaf(recv[4]._vf.z, -0.809017, tmp._vf.z), __fmul_rn(recv[4]._vf.w, 0.5877853)));

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf2;
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vf = recv[0]._vf;
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[1]._vf, de::CPf(0.309017, -0.9510565), tmp._vf);
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[2]._vf, de::CPf(-0.809017, -0.5877853), tmp._vf);
        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1_fp32(recv[3]._vf, de::CPf(-0.809017, 0.5877853), tmp._vf);
        res._vf2 = make_float2(__fsub_rn(fmaf(recv[4]._vf.x, 0.309017, tmp._vf.x), __fmul_rn(recv[4]._vf.y, 0.9510565)),
            __fsub_rn(fmaf(recv[4]._vf.z, 0.309017, tmp._vf.z), __fmul_rn(recv[4]._vf.w, 0.9510565)));

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf2;
    }
}