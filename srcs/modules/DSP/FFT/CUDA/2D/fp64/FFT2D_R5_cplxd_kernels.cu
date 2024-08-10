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


#include "../FFT2D_kernels.cuh"


// [32 * 2, 8] = [64, 8]
__global__ void decx::dsp::fft::GPUK::
cu_FFT2_R5_1st_R2C_uc8_cplxd(const uint8_t* __restrict src,
                             double2* __restrict dst,
                             const uint32_t _signal_len,
                             const uint32_t _pitchsrc_v1,
                             const uint32_t _pitchdst_v1)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _signal_len / 5;

    uint8_t recv[5];
    decx::utils::_cuda_vec128 res;

    uint32_t _FFT_domain_dex = tidy;

    if (tidy < _Bops_num && tidx < _pitchsrc_v1)
    {
#pragma unroll 5
        for (uint8_t i = 0; i < 5; ++i) {
            recv[i] = src[_FFT_domain_dex * _pitchsrc_v1 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    _FFT_domain_dex = tidy * 5;

    if (tidy < _Bops_num && tidx < _pitchdst_v1)
    {
        res._vd = decx::utils::vec2_set1_fp64(0.0);

        res._cplxd.real = __dadd_rn(__dadd_rn(recv[0], recv[1]),
                                    __dadd_rn(recv[2], recv[3]));
        res._cplxd.real = __dadd_rn(res._cplxd.real, recv[4]);
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
        ++_FFT_domain_dex;

        res._cplxd.real = __fma_rn(0.30901699437494742, recv[1], recv[0]);
        res._cplxd.image = __dmul_rn(recv[1], 0.95105651629515357);
        res._cplxd.real = __fma_rn(-0.80901699437494742, recv[2], res._cplxd.real);
        res._cplxd.image = __fma_rn(0.58778525229247313, recv[2], res._cplxd.image);
        res._cplxd.real = __fma_rn(-0.80901699437494742, recv[3], res._cplxd.real);
        res._cplxd.image = __fma_rn(-0.58778525229247313, recv[3], res._cplxd.image);
        res._cplxd.real = __fma_rn(0.30901699437494742, recv[4], res._cplxd.real);
        res._cplxd.image = __fma_rn(-0.95105651629515357, recv[4], res._cplxd.image);
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
        ++_FFT_domain_dex;

        res._cplxd.real = __fma_rn(-0.80901699437494742, recv[1], recv[0]);
        res._cplxd.image = __dmul_rn(recv[1], 0.58778525229247313);
        res._cplxd.real = __fma_rn(0.30901699437494742, recv[2], res._cplxd.real);
        res._cplxd.image = __fma_rn(-0.95105651629515357, recv[2], res._cplxd.image);
        res._cplxd.real = __fma_rn(0.30901699437494742, recv[3], res._cplxd.real);
        res._cplxd.image = __fma_rn(0.95105651629515357, recv[3], res._cplxd.image);
        res._cplxd.real = __fma_rn(-0.80901699437494742, recv[4], res._cplxd.real);
        res._cplxd.image = __fma_rn(-0.58778525229247313, recv[4], res._cplxd.image);
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
        ++_FFT_domain_dex;

        res._cplxd.real = __fma_rn(-0.80901699437494742, recv[1], recv[0]);
        res._cplxd.image = __dmul_rn(recv[1], -0.58778525229247313);
        res._cplxd.real = __fma_rn(0.30901699437494742, recv[2], res._cplxd.real);
        res._cplxd.image = __fma_rn(0.95105651629515357, recv[2], res._cplxd.image);
        res._cplxd.real = __fma_rn(0.30901699437494742, recv[3], res._cplxd.real);
        res._cplxd.image = __fma_rn(-0.95105651629515357, recv[3], res._cplxd.image);
        res._cplxd.real = __fma_rn(-0.80901699437494742, recv[4], res._cplxd.real);
        res._cplxd.image = __fma_rn(0.58778525229247313, recv[4], res._cplxd.image);
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
        ++_FFT_domain_dex;

        res._cplxd.real = __fma_rn(0.30901699437494742, recv[1], recv[0]);
        res._cplxd.image = __dmul_rn(recv[1], -0.95105651629515357);
        res._cplxd.real = __fma_rn(-0.80901699437494742, recv[2], res._cplxd.real);
        res._cplxd.image = __fma_rn(-0.58778525229247313, recv[2], res._cplxd.image);
        res._cplxd.real = __fma_rn(-0.80901699437494742, recv[3], res._cplxd.real);
        res._cplxd.image = __fma_rn(0.58778525229247313, recv[3], res._cplxd.image);
        res._cplxd.real = __fma_rn(0.30901699437494742, recv[4], res._cplxd.real);
        res._cplxd.image = __fma_rn(0.95105651629515357, recv[4], res._cplxd.image);
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
    }
}



__global__ void 
decx::dsp::fft::GPUK::cu_FFT2_R5_C2R_cplxd_u8(const double2* __restrict src,
                                              uint8_t* __restrict dst,
                                              const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                              const uint32_t _pitchsrc_v2,
                                              const uint32_t _pitchdst_v2)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 5;

    decx::utils::_cuda_vec128 recv[5], tmp;
    decx::utils::_cuda_vec64 res;
    de::CPd W;

    uint32_t _FFT_domain_dex = tidy;
    uint32_t _warp_loc_id = tidy % _kernel_info._store_pitch;

    if (tidy < _Bops_num && tidx < _pitchsrc_v2)
    {
#pragma unroll 5
        for (uint8_t i = 0; i < 5; ++i) {
            recv[i]._vd = src[_FFT_domain_dex * _pitchsrc_v2 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    const float _frac = __ddiv_rn(__ull2double_rn(_warp_loc_id), 
                                  __ull2double_rn(_kernel_info._warp_proc_len));
    W.construct_with_phase(__dmul_rn(Two_Pi, _frac));
    recv[1]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[1]._cplxd, W);

    W.construct_with_phase(__dmul_rn(Four_Pi, _frac));
    recv[2]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[2]._cplxd, W);

    W.construct_with_phase(__dmul_rn(Six_Pi, _frac));
    recv[3]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[3]._cplxd, W);

    W.construct_with_phase(__dmul_rn(Eight_Pi, _frac));
    recv[4]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[4]._cplxd, W);

    _FFT_domain_dex = (tidy / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + _warp_loc_id;

    if (tidy < _Bops_num && tidx < _pitchdst_v2)
    {
        res._fp64 = __dadd_rn(__dadd_rn(recv[0]._cplxd.real, recv[1]._cplxd.real) +
            recv[2]._cplxd.real, __dadd_rn(recv[3]._cplxd.real, recv[4]._cplxd.real));
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = __double2ull_rn(res._fp64);
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vd = recv[0]._vd;
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(0.30901699437494742, 0.95105651629515357), tmp._cplxd);
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(-0.80901699437494742, 0.58778525229247313), tmp._cplxd);
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(-0.80901699437494742, -0.58778525229247313), tmp._cplxd);
        res._fp64 = __dsub_rn(__fma_rn(recv[4]._vd.x, 0.30901699437494742, tmp._vd.x), 
                              __dmul_rn(recv[4]._vd.y, -0.95105651629515357));
        
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = __double2ull_rn(res._fp64);
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vd = recv[0]._vd;
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(-0.80901699437494742, 0.58778525229247313), tmp._cplxd);
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(0.30901699437494742, -0.95105651629515357), tmp._cplxd);
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(0.30901699437494742, 0.95105651629515357), tmp._cplxd);
        res._fp64 = __dsub_rn(__fma_rn(recv[4]._vd.x, -0.80901699437494742, tmp._vd.x),
                              __dmul_rn(recv[4]._vd.y, -0.58778525229247313));

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = __double2ull_rn(res._fp64);
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vd = recv[0]._vd;
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(-0.80901699437494742, -0.58778525229247313), tmp._cplxd);
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(0.30901699437494742, 0.95105651629515357), tmp._cplxd);
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(0.30901699437494742, -0.95105651629515357), tmp._cplxd);
        res._fp64 = __dsub_rn(__fma_rn(recv[4]._vd.x, -0.80901699437494742, tmp._vd.x),
                              __dmul_rn(recv[4]._vd.y, 0.58778525229247313));

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = __double2ull_rn(res._fp64);
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vd = recv[0]._vd;
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(0.30901699437494742, -0.95105651629515357), tmp._cplxd);
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(-0.80901699437494742, -0.58778525229247313), tmp._cplxd);
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(-0.80901699437494742, 0.58778525229247313), tmp._cplxd);
        res._fp64 = __dsub_rn(__fma_rn(recv[4]._vd.x, 0.30901699437494742, tmp._vd.x), 
                              __dmul_rn(recv[4]._vd.y, 0.95105651629515357));

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = __double2ull_rn(res._fp64);
    }
}
