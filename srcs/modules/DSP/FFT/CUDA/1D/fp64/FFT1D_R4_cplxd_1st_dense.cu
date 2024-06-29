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
decx::dsp::fft::GPUK::cu_FFT2D_R4_1st_cplxd_R2C(const double* __restrict src,
                                                      double2* __restrict dst,
                                                      const uint32_t _signal_len,
                                                      const uint32_t _pitchsrc,
                                                      const uint32_t _pitchdst)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _signal_len / 4;

    double recv[4];
    decx::utils::_cuda_vec128 res;

    uint32_t _FFT_domain_dex = tidy;

    if (tidy < _Bops_num && tidx < _pitchsrc)
    {
#pragma unroll 4
        for (uint8_t i = 0; i < 4; ++i) {
            recv[i] = src[_FFT_domain_dex * _pitchsrc + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    _FFT_domain_dex = tidy * 4;

    if (tidy < _Bops_num && tidx < _pitchdst)
    {
        // 1st
        res._cplxd.image = 0.0;
        res._cplxd.real = __dadd_rn(__dadd_rn(recv[0], recv[1]), __dadd_rn(recv[2], recv[3]));
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
        ++_FFT_domain_dex;

        // 2nd
        res._cplxd.real = __dsub_rn(recv[0], recv[2]);
        res._cplxd.image = __fsub_rn(recv[1], recv[3]);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
        ++_FFT_domain_dex;

        // 3rd
        res._cplxd.image = 0.f;
        res._cplxd.real = __dadd_rn(__dsub_rn(recv[0], recv[1]), __dsub_rn(recv[2], recv[3]));
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
        ++_FFT_domain_dex;

        // 4th 
        res._cplxd.real = __dsub_rn(recv[0], recv[2]);
        res._cplxd.image = __dsub_rn(recv[3], recv[1]);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
    }
}



template<bool _div> __global__ void 
decx::dsp::fft::GPUK::cu_FFT2D_R4_1st_cplxd_C2C(const double2* __restrict src,
                                               double2* __restrict dst,
                                               const uint32_t _signal_len,
                                               const uint32_t _pitchsrc,
                                               const uint32_t _pitchdst,
                                               const uint64_t _div_length)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const float _numer = __ull2float_rn(_div_length ? _div_length : _signal_len);
    const uint32_t _Bops_num = _signal_len / 4;

    decx::utils::_cuda_vec128 recv[4];
    decx::utils::_cuda_vec128 res, tmp1, tmp2;

    uint32_t _FFT_domain_dex = tidy;

    if (tidy < _Bops_num && tidx < _pitchsrc)
    {
#pragma unroll 4
        for (uint8_t i = 0; i < 4; ++i) {
            recv[i]._vd = src[_FFT_domain_dex * _pitchsrc + tidx];
            if (_div) { recv[i]._vd = decx::utils::cuda::__double_div2_1(recv[i]._vd, _numer); }
            _FFT_domain_dex += _Bops_num;
        }
    }

    _FFT_domain_dex = tidy * 4;

    if (tidy < _Bops_num && tidx < _pitchdst)
    {
        // Calculate the first and third output
        tmp1._vd = decx::utils::cuda::__double_add2(recv[0]._vd, recv[2]._vd);
        tmp2._vd = decx::utils::cuda::__double_add2(recv[1]._vd, recv[3]._vd);

        // Store the first output
        res._vd = decx::utils::cuda::__double_add2(tmp1._vd, tmp2._vd);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
        _FFT_domain_dex += 2;

        // Store the third output
        res._vd = decx::utils::cuda::__double_sub2(tmp1._vd, tmp2._vd);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;

        // Calculate and store the second output
        res._cplxd.real = __dsub_rn(__dadd_rn(recv[0]._cplxd.real, recv[3]._cplxd.image),
            __fadd_rn(recv[1]._cplxd.image, recv[2]._cplxd.real));
        res._cplxd.image = __dadd_rn(__dsub_rn(recv[0]._cplxd.image, recv[2]._cplxd.image),
            __dsub_rn(recv[1]._cplxd.real, recv[3]._cplxd.real));
        --_FFT_domain_dex;
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;

        res._cplxd.real = __dsub_rn(__dadd_rn(recv[0]._cplxd.real, recv[1]._cplxd.image),
            __dadd_rn(recv[2]._cplxd.real, recv[3]._cplxd.image));
        res._cplxd.image = __dsub_rn(__dsub_rn(recv[0]._cplxd.image, recv[1]._cplxd.real),
            __dsub_rn(recv[2]._cplxd.image, recv[3]._cplxd.real));

        _FFT_domain_dex += 2;
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
    }
}

template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R4_1st_cplxd_C2C<true>(const double2* __restrict, double2* __restrict,
    const uint32_t, const uint32_t, const uint32_t, const uint64_t);

template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R4_1st_cplxd_C2C<false>(const double2* __restrict, double2* __restrict,
    const uint32_t, const uint32_t, const uint32_t, const uint64_t);



template <bool _conj> __global__ void 
decx::dsp::fft::GPUK::cu_FFT2D_R4_end_cplxd_C2C(const double2* __restrict src, 
                                                      double2* __restrict dst,
                                                      const decx::dsp::fft::FKI_4_2DK _kernel_info, 
                                                      const uint32_t _pitchsrc, 
                                                      const uint32_t _pitchdst)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 4;

    decx::utils::_cuda_vec128 recv[4];
    decx::utils::_cuda_vec128 res, tmp1, tmp2;

    uint32_t _FFT_domain_dex, warp_loc_id;

    decx::utils::_cuda_vec128 W;

    warp_loc_id = tidy % _kernel_info._store_pitch;

    if (tidy < _Bops_num && tidx < _pitchdst)
    {
        _FFT_domain_dex = tidy;
#pragma unroll 4
        for (uint8_t i = 0; i < 4; ++i) {
            recv[i]._vd = src[_FFT_domain_dex * _pitchsrc + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    W._cplxd.construct_with_phase(__dmul_rn(Two_Pi, __ddiv_rn((double)warp_loc_id, (double)_kernel_info._warp_proc_len)));
    recv[1]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[1]._cplxd, W._cplxd);

    W._cplxd.construct_with_phase(__dmul_rn(Four_Pi, __ddiv_rn((double)warp_loc_id, (double)_kernel_info._warp_proc_len)));
    recv[2]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[2]._cplxd, W._cplxd);

    W._cplxd.construct_with_phase(__dmul_rn(Six_Pi, __ddiv_rn((double)warp_loc_id, (double)_kernel_info._warp_proc_len)));
    recv[3]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[3]._cplxd, W._cplxd);

    _FFT_domain_dex = (tidy / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + warp_loc_id;

    if (tidy < _Bops_num && tidx < _pitchdst)
    {
        // Calculate the first and third output
        tmp1._vd = decx::utils::cuda::__double_add2(recv[0]._vd, recv[2]._vd);
        tmp2._vd = decx::utils::cuda::__double_add2(recv[1]._vd, recv[3]._vd);

        // Store the first output
        res._vd = decx::utils::cuda::__double_add2(tmp1._vd, tmp2._vd);
        if (_conj) { res._cplxd = decx::dsp::fft::GPUK::_complex_conjugate_fp64(res._cplxd); }
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
        _FFT_domain_dex += (_kernel_info._store_pitch << 1);
        
        // Store the third output
        res._vd = decx::utils::cuda::__double_sub2(tmp1._vd, tmp2._vd);
        if (_conj) { res._cplxd = decx::dsp::fft::GPUK::_complex_conjugate_fp64(res._cplxd); }
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;

        // Calculate and store the second output
        res._cplxd.real = __dsub_rn(__dadd_rn(recv[0]._cplxd.real, recv[3]._cplxd.image), 
                                           __dadd_rn(recv[1]._cplxd.image, recv[2]._cplxd.real));
        res._cplxd.image = __dadd_rn(__dsub_rn(recv[0]._cplxd.image, recv[2]._cplxd.image), 
                                            __dsub_rn(recv[1]._cplxd.real, recv[3]._cplxd.real));
        _FFT_domain_dex -= (_kernel_info._store_pitch);
        if (_conj) { res._cplxd = decx::dsp::fft::GPUK::_complex_conjugate_fp64(res._cplxd); }
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;

        res._cplxd.real = __dsub_rn(__dadd_rn(recv[0]._cplxd.real, recv[1]._cplxd.image), 
                                           __dadd_rn(recv[2]._cplxd.real, recv[3]._cplxd.image));
        res._cplxd.image = __dsub_rn(__dsub_rn(recv[0]._cplxd.image, recv[1]._cplxd.real), 
                                            __dsub_rn(recv[2]._cplxd.image, recv[3]._cplxd.real));

        _FFT_domain_dex += (_kernel_info._store_pitch << 1);
        if (_conj) { res._cplxd = decx::dsp::fft::GPUK::_complex_conjugate_fp64(res._cplxd); }
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
    }
}

template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R4_end_cplxd_C2C<true>(const double2* __restrict, double2* __restrict,
    const decx::dsp::fft::FKI_4_2DK, const uint32_t, const uint32_t);

template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R4_end_cplxd_C2C<false>(const double2* __restrict, double2* __restrict,
    const decx::dsp::fft::FKI_4_2DK, const uint32_t, const uint32_t);



__global__ void 
decx::dsp::fft::GPUK::cu_FFT2D_R4_end_cplxd_C2R(const double2* __restrict src,
                                              double* __restrict dst,
                                              const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                              const uint32_t _pitchsrc,
                                              const uint32_t _pitchdst)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 4;

    decx::utils::_cuda_vec128 recv[4], tmp;
    decx::utils::_cuda_vec128 tmp1, tmp2;
    double res;
    de::CPd W;

    uint32_t _FFT_domain_dex = tidy;
    uint32_t _warp_loc_id = tidy % _kernel_info._store_pitch;

    if (tidy < _Bops_num && tidx < _pitchsrc)
    {
#pragma unroll 4
        for (uint8_t i = 0; i < 4; ++i) {
            recv[i]._vd = src[_FFT_domain_dex * _pitchsrc + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    W.construct_with_phase(__dmul_rn(Two_Pi, __fdividef((float)_warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[1]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[1]._cplxd, W);

    W.construct_with_phase(__dmul_rn(Four_Pi, __fdividef((float)_warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[2]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[2]._cplxd, W);

    W.construct_with_phase(__dmul_rn(Six_Pi, __fdividef((float)_warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[3]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[3]._cplxd, W);

    _FFT_domain_dex = (tidy / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + _warp_loc_id;

    if (tidy < _Bops_num && tidx < _pitchdst)
    {
        // Calculate the first and third output
        tmp1._vd = decx::utils::cuda::__double_add2(recv[0]._vd, recv[2]._vd);
        tmp2._vd = decx::utils::cuda::__double_add2(recv[1]._vd, recv[3]._vd);

        // Store the first output
        res = __dadd_rn(tmp1._vd.x, tmp2._vd.x);

        dst[_FFT_domain_dex * _pitchdst + tidx] = res;
        _FFT_domain_dex += (_kernel_info._store_pitch << 1);
        
        // Store the third output
        res = __dsub_rn(tmp1._vd.x, tmp2._vd.x);

        dst[_FFT_domain_dex * _pitchdst + tidx] = res;

        // Calculate and store the second output
        res = __dsub_rn(__dadd_rn(recv[0]._cplxd.real, recv[3]._cplxd.image), 
                        __dadd_rn(recv[1]._cplxd.image, recv[2]._cplxd.real));
        _FFT_domain_dex -= (_kernel_info._store_pitch);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res;

        res = __dsub_rn(__fadd_rn(recv[0]._cplxd.real, recv[1]._cplxd.image), 
                        __fadd_rn(recv[2]._cplxd.real, recv[3]._cplxd.image));

        _FFT_domain_dex += (_kernel_info._store_pitch << 1);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res;
    }
}
