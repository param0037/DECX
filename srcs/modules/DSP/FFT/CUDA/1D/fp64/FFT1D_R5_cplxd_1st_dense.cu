/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "../FFT1D_1st_kernels_dense.cuh"



__global__ void 
decx::dsp::fft::GPUK::cu_FFT2D_R5_1st_cplxd_R2C(const double* __restrict src, 
                                                double2* __restrict dst,
                                                const uint32_t _signal_len, 
                                                const uint32_t _pitchsrc, 
                                                const uint32_t _pitchdst)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _signal_len / 5;

    double recv[5];
    decx::utils::_cuda_vec128 res;

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
        res._vd = decx::utils::vec2_set1_fp64(0.0);

        res._cplxd.real = __dadd_rn(__dadd_rn(recv[0], recv[1]),
                                    __dadd_rn(recv[2], recv[3]));
        res._cplxd.real = __dadd_rn(res._cplxd.real, recv[4]);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
        ++_FFT_domain_dex;

        res._cplxd.real = __fma_rn(0.309017, recv[1], recv[0]);
        res._cplxd.image = __dmul_rn(recv[1], 0.9510565);
        res._cplxd.real = __fma_rn(-0.809017, recv[2], res._cplxd.real);
        res._cplxd.image = __fma_rn(0.5877853, recv[2], res._cplxd.image);
        res._cplxd.real = __fma_rn(-0.809017, recv[3], res._cplxd.real);
        res._cplxd.image = __fma_rn(-0.5877853, recv[3], res._cplxd.image);
        res._cplxd.real = __fma_rn(0.309017, recv[4], res._cplxd.real);
        res._cplxd.image = __fma_rn(-0.9510565, recv[4], res._cplxd.image);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
        ++_FFT_domain_dex;

        res._cplxd.real = __fma_rn(-0.809017, recv[1], recv[0]);
        res._cplxd.image = __dmul_rn(recv[1], 0.5877853);
        res._cplxd.real = __fma_rn(0.309017, recv[2], res._cplxd.real);
        res._cplxd.image = __fma_rn(-0.9510565, recv[2], res._cplxd.image);
        res._cplxd.real = __fma_rn(0.309017, recv[3], res._cplxd.real);
        res._cplxd.image = __fma_rn(0.9510565, recv[3], res._cplxd.image);
        res._cplxd.real = __fma_rn(-0.809017, recv[4], res._cplxd.real);
        res._cplxd.image = __fma_rn(-0.5877853, recv[4], res._cplxd.image);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
        ++_FFT_domain_dex;

        res._cplxd.real = __fma_rn(-0.809017, recv[1], recv[0]);
        res._cplxd.image = __dmul_rn(recv[1], -0.5877853);
        res._cplxd.real = __fma_rn(0.309017, recv[2], res._cplxd.real);
        res._cplxd.image = __fma_rn(0.9510565, recv[2], res._cplxd.image);
        res._cplxd.real = __fma_rn(0.309017, recv[3], res._cplxd.real);
        res._cplxd.image = __fma_rn(-0.9510565, recv[3], res._cplxd.image);
        res._cplxd.real = __fma_rn(-0.809017, recv[4], res._cplxd.real);
        res._cplxd.image = __fma_rn(0.5877853, recv[4], res._cplxd.image);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
        ++_FFT_domain_dex;

        res._cplxd.real = __fma_rn(0.309017, recv[1], recv[0]);
        res._cplxd.image = __dmul_rn(recv[1], -0.9510565);
        res._cplxd.real = __fma_rn(-0.809017, recv[2], res._cplxd.real);
        res._cplxd.image = __fma_rn(-0.5877853, recv[2], res._cplxd.image);
        res._cplxd.real = __fma_rn(-0.809017, recv[3], res._cplxd.real);
        res._cplxd.image = __fma_rn(0.5877853, recv[3], res._cplxd.image);
        res._cplxd.real = __fma_rn(0.309017, recv[4], res._cplxd.real);
        res._cplxd.image = __fma_rn(0.9510565, recv[4], res._cplxd.image);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
    }
}



template <bool _div> __global__ void 
decx::dsp::fft::GPUK::cu_FFT2D_R5_1st_cplxd_C2C(const double2* __restrict src,
                                                      double2* __restrict dst,
                                                      const uint32_t _signal_len,
                                                      const uint32_t _pitchsrc,
                                                      const uint32_t _pitchdst,
                                                      const uint64_t _div_length)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const float _numer = __ull2float_rn(_div_length ? _div_length : _signal_len);
    const uint32_t _Bops_num = _signal_len / 5;

    decx::utils::_cuda_vec128 recv[5];
    decx::utils::_cuda_vec128 res;

    
    uint32_t _FFT_domain_dex = tidy;

    if (tidy < _Bops_num && tidx < _pitchsrc)
    {
#pragma unroll 5
        for (uint8_t i = 0; i < 5; ++i) {
            recv[i]._vd = src[_FFT_domain_dex * _pitchsrc + tidx];
            if (_div) { recv[i]._vd = decx::utils::cuda::__double_div2_1(recv[i]._vd, _numer); }
            _FFT_domain_dex += _Bops_num;
        }
    }

    _FFT_domain_dex = tidy * 5;

    if (tidy < _Bops_num && tidx < _pitchdst)
    {
        res._cplxd.real = __dadd_rn(__dadd_rn(recv[0]._cplxd.real, recv[1]._cplxd.real), 
                                      __dadd_rn(recv[2]._cplxd.real, recv[3]._cplxd.real));
        res._cplxd.real = __dadd_rn(res._cplxd.real, recv[4]._cplxd.real);
        res._cplxd.image = __dadd_rn(__dadd_rn(recv[0]._cplxd.image, recv[1]._cplxd.image), 
                                       __dadd_rn(recv[2]._cplxd.image, recv[3]._cplxd.image));
        res._cplxd.image = __dadd_rn(res._cplxd.image, recv[4]._cplxd.image);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
        ++_FFT_domain_dex;

        res._vd = recv[0]._vd;
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(0.309017, 0.9510565), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(-0.809017, 0.5877853), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(-0.809017, -0.5877853), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[4]._cplxd, de::CPd(0.309017, -0.9510565), res._cplxd);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
        ++_FFT_domain_dex;

        res._vd = recv[0]._vd;
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(-0.809017, 0.5877853), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(0.309017, -0.9510565), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(0.309017, 0.9510565), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[4]._cplxd, de::CPd(-0.809017, -0.5877853), res._cplxd);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
        ++_FFT_domain_dex;

        res._vd = recv[0]._vd;
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(-0.809017, -0.5877853), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(0.309017, 0.9510565), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(0.309017, -0.9510565), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[4]._cplxd, de::CPd(-0.809017, 0.5877853), res._cplxd);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
        ++_FFT_domain_dex;

        res._vd = recv[0]._vd;
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(0.309017, -0.9510565), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(-0.809017, -0.5877853), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(-0.809017, 0.5877853), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[4]._cplxd, de::CPd(0.309017, 0.9510565), res._cplxd);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
    }
}

template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R5_1st_cplxd_C2C<true>(const double2* __restrict, double2* __restrict,
    const uint32_t, const uint32_t, const uint32_t, const uint64_t);

template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R5_1st_cplxd_C2C<false>(const double2* __restrict, double2* __restrict,
    const uint32_t, const uint32_t, const uint32_t, const uint64_t);



template <bool _conj> __global__ void 
decx::dsp::fft::GPUK::cu_FFT2D_R5_end_cplxd_C2C(const double2* __restrict src,
                                           double2* __restrict dst,
                                           const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                           const uint32_t _pitchsrc,
                                           const uint32_t _pitchdst)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 5;

    decx::utils::_cuda_vec128 recv[5];
    decx::utils::_cuda_vec128 res;
    de::CPd W;

    uint32_t _FFT_domain_dex = tidy;
    const uint32_t _warp_loc_id = tidy % _kernel_info._store_pitch;

    if (tidy < _Bops_num && tidx < _pitchsrc)
    {
#pragma unroll 5
        for (uint8_t i = 0; i < 5; ++i) {
            recv[i]._vd = src[_FFT_domain_dex * _pitchsrc + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    const double _frac = __ddiv_rn(_warp_loc_id, _kernel_info._warp_proc_len);
    W.construct_with_phase(__dmul_rn(Two_Pi, _frac));
    recv[1]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[1]._cplxd, W);

    W.construct_with_phase(__dmul_rn(Four_Pi, _frac));
    recv[2]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[2]._cplxd, W);

    W.construct_with_phase(__dmul_rn(Six_Pi, _frac));
    recv[3]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[3]._cplxd, W);

    W.construct_with_phase(__dmul_rn(Eight_Pi, _frac));
    recv[4]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[4]._cplxd, W);

    _FFT_domain_dex = (tidy / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + _warp_loc_id;

    if (tidy < _Bops_num && tidx < _pitchdst) 
    {
        res._cplxd.real = __dadd_rn(__dadd_rn(recv[0]._cplxd.real, recv[1]._cplxd.real), 
                                      __dadd_rn(recv[2]._cplxd.real, recv[3]._cplxd.real));
        res._cplxd.real = __dadd_rn(res._cplxd.real, recv[4]._cplxd.real);
        res._cplxd.image = __dadd_rn(__dadd_rn(recv[0]._cplxd.image, recv[1]._cplxd.image), 
                                       __dadd_rn(recv[2]._cplxd.image, recv[3]._cplxd.image));
        res._cplxd.image = __dadd_rn(res._cplxd.image, recv[4]._cplxd.image);
        if (_conj) { res._cplxd = decx::dsp::fft::GPUK::_complex_conjugate_fp64(res._cplxd); }
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
        _FFT_domain_dex += _kernel_info._store_pitch;

        res._vd = recv[0]._vd;
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(0.309017, 0.9510565), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(-0.809017, 0.5877853), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(-0.809017, -0.5877853), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[4]._cplxd, de::CPd(0.309017, -0.9510565), res._cplxd);
        if (_conj) { res._cplxd = decx::dsp::fft::GPUK::_complex_conjugate_fp64(res._cplxd); }
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
        _FFT_domain_dex += _kernel_info._store_pitch;

        res._vd = recv[0]._vd;
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(-0.809017, 0.5877853), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(0.309017, -0.9510565), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(0.309017, 0.9510565), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[4]._cplxd, de::CPd(-0.809017, -0.5877853), res._cplxd);
        if (_conj) { res._cplxd = decx::dsp::fft::GPUK::_complex_conjugate_fp64(res._cplxd); }
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
        _FFT_domain_dex += _kernel_info._store_pitch;

        res._vd = recv[0]._vd;
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(-0.809017, -0.5877853), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(0.309017, 0.9510565), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(0.309017, -0.9510565), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[4]._cplxd, de::CPd(-0.809017, 0.5877853), res._cplxd);
        if (_conj) { res._cplxd = decx::dsp::fft::GPUK::_complex_conjugate_fp64(res._cplxd); }
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
        _FFT_domain_dex += _kernel_info._store_pitch;

        res._vd = recv[0]._vd;
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(0.309017, -0.9510565), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(-0.809017, -0.5877853), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(-0.809017, 0.5877853), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[4]._cplxd, de::CPd(0.309017, 0.9510565), res._cplxd);
        if (_conj) { res._cplxd = decx::dsp::fft::GPUK::_complex_conjugate_fp64(res._cplxd); }
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
    }
}

template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R5_end_cplxd_C2C<true>(const double2* __restrict, double2* __restrict,
    const decx::dsp::fft::FKI_4_2DK, const uint32_t, const uint32_t);

template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R5_end_cplxd_C2C<false>(const double2* __restrict, double2* __restrict,
    const decx::dsp::fft::FKI_4_2DK, const uint32_t, const uint32_t);



__global__ void 
decx::dsp::fft::GPUK::cu_FFT2D_R5_end_cplxd_C2R(const double2* __restrict src,
                                              double* __restrict dst,
                                              const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                              const uint32_t _pitchsrc,
                                              const uint32_t _pitchdst)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 5;

    decx::utils::_cuda_vec128 recv[5], tmp;
    double res;
    de::CPd W;

    uint32_t _FFT_domain_dex = tidy;
    uint32_t _warp_loc_id = tidy % _kernel_info._store_pitch;

    if (tidy < _Bops_num && tidx < _pitchsrc)
    {
#pragma unroll 5
        for (uint8_t i = 0; i < 5; ++i) {
            recv[i]._vd = src[_FFT_domain_dex * _pitchsrc + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    const double _frac = __ddiv_rn(_warp_loc_id, _kernel_info._warp_proc_len);
    W.construct_with_phase(__dmul_rn(Two_Pi, _frac));
    recv[1]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[1]._cplxd, W);

    W.construct_with_phase(__dmul_rn(Four_Pi, _frac));
    recv[2]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[2]._cplxd, W);

    W.construct_with_phase(__dmul_rn(Six_Pi, _frac));
    recv[3]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[3]._cplxd, W);

    W.construct_with_phase(__dmul_rn(Eight_Pi, _frac));
    recv[4]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[4]._cplxd, W);

    _FFT_domain_dex = (tidy / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + _warp_loc_id;

    if (tidy < _Bops_num && tidx < _pitchdst)
    {
        res = __dadd_rn(__dadd_rn(recv[0]._cplxd.real, recv[1]._cplxd.real), 
                        __dadd_rn(recv[2]._cplxd.real, recv[3]._cplxd.real));
        res = __dadd_rn(res, recv[4]._cplxd.real);

        dst[_FFT_domain_dex * _pitchdst + tidx] = res;
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vd = recv[0]._vd;
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(0.309017, 0.9510565), tmp._cplxd);
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(-0.809017, 0.5877853), tmp._cplxd);
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(-0.809017, -0.5877853), tmp._cplxd);
        res = __dsub_rn(__fma_rn(recv[4]._vd.x, 0.309017, tmp._vd.x), __dmul_rn(recv[4]._vd.y, -0.9510565));

        dst[_FFT_domain_dex * _pitchdst + tidx] = res;
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vd = recv[0]._vd;
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(-0.809017, 0.5877853), tmp._cplxd);
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(0.309017, -0.9510565), tmp._cplxd);
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(0.309017, 0.9510565), tmp._cplxd);
        res = __dsub_rn(__fma_rn(recv[4]._vd.x, -0.809017, tmp._vd.x), __dmul_rn(recv[4]._vd.y, -0.5877853));

        dst[_FFT_domain_dex * _pitchdst + tidx] = res;
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vd = recv[0]._vd;
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(-0.809017, -0.5877853), tmp._cplxd);
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(0.309017, 0.9510565), tmp._cplxd);
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(0.309017, -0.9510565), tmp._cplxd);
        res = __dsub_rn(__fma_rn(recv[4]._vd.x, -0.809017, tmp._vd.x), __dmul_rn(recv[4]._vd.y, 0.5877853));

        dst[_FFT_domain_dex * _pitchdst + tidx] = res;
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vd = recv[0]._vd;
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(0.309017, -0.9510565), tmp._cplxd);
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(-0.809017, -0.5877853), tmp._cplxd);
        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(-0.809017, 0.5877853), tmp._cplxd);
        res = __dsub_rn(__fma_rn(recv[4]._vd.x, 0.309017, tmp._vd.x), __dmul_rn(recv[4]._vd.y, 0.9510565));

        dst[_FFT_domain_dex * _pitchdst + tidx] = res;
    }
}
