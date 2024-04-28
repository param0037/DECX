/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "../FFT3D_kernels.cuh"


template <bool _div> __global__ void 
decx::dsp::fft::GPUK::cu_FFT3_R5_1st_C2C_cplxd(const double2* __restrict src, 
                                               double2* __restrict dst, 
                                               const uint32_t _signal_len,
                                               const uint2 _signal_pitch, 
                                               const uint32_t _pitchsrc_v1, 
                                               const uint32_t _pitchdst_v1,
                                               const uint32_t _paral)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _signal_len / 5;

    decx::utils::_cuda_vec128 recv[5];
    decx::utils::_cuda_vec128 res;

    uint32_t _FFT_domain_dex = (tidy % _Bops_num);
    const uint32_t _lane_id = tidy / _Bops_num;

    if (tidy < _Bops_num * _paral && tidx < _pitchsrc_v1)
    {
#pragma unroll 5
        for (uint8_t i = 0; i < 5; ++i) {
            recv[i]._vd = src[(_FFT_domain_dex + _lane_id * _signal_pitch.x) * _pitchsrc_v1 + tidx];
            if (_div) { recv[i]._vd = decx::utils::cuda::__double_div2_1(recv[i]._vd, _signal_len); }
            _FFT_domain_dex += _Bops_num;
        }
    }

    _FFT_domain_dex = (tidy % _Bops_num) * 5 + _lane_id * _signal_pitch.y;

    if (tidy < _Bops_num * _paral && tidx < _pitchdst_v1)
    {
        res._cplxd.real = __dadd_rn(__dadd_rn(recv[0]._cplxd.real, recv[1]._cplxd.real) + 
                                    recv[2]._cplxd.real, __dadd_rn(recv[3]._cplxd.real, recv[4]._cplxd.real));
        res._cplxd.image = __dadd_rn(__dadd_rn(recv[0]._cplxd.image, recv[1]._cplxd.image) +
                                    recv[2]._cplxd.image, __dadd_rn(recv[3]._cplxd.image, recv[4]._cplxd.image));
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
        ++_FFT_domain_dex;

        res._cplxd = recv[0]._cplxd;
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(0.309017, 0.9510565),  res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(-0.809017, 0.5877853), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(-0.809017, -0.5877853),res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[4]._cplxd, de::CPd(0.309017, -0.9510565), res._cplxd);
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
        ++_FFT_domain_dex;

        res._cplxd = recv[0]._cplxd;
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(-0.809017, 0.5877853), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(0.309017, -0.9510565), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(0.309017, 0.9510565),  res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[4]._cplxd, de::CPd(-0.809017, -0.5877853),res._cplxd);
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
        ++_FFT_domain_dex;

        res._cplxd = recv[0]._cplxd;
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(-0.809017, -0.5877853), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(0.309017, 0.9510565),   res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(0.309017, -0.9510565),  res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[4]._cplxd, de::CPd(-0.809017, 0.5877853),  res._cplxd);
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
        ++_FFT_domain_dex;

        res._cplxd = recv[0]._cplxd;
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(0.309017, -0.9510565),  res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(-0.809017, -0.5877853), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(-0.809017, 0.5877853),  res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[4]._cplxd, de::CPd(0.309017, 0.9510565),   res._cplxd);
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
    }
}

template __global__ void
decx::dsp::fft::GPUK::cu_FFT3_R5_1st_C2C_cplxd<true>(const double2* __restrict, double2* __restrict,
    const uint32_t, const uint2, const uint32_t, const uint32_t, const uint32_t);

template __global__ void
decx::dsp::fft::GPUK::cu_FFT3_R5_1st_C2C_cplxd<false>(const double2* __restrict, double2* __restrict,
    const uint32_t, const uint2, const uint32_t, const uint32_t, const uint32_t);



__global__ void
decx::dsp::fft::GPUK::cu_FFT3_R5_C2C_cplxd(const double2* __restrict src, 
                                           double2* __restrict dst, 
                                           const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                           const uint32_t signal_pitch, 
                                           const uint32_t _pitchsrc_v1, 
                                           const uint32_t _pitchdst_v1, 
                                           const uint32_t _paral)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 5;

    decx::utils::_cuda_vec128 recv[5];
    decx::utils::_cuda_vec128 res;
    de::CPd W;

    uint32_t _FFT_domain_dex = (tidy % _Bops_num);
    const uint32_t _lane_id = tidy / _Bops_num;

    if (tidy < _Bops_num * _paral && tidx < _pitchsrc_v1)
    {
#pragma unroll 5
        for (uint8_t i = 0; i < 5; ++i) {
            recv[i]._vd = src[(_FFT_domain_dex + _lane_id * signal_pitch) * _pitchsrc_v1 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }
    const uint32_t _warp_loc_id = (tidy % _Bops_num) % _kernel_info._store_pitch;

    const double _frac = __ddiv_rn(_warp_loc_id, _kernel_info._warp_proc_len);
    W.construct_with_phase(__dmul_rn(Two_Pi, _frac));
    recv[1]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[1]._cplxd, W);

    W.construct_with_phase(__dmul_rn(Four_Pi, _frac));
    recv[2]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[2]._cplxd, W);

    W.construct_with_phase(__dmul_rn(Six_Pi, _frac));
    recv[3]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[3]._cplxd, W);

    W.construct_with_phase(__dmul_rn(Eight_Pi, _frac));
    recv[4]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[4]._cplxd, W);

    _FFT_domain_dex = ((tidy % _Bops_num) / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + _warp_loc_id
        + _lane_id * signal_pitch;

    if (tidy < _Bops_num * _paral && tidx < _pitchdst_v1) 
    {
        res._cplxd.real = __dadd_rn(__dadd_rn(recv[0]._cplxd.real, recv[1]._cplxd.real) +
                                              recv[2]._cplxd.real, __dadd_rn(recv[3]._cplxd.real, recv[4]._cplxd.real));
        res._cplxd.image = __dadd_rn(__dadd_rn(recv[0]._cplxd.image, recv[1]._cplxd.image) +
            recv[2]._cplxd.image, __dadd_rn(recv[3]._cplxd.image, recv[4]._cplxd.image));
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
        _FFT_domain_dex += _kernel_info._store_pitch;

        res._vd = recv[0]._vd;
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(0.309017, 0.9510565), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(-0.809017, 0.5877853), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(-0.809017, -0.5877853), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[4]._cplxd, de::CPd(0.309017, -0.9510565), res._cplxd);

        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
        _FFT_domain_dex += _kernel_info._store_pitch;

        res._vd = recv[0]._vd;
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(-0.809017, 0.5877853), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(0.309017, -0.9510565), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(0.309017, 0.9510565), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[4]._cplxd, de::CPd(-0.809017, -0.5877853), res._cplxd);

        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
        _FFT_domain_dex += _kernel_info._store_pitch;

        res._vd = recv[0]._vd;
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(-0.809017, -0.5877853), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(0.309017, 0.9510565), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(0.309017, -0.9510565), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[4]._cplxd, de::CPd(-0.809017, 0.5877853), res._cplxd);

        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
        _FFT_domain_dex += _kernel_info._store_pitch;

        res._vd = recv[0]._vd;
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(0.309017, -0.9510565), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(-0.809017, -0.5877853), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[3]._cplxd, de::CPd(-0.809017, 0.5877853), res._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[4]._cplxd, de::CPd(0.309017, 0.9510565), res._cplxd);

        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
    }
}
