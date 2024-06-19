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



// [32 * 2, 8] = [64, 8]
__global__ void 
decx::dsp::fft::GPUK::cu_FFT2D_R3_1st_cplxd_R2C(const double* __restrict src,
                                               double2* __restrict dst,
                                               const uint32_t _signal_len,
                                               const uint32_t _pitchsrc_v1,
                                               const uint32_t _pitchdst_v1)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _signal_len / 3;

    double recv[3], tmp;
    decx::utils::_cuda_vec128 res;

    uint32_t _FFT_domain_dex = tidy;

    if (tidy < _Bops_num && tidx < _pitchsrc_v1)
    {
#pragma unroll 3
        for (uint8_t i = 0; i < 3; ++i) {
            recv[i] = src[_FFT_domain_dex * _pitchsrc_v1 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    _FFT_domain_dex = tidy * 3;

    if (tidy < _Bops_num && tidx < _pitchdst_v1)
    {
        res._cplxd.real = __dadd_rn(__dadd_rn(recv[0], recv[1]), recv[2]);
        res._cplxd.image = 0.0;
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
        ++_FFT_domain_dex;
        
        tmp = __fma_rn(-0.5f, __dadd_rn(recv[1], recv[2]), recv[0]);

        res._cplxd.real = tmp;
        res._cplxd.image = __dmul_rn(__dsub_rn(recv[1], recv[2]), 0.8660254);
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
        ++_FFT_domain_dex;

        res._cplxd.real = tmp;
        res._cplxd.image = __dmul_rn(__dsub_rn(recv[2], recv[1]), 0.8660254);
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
    }
}



template<bool _div> __global__ void 
decx::dsp::fft::GPUK::cu_FFT2D_R3_1st_cplxd_C2C(const double2* __restrict src,
                                               double2* __restrict dst,
                                               const uint32_t _signal_len,
                                               const uint32_t _pitchsrc,
                                               const uint32_t _pitchdst,
                                               const uint64_t _div_length)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const double _numer = __ull2double_rn(_div_length ? _div_length : _signal_len);
    const uint32_t _Bops_num = _signal_len / 3;

    decx::utils::_cuda_vec128 recv[3];
    decx::utils::_cuda_vec128 res, tmp1, tmp2;

    uint32_t _FFT_domain_dex = tidy;

    if (tidy < _Bops_num && tidx < _pitchsrc)
    {
#pragma unroll 3
        for (uint8_t i = 0; i < 3; ++i) {
            recv[i]._vd = src[_FFT_domain_dex * _pitchsrc + tidx];
            if (_div) { recv[i]._vd = decx::utils::cuda::__double_div2_1(recv[i]._vd, _numer); }
            _FFT_domain_dex += _Bops_num;
        }
    }

    _FFT_domain_dex = tidy * 3;

    if (tidy < _Bops_num && tidx < _pitchdst)
    {
        res._vd = decx::utils::cuda::__double_add2(recv[2]._vd,
            decx::utils::cuda::__double_add2(recv[0]._vd, recv[1]._vd));

        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
        ++_FFT_domain_dex;

        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(-0.5, 0.8660254), recv[0]._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(-0.5, -0.8660254), res._cplxd);

        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
        ++_FFT_domain_dex;

        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(-0.5, -0.8660254), recv[0]._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(-0.5, 0.8660254), res._cplxd);

        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vd;
    }
}


template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R3_1st_cplxd_C2C<true>(const double2* __restrict, double2* __restrict,
    const uint32_t, const uint32_t, const uint32_t, const uint64_t);

template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R3_1st_cplxd_C2C<false>(const double2* __restrict, double2* __restrict,
    const uint32_t, const uint32_t, const uint32_t, const uint64_t);



template <bool _conj> __global__ void 
decx::dsp::fft::GPUK::cu_FFT2D_R3_end_cplxd_C2C(const double2* __restrict src,
                                                      double2* __restrict dst,
                                                      const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                                      const uint32_t _pitchsrc_v1,
                                                      const uint32_t _pitchdst_v1)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 3;

    decx::utils::_cuda_vec128 recv[3];
    decx::utils::_cuda_vec128 res, tmp1, tmp2;

    uint32_t _FFT_domain_dex, warp_loc_id;

    decx::utils::_cuda_vec128 W;

    warp_loc_id = tidy % _kernel_info._store_pitch;

    if (tidy < _Bops_num && tidx < _pitchdst_v1)
    {
        _FFT_domain_dex = tidy;
#pragma unroll 3
        for (uint8_t i = 0; i < 3; ++i) {
            recv[i]._vd = src[_FFT_domain_dex * _pitchsrc_v1 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    W._cplxd.construct_with_phase(__dmul_rn(Two_Pi, __ddiv_rn((double)warp_loc_id, (double)_kernel_info._warp_proc_len)));
    recv[1]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[1]._cplxd, W._cplxd);

    W._cplxd.construct_with_phase(__dmul_rn(Four_Pi, __ddiv_rn((double)warp_loc_id, (double)_kernel_info._warp_proc_len)));
    recv[2]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[2]._cplxd, W._cplxd);

    _FFT_domain_dex = (tidy / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + warp_loc_id;

    if (tidy < _Bops_num && tidx < _pitchdst_v1)
    {
        res._vd = decx::utils::cuda::__double_add2(recv[2]._vd,
            decx::utils::cuda::__double_add2(recv[0]._vd, recv[1]._vd));

        if (_conj) { res._cplxd = decx::dsp::fft::GPUK::_complex_conjugate_fp64(res._cplxd); }
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
        _FFT_domain_dex += _kernel_info._store_pitch;

        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(-0.5, 0.8660254f), recv[0]._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(-0.5, -0.8660254f), res._cplxd);

        if (_conj) { res._cplxd = decx::dsp::fft::GPUK::_complex_conjugate_fp64(res._cplxd); }
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
        _FFT_domain_dex += _kernel_info._store_pitch;

        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(-0.5, -0.8660254f), recv[0]._cplxd);
        res._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[2]._cplxd, de::CPd(-0.5, 0.8660254f), res._cplxd);

        if (_conj) { res._cplxd = decx::dsp::fft::GPUK::_complex_conjugate_fp64(res._cplxd); }
        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res._vd;
    }
}


template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R3_end_cplxd_C2C<true>(const double2* __restrict, double2* __restrict,
    const decx::dsp::fft::FKI_4_2DK, const uint32_t, const uint32_t);

template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R3_end_cplxd_C2C<false>(const double2* __restrict, double2* __restrict,
    const decx::dsp::fft::FKI_4_2DK, const uint32_t, const uint32_t);




__global__ void 
decx::dsp::fft::GPUK::cu_FFT2D_R3_end_cplxd_C2R(const double2* __restrict src,
                                              double* __restrict dst,
                                              const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                              const uint32_t _pitchsrc_v1,
                                              const uint32_t _pitchdst_v1)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 3;

    decx::utils::_cuda_vec128 recv[3], tmp;
    double res;
    de::CPd W;

    uint32_t _FFT_domain_dex = tidy;
    uint32_t _warp_loc_id = tidy % _kernel_info._store_pitch;

    if (tidy < _Bops_num && tidx < _pitchsrc_v1)
    {
#pragma unroll 3
        for (uint8_t i = 0; i < 3; ++i) {
            recv[i]._vd = src[_FFT_domain_dex * _pitchsrc_v1 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }
    
    W.construct_with_phase(__dmul_rn(Two_Pi, __ddiv_rn(__uint2double_rn(_warp_loc_id), __uint2double_rn(_kernel_info._warp_proc_len))));
    recv[1]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[1]._cplxd, W);

    W.construct_with_phase(__fmul_rn(Four_Pi, __ddiv_rn(__uint2double_rn(_warp_loc_id), __uint2double_rn(_kernel_info._warp_proc_len))));
    recv[2]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[2]._cplxd, W);

    _FFT_domain_dex = (tidy / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + _warp_loc_id;

    if (tidy < _Bops_num && tidx < _pitchdst_v1)
    {
        res = __dadd_rn(__dadd_rn(recv[0]._cplxd.real, recv[1]._cplxd.real), recv[2]._cplxd.real);

        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res;
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(-0.5, 0.8660254), recv[0]._cplxd);
        res = __dsub_rn(__fma_rn(recv[2]._vd.x, -0.5, tmp._vd.x), __fmul_rn(recv[2]._vd.y, -0.8660254f));

        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res;
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(recv[1]._cplxd, de::CPd(-0.5, -0.8660254f), recv[0]._cplxd);
        res = __dsub_rn(fmaf(recv[2]._vd.x, -0.5, tmp._vd.x), __fmul_rn(recv[2]._vd.y, 0.8660254f));

        dst[_FFT_domain_dex * _pitchdst_v1 + tidx] = res;
    }
}