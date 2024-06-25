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
decx::dsp::fft::GPUK::cu_FFT2_R3_1st_R2C_cplxf(const float2* __restrict src,
                                               float4* __restrict dst,
                                               const uint32_t _signal_len,
                                               const uint32_t _pitchsrc_v2,
                                               const uint32_t _pitchdst_v2)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _signal_len / 3;

    decx::utils::_cuda_vec64 recv[3], tmp;
    decx::utils::_cuda_vec128 res;

    uint32_t _FFT_domain_dex = tidy;

    if (tidy < _Bops_num && tidx < _pitchsrc_v2)
    {
#pragma unroll 3
        for (uint8_t i = 0; i < 3; ++i) {
            recv[i]._vf2 = src[_FFT_domain_dex * _pitchsrc_v2 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    _FFT_domain_dex = tidy * 3;

    if (tidy < _Bops_num && tidx < _pitchdst_v2)
    {
        res._arrcplxf2[0].real = __fadd_rn(__fadd_rn(recv[0]._arrf[0], recv[1]._arrf[0]), recv[2]._arrf[0]);
        res._arrcplxf2[0].image = 0.f;
        res._arrcplxf2[1].real = __fadd_rn(__fadd_rn(recv[0]._arrf[1], recv[1]._arrf[1]), recv[2]._arrf[1]);
        res._arrcplxf2[1].image = 0.f;
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        tmp._arrf[0] = __fmaf_rn(-0.5f, __fadd_rn(recv[1]._arrf[0], recv[2]._arrf[0]), recv[0]._arrf[0]);
        tmp._arrf[1] = __fmaf_rn(-0.5f, __fadd_rn(recv[1]._arrf[1], recv[2]._arrf[1]), recv[0]._arrf[1]);

        res._arrcplxf2[0].real = tmp._arrf[0];
        res._arrcplxf2[0].image = __fmul_rn(__fsub_rn(recv[1]._arrf[0], recv[2]._arrf[0]), 0.8660254f);
        res._arrcplxf2[1].real = tmp._arrf[1];
        res._arrcplxf2[1].image = __fmul_rn(__fsub_rn(recv[1]._arrf[1], recv[2]._arrf[1]), 0.8660254f);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        res._arrcplxf2[0].real = tmp._arrf[0];
        res._arrcplxf2[0].image = __fmul_rn(__fsub_rn(recv[2]._arrf[0], recv[1]._arrf[0]), 0.8660254f);
        res._arrcplxf2[1].real = tmp._arrf[1];
        res._arrcplxf2[1].image = __fmul_rn(__fsub_rn(recv[2]._arrf[1], recv[1]._arrf[1]), 0.8660254f);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
    }
}



// [32 * 2, 8] = [64, 8]
__global__ void 
decx::dsp::fft::GPUK::cu_FFT2_R3_1st_R2C_uc8_cplxf(const ushort* __restrict src,
                                               float4* __restrict dst,
                                               const uint32_t _signal_len,
                                               const uint32_t _pitchsrc_v2,
                                               const uint32_t _pitchdst_v2)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _signal_len / 3;

    uchar2 recv[3];
    decx::utils::_cuda_vec128 tmp, res;

    uint32_t _FFT_domain_dex = tidy;

    if (tidy < _Bops_num && tidx < _pitchsrc_v2)
    {
#pragma unroll 3
        for (uint8_t i = 0; i < 3; ++i) {
            *((ushort*)&recv[i]) = src[_FFT_domain_dex * _pitchsrc_v2 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    _FFT_domain_dex = tidy * 3;

    if (tidy < _Bops_num && tidx < _pitchdst_v2)
    {
        res._arrcplxf2[0].real = __fadd_rn(__fadd_rn(recv[0].x, recv[1].x), recv[2].x);
        res._arrcplxf2[0].image = 0.f;
        res._arrcplxf2[1].real = __fadd_rn(__fadd_rn(recv[0].y, recv[1].y), recv[2].y);
        res._arrcplxf2[1].image = 0.f;
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        tmp._arrf[0] = __fmaf_rn(-0.5f, __fadd_rn(recv[1].x, recv[2].x), recv[0].x);
        tmp._arrf[1] = __fmaf_rn(-0.5f, __fadd_rn(recv[1].y, recv[2].y), recv[0].y);

        res._arrcplxf2[0].real = tmp._arrf[0];
        res._arrcplxf2[0].image = __fmul_rn(__fsub_rn(recv[1].x, recv[2].x), 0.8660254f);
        res._arrcplxf2[1].real = tmp._arrf[1];
        res._arrcplxf2[1].image = __fmul_rn(__fsub_rn(recv[1].y, recv[2].y), 0.8660254f);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        res._arrcplxf2[0].real = tmp._arrf[0];
        res._arrcplxf2[0].image = __fmul_rn(__fsub_rn(recv[2].x, recv[1].x), 0.8660254f);
        res._arrcplxf2[1].real = tmp._arrf[1];
        res._arrcplxf2[1].image = __fmul_rn(__fsub_rn(recv[2].y, recv[1].y), 0.8660254f);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
    }
}



template<bool _div> __global__ void 
decx::dsp::fft::GPUK::cu_FFT2_R3_1st_C2C_cplxf(const float4* __restrict src,
                                               float4* __restrict dst,
                                               const uint32_t _signal_len,
                                               const uint32_t _pitchsrc_v2,
                                               const uint32_t _pitchdst_v2,
                                               const uint64_t _div_length)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _signal_len / 3;

    decx::utils::_cuda_vec128 recv[3];
    decx::utils::_cuda_vec128 res, tmp1, tmp2;

    const float _numer = __ull2float_rn(_div_length ? _div_length : _signal_len);
    uint32_t _FFT_domain_dex = tidy;

    if (tidy < _Bops_num && tidx < _pitchsrc_v2)
    {
#pragma unroll 3
        for (uint8_t i = 0; i < 3; ++i) {
            recv[i]._vf = src[_FFT_domain_dex * _pitchsrc_v2 + tidx];
            if (_div) { recv[i]._vf = decx::utils::cuda::__float_div4_1(recv[i]._vf, _numer); }
            _FFT_domain_dex += _Bops_num;
        }
    }

    _FFT_domain_dex = tidy * 3;

    if (tidy < _Bops_num && tidx < _pitchdst_v2) 
    {
        res._vf = decx::utils::cuda::__float_add4(recv[2]._vf,
            decx::utils::cuda::__float_add4(recv[0]._vf, recv[1]._vf));
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        res._vf = decx::dsp::fft::GPUK::_complex_2fma1(recv[1]._vf, de::CPf(-0.5, 0.8660254f), recv[0]._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1(recv[2]._vf, de::CPf(-0.5, -0.8660254f), res._vf);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        res._vf = decx::dsp::fft::GPUK::_complex_2fma1(recv[1]._vf, de::CPf(-0.5, -0.8660254f), recv[0]._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1(recv[2]._vf, de::CPf(-0.5, 0.8660254f), res._vf);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
    }
}

template __global__ void
decx::dsp::fft::GPUK::cu_FFT2_R3_1st_C2C_cplxf<true>(const float4* __restrict, float4* __restrict,
    const uint32_t, const uint32_t, const uint32_t, const uint64_t);

template __global__ void
decx::dsp::fft::GPUK::cu_FFT2_R3_1st_C2C_cplxf<false>(const float4* __restrict, float4* __restrict,
    const uint32_t, const uint32_t, const uint32_t, const uint64_t);




template <bool _conj> __global__ void 
decx::dsp::fft::GPUK::cu_FFT2_R3_C2C_cplxf(const float4* __restrict src,
                                           float4* __restrict dst,
                                           const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                           const uint32_t _pitchsrc_v2,
                                           const uint32_t _pitchdst_v2)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 3;

    decx::utils::_cuda_vec128 recv[3];
    decx::utils::_cuda_vec128 res, tmp1, tmp2;

    uint32_t _FFT_domain_dex, warp_loc_id;

    decx::utils::_cuda_vec64 W;

    warp_loc_id = tidy % _kernel_info._store_pitch;

    if (tidy < _Bops_num && tidx < _pitchdst_v2)
    {
        _FFT_domain_dex = tidy;
#pragma unroll 3
        for (uint8_t i = 0; i < 3; ++i) {
            recv[i]._vf = src[_FFT_domain_dex * _pitchsrc_v2 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    W._cplxf32.construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[1]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[1]._vf, W._cplxf32);

    W._cplxf32.construct_with_phase(__fmul_rn(Four_Pi, __fdividef((float)warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[2]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[2]._vf, W._cplxf32);

    _FFT_domain_dex = (tidy / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + warp_loc_id;

    if (tidy < _Bops_num && tidx < _pitchdst_v2)
    {
        res._vf = decx::utils::cuda::__float_add4(recv[2]._vf,
            decx::utils::cuda::__float_add4(recv[0]._vf, recv[1]._vf));

        if (_conj) { res = decx::dsp::fft::GPUK::_complex4_conjugate_fp32(res); }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        _FFT_domain_dex += _kernel_info._store_pitch;

        res._vf = decx::dsp::fft::GPUK::_complex_2fma1(recv[1]._vf, de::CPf(-0.5, 0.8660254f), recv[0]._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1(recv[2]._vf, de::CPf(-0.5, -0.8660254f), res._vf);

        if (_conj) { res = decx::dsp::fft::GPUK::_complex4_conjugate_fp32(res); }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        _FFT_domain_dex += _kernel_info._store_pitch;

        res._vf = decx::dsp::fft::GPUK::_complex_2fma1(recv[1]._vf, de::CPf(-0.5, -0.8660254f), recv[0]._vf);
        res._vf = decx::dsp::fft::GPUK::_complex_2fma1(recv[2]._vf, de::CPf(-0.5, 0.8660254f), res._vf);

        if (_conj) { res = decx::dsp::fft::GPUK::_complex4_conjugate_fp32(res); }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
    }
}

template __global__ void decx::dsp::fft::GPUK::cu_FFT2_R3_C2C_cplxf<true>(const float4* __restrict, float4* __restrict,
    const decx::dsp::fft::FKI_4_2DK, const uint32_t, const uint32_t);

template __global__ void decx::dsp::fft::GPUK::cu_FFT2_R3_C2C_cplxf<false>(const float4* __restrict, float4* __restrict,
    const decx::dsp::fft::FKI_4_2DK, const uint32_t, const uint32_t);




__global__ void 
decx::dsp::fft::GPUK::cu_FFT2_R3_C2R_cplxf_u8(const float4* __restrict src,
                                              uchar2* __restrict dst,
                                              const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                              const uint32_t _pitchsrc_v2,
                                              const uint32_t _pitchdst_v2)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 3;

    decx::utils::_cuda_vec128 recv[3], tmp;
    decx::utils::_cuda_vec64 res;
    de::CPf W;

    uint32_t _FFT_domain_dex = tidy;
    uint32_t _warp_loc_id = tidy % _kernel_info._store_pitch;

    if (tidy < _Bops_num && tidx < _pitchsrc_v2)
    {
#pragma unroll 3
        for (uint8_t i = 0; i < 3; ++i) {
            recv[i]._vf = src[_FFT_domain_dex * _pitchsrc_v2 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    W.construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)_warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[1]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[1]._vf, W);

    W.construct_with_phase(__fmul_rn(Four_Pi, __fdividef((float)_warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[2]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[2]._vf, W);

    _FFT_domain_dex = (tidy / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + _warp_loc_id;

    if (tidy < _Bops_num && tidx < _pitchdst_v2)
    {
        res._vf2.x = __fadd_rn(__fadd_rn(recv[0]._vf.x, recv[1]._vf.x), recv[2]._vf.x);
        res._vf2.y = __fadd_rn(__fadd_rn(recv[0]._vf.z, recv[1]._vf.z), recv[2]._vf.z);

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = make_uchar2(res._vf2.x, res._vf2.y);
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1(recv[1]._vf, de::CPf(-0.5, 0.8660254f), recv[0]._vf);
        res._vf2.x = __fsub_rn(__fmaf_rn(recv[2]._vf.x, -0.5, tmp._vf.x), __fmul_rn(recv[2]._vf.y, -0.8660254f));
        res._vf2.y = __fsub_rn(__fmaf_rn(recv[2]._vf.z, -0.5, tmp._vf.z), __fmul_rn(recv[2]._vf.w, -0.8660254f));

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = make_uchar2(res._vf2.x, res._vf2.y);
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1(recv[1]._vf, de::CPf(-0.5, -0.8660254f), recv[0]._vf);
        res._vf2.x = __fsub_rn(__fmaf_rn(recv[2]._vf.x, -0.5, tmp._vf.x), __fmul_rn(recv[2]._vf.y, 0.8660254f));
        res._vf2.y = __fsub_rn(__fmaf_rn(recv[2]._vf.z, -0.5, tmp._vf.z), __fmul_rn(recv[2]._vf.w, 0.8660254f));

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = make_uchar2(res._vf2.x, res._vf2.y);
    }
}



__global__ void 
decx::dsp::fft::GPUK::cu_FFT2_R3_C2R_cplxf_fp32(const float4* __restrict src,
                                              float2* __restrict dst,
                                              const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                              const uint32_t _pitchsrc_v2,
                                              const uint32_t _pitchdst_v2)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 3;

    decx::utils::_cuda_vec128 recv[3], tmp;
    decx::utils::_cuda_vec64 res;
    de::CPf W;

    uint32_t _FFT_domain_dex = tidy;
    uint32_t _warp_loc_id = tidy % _kernel_info._store_pitch;

    if (tidy < _Bops_num && tidx < _pitchsrc_v2)
    {
#pragma unroll 3
        for (uint8_t i = 0; i < 3; ++i) {
            recv[i]._vf = src[_FFT_domain_dex * _pitchsrc_v2 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    W.construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)_warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[1]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[1]._vf, W);

    W.construct_with_phase(__fmul_rn(Four_Pi, __fdividef((float)_warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[2]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[2]._vf, W);

    _FFT_domain_dex = (tidy / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + _warp_loc_id;

    if (tidy < _Bops_num && tidx < _pitchdst_v2)
    {
        res._vf2.x = __fadd_rn(__fadd_rn(recv[0]._vf.x, recv[1]._vf.x), recv[2]._vf.x);
        res._vf2.y = __fadd_rn(__fadd_rn(recv[0]._vf.z, recv[1]._vf.z), recv[2]._vf.z);

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf2;
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1(recv[1]._vf, de::CPf(-0.5, 0.8660254f), recv[0]._vf);
        res._vf2.x = __fsub_rn(__fmaf_rn(recv[2]._vf.x, -0.5, tmp._vf.x), __fmul_rn(recv[2]._vf.y, -0.8660254f));
        res._vf2.y = __fsub_rn(__fmaf_rn(recv[2]._vf.z, -0.5, tmp._vf.z), __fmul_rn(recv[2]._vf.w, -0.8660254f));

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf2;
        _FFT_domain_dex += _kernel_info._store_pitch;

        tmp._vf = decx::dsp::fft::GPUK::_complex_2fma1(recv[1]._vf, de::CPf(-0.5, -0.8660254f), recv[0]._vf);
        res._vf2.x = __fsub_rn(__fmaf_rn(recv[2]._vf.x, -0.5, tmp._vf.x), __fmul_rn(recv[2]._vf.y, 0.8660254f));
        res._vf2.y = __fsub_rn(__fmaf_rn(recv[2]._vf.z, -0.5, tmp._vf.z), __fmul_rn(recv[2]._vf.w, 0.8660254f));

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf2;
    }
}