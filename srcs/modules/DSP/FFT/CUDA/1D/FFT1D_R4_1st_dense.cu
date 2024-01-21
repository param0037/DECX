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


// [32 * 2, 8] = [64, 8]
__global__ void 
decx::dsp::fft::GPUK::cu_FFT2D_R4_1st_cplxf_R2C_dense(const float* __restrict src,
                                                      float2* __restrict dst,
                                                      const uint32_t _signal_len,
                                                      const uint32_t _pitchsrc,
                                                      const uint32_t _pitchdst)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _signal_len / 4;

    float recv[4];
    decx::utils::_cuda_vec64 res;

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
        res._cplxf32.image = 0.f;
        res._cplxf32.real = __fadd_rn(__fadd_rn(recv[0], recv[1]), __fadd_rn(recv[2], recv[3]));
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
        ++_FFT_domain_dex;

        // 2nd
        res._cplxf32.real = __fsub_rn(recv[0], recv[2]);
        res._cplxf32.image = __fsub_rn(recv[1], recv[3]);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
        ++_FFT_domain_dex;

        // 3rd
        res._cplxf32.image = 0.f;
        res._cplxf32.real = __fadd_rn(__fsub_rn(recv[0], recv[1]), __fsub_rn(recv[2], recv[3]));
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
        ++_FFT_domain_dex;

        // 4th 
        res._cplxf32.real = __fsub_rn(recv[0], recv[2]);
        res._cplxf32.image = __fsub_rn(recv[3], recv[1]);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
    }
}



template<bool _div> __global__ void 
decx::dsp::fft::GPUK::cu_FFT2D_R4_1st_cplxf_C2C_dense(const float2* __restrict src,
                                               float2* __restrict dst,
                                               const uint32_t _signal_len,
                                               const uint32_t _pitchsrc,
                                               const uint32_t _pitchdst,
                                               const uint64_t _div_length)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const float _numer = __ull2float_rn(_div_length ? _div_length : _signal_len);
    const uint32_t _Bops_num = _signal_len / 4;

    decx::utils::_cuda_vec64 recv[4];
    decx::utils::_cuda_vec64 res, tmp1, tmp2;

    uint32_t _FFT_domain_dex = tidy;

    if (tidy < _Bops_num && tidx < _pitchsrc)
    {
#pragma unroll 4
        for (uint8_t i = 0; i < 4; ++i) {
            recv[i]._vf2 = src[_FFT_domain_dex * _pitchsrc + tidx];
            if (_div) { recv[i]._vf2 = decx::utils::cuda::__float_div2_1(recv[i]._vf2, _numer); }
            _FFT_domain_dex += _Bops_num;
        }
    }

    _FFT_domain_dex = tidy * 4;

    if (tidy < _Bops_num && tidx < _pitchdst)
    {
        // Calculate the first and third output
        tmp1._fp64 = decx::utils::cuda::__float_add2(recv[0]._fp64, recv[2]._fp64);
        tmp2._fp64 = decx::utils::cuda::__float_add2(recv[1]._fp64, recv[3]._fp64);

        // Store the first output
        res._fp64 = decx::utils::cuda::__float_add2(tmp1._fp64, tmp2._fp64);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
        _FFT_domain_dex += 2;

        // Store the third output
        res._vf2 = decx::utils::cuda::__float_sub2(tmp1._vf2, tmp2._vf2);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;

        // Calculate and store the second output
        res._cplxf32.real = __fsub_rn(__fadd_rn(recv[0]._cplxf32.real, recv[3]._cplxf32.image),
            __fadd_rn(recv[1]._cplxf32.image, recv[2]._cplxf32.real));
        res._cplxf32.image = __fadd_rn(__fsub_rn(recv[0]._cplxf32.image, recv[2]._cplxf32.image),
            __fsub_rn(recv[1]._cplxf32.real, recv[3]._cplxf32.real));
        --_FFT_domain_dex;
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;

        res._cplxf32.real = __fsub_rn(__fadd_rn(recv[0]._cplxf32.real, recv[1]._cplxf32.image),
            __fadd_rn(recv[2]._cplxf32.real, recv[3]._cplxf32.image));
        res._cplxf32.image = __fsub_rn(__fsub_rn(recv[0]._cplxf32.image, recv[1]._cplxf32.real),
            __fsub_rn(recv[2]._cplxf32.image, recv[3]._cplxf32.real));

        _FFT_domain_dex += 2;
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
    }
}

template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R4_1st_cplxf_C2C_dense<true>(const float2* __restrict, float2* __restrict,
    const uint32_t, const uint32_t, const uint32_t, const uint64_t);

template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R4_1st_cplxf_C2C_dense<false>(const float2* __restrict, float2* __restrict,
    const uint32_t, const uint32_t, const uint32_t, const uint64_t);


template <bool _conj> __global__ void 
decx::dsp::fft::GPUK::cu_FFT2D_R4_end_cplxf_C2C_dense(const float2* __restrict src, 
                                                      float2* __restrict dst,
                                                      const decx::dsp::fft::FKI_4_2DK _kernel_info, 
                                                      const uint32_t _pitchsrc, 
                                                      const uint32_t _pitchdst)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 4;

    decx::utils::_cuda_vec64 recv[4];
    decx::utils::_cuda_vec64 res, tmp1, tmp2;

    uint32_t _FFT_domain_dex, warp_loc_id;

    decx::utils::_cuda_vec64 W;

    warp_loc_id = tidy % _kernel_info._store_pitch;

    if (tidy < _Bops_num && tidx < _pitchdst)
    {
        _FFT_domain_dex = tidy;
#pragma unroll 4
        for (uint8_t i = 0; i < 4; ++i) {
            recv[i]._vf2 = src[_FFT_domain_dex * _pitchsrc + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    W._cplxf32.dev_construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[1]._cplxf32 = decx::dsp::fft::GPUK::_complex_mul_fp32(recv[1]._cplxf32, W._cplxf32);

    W._cplxf32.dev_construct_with_phase(__fmul_rn(Four_Pi, __fdividef((float)warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[2]._cplxf32 = decx::dsp::fft::GPUK::_complex_mul_fp32(recv[2]._cplxf32, W._cplxf32);

    W._cplxf32.dev_construct_with_phase(__fmul_rn(Six_Pi, __fdividef((float)warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[3]._cplxf32 = decx::dsp::fft::GPUK::_complex_mul_fp32(recv[3]._cplxf32, W._cplxf32);

    _FFT_domain_dex = (tidy / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + warp_loc_id;

    if (tidy < _Bops_num && tidx < _pitchdst)
    {
        // Calculate the first and third output
        tmp1._fp64 = decx::utils::cuda::__float_add2(recv[0]._fp64, recv[2]._fp64);
        tmp2._fp64 = decx::utils::cuda::__float_add2(recv[1]._fp64, recv[3]._fp64);

        // Store the first output
        res._fp64 = decx::utils::cuda::__float_add2(tmp1._fp64, tmp2._fp64);
        if (_conj) { res._cplxf32 = decx::dsp::fft::GPUK::_complex_conjugate_fp32(res._cplxf32); }
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
        _FFT_domain_dex += (_kernel_info._store_pitch << 1);
        
        // Store the third output
        res._vf2 = decx::utils::cuda::__float_sub2(tmp1._vf2, tmp2._vf2);
        if (_conj) { res._cplxf32 = decx::dsp::fft::GPUK::_complex_conjugate_fp32(res._cplxf32); }
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;

        // Calculate and store the second output
        res._cplxf32.real = __fsub_rn(__fadd_rn(recv[0]._cplxf32.real, recv[3]._cplxf32.image), 
                                           __fadd_rn(recv[1]._cplxf32.image, recv[2]._cplxf32.real));
        res._cplxf32.image = __fadd_rn(__fsub_rn(recv[0]._cplxf32.image, recv[2]._cplxf32.image), 
                                            __fsub_rn(recv[1]._cplxf32.real, recv[3]._cplxf32.real));
        _FFT_domain_dex -= (_kernel_info._store_pitch);
        if (_conj) { res._cplxf32 = decx::dsp::fft::GPUK::_complex_conjugate_fp32(res._cplxf32); }
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;

        res._cplxf32.real = __fsub_rn(__fadd_rn(recv[0]._cplxf32.real, recv[1]._cplxf32.image), 
                                           __fadd_rn(recv[2]._cplxf32.real, recv[3]._cplxf32.image));
        res._cplxf32.image = __fsub_rn(__fsub_rn(recv[0]._cplxf32.image, recv[1]._cplxf32.real), 
                                            __fsub_rn(recv[2]._cplxf32.image, recv[3]._cplxf32.real));

        _FFT_domain_dex += (_kernel_info._store_pitch << 1);
        if (_conj) { res._cplxf32 = decx::dsp::fft::GPUK::_complex_conjugate_fp32(res._cplxf32); }
        dst[_FFT_domain_dex * _pitchdst + tidx] = res._vf2;
    }
}

template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R4_end_cplxf_C2C_dense<true>(const float2* __restrict, float2* __restrict,
    const decx::dsp::fft::FKI_4_2DK, const uint32_t, const uint32_t);

template __global__ void decx::dsp::fft::GPUK::cu_FFT2D_R4_end_cplxf_C2C_dense<false>(const float2* __restrict, float2* __restrict,
    const decx::dsp::fft::FKI_4_2DK, const uint32_t, const uint32_t);




__global__ void 
decx::dsp::fft::GPUK::cu_FFT2D_R4_end_cplxf_C2R_dense(const float2* __restrict src,
                                              float* __restrict dst,
                                              const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                              const uint32_t _pitchsrc,
                                              const uint32_t _pitchdst)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 4;

    decx::utils::_cuda_vec64 recv[4], tmp;
    decx::utils::_cuda_vec64 tmp1, tmp2;
    float res;
    de::CPf W;

    uint32_t _FFT_domain_dex = tidy;
    uint32_t _warp_loc_id = tidy % _kernel_info._store_pitch;

    if (tidy < _Bops_num && tidx < _pitchsrc)
    {
#pragma unroll 4
        for (uint8_t i = 0; i < 4; ++i) {
            recv[i]._vf2 = src[_FFT_domain_dex * _pitchsrc + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    W.dev_construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)_warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[1]._cplxf32 = decx::dsp::fft::GPUK::_complex_mul_fp32(recv[1]._cplxf32, W);

    W.dev_construct_with_phase(__fmul_rn(Four_Pi, __fdividef((float)_warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[2]._cplxf32 = decx::dsp::fft::GPUK::_complex_mul_fp32(recv[2]._cplxf32, W);

    W.dev_construct_with_phase(__fmul_rn(Six_Pi, __fdividef((float)_warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[3]._cplxf32 = decx::dsp::fft::GPUK::_complex_mul_fp32(recv[3]._cplxf32, W);

    _FFT_domain_dex = (tidy / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + _warp_loc_id;

    if (tidy < _Bops_num && tidx < _pitchdst)
    {
        // Calculate the first and third output
        tmp1._fp64 = decx::utils::cuda::__float_add2(recv[0]._fp64, recv[2]._fp64);
        tmp2._fp64 = decx::utils::cuda::__float_add2(recv[1]._fp64, recv[3]._fp64);

        // Store the first output
        res = __fadd_rn(tmp1._vf2.x, tmp2._vf2.x);

        dst[_FFT_domain_dex * _pitchdst + tidx] = res;
        _FFT_domain_dex += (_kernel_info._store_pitch << 1);
        
        // Store the third output
        res = __fsub_rn(tmp1._vf2.x, tmp2._vf2.x);

        dst[_FFT_domain_dex * _pitchdst + tidx] = res;

        // Calculate and store the second output
        res = __fsub_rn(__fadd_rn(recv[0]._cplxf32.real, recv[3]._cplxf32.image), 
                        __fadd_rn(recv[1]._cplxf32.image, recv[2]._cplxf32.real));
        _FFT_domain_dex -= (_kernel_info._store_pitch);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res;

        res = __fsub_rn(__fadd_rn(recv[0]._cplxf32.real, recv[1]._cplxf32.image), 
                        __fadd_rn(recv[2]._cplxf32.real, recv[3]._cplxf32.image));

        _FFT_domain_dex += (_kernel_info._store_pitch << 1);
        dst[_FFT_domain_dex * _pitchdst + tidx] = res;
    }
}