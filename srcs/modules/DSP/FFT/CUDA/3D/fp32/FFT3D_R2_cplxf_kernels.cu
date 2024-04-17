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


template<bool _div> __global__ void 
decx::dsp::fft::GPUK::cu_FFT3_R2_1st_C2C_cplxf(const float4* __restrict src, 
                                               float4* __restrict dst, 
                                               const uint32_t _signal_len,
                                               const uint2 _signal_pitch, 
                                               const uint32_t _pitchsrc_v2, 
                                               const uint32_t _pitchdst_v2, 
                                               const uint32_t _paral)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _signal_len / 2;

    decx::utils::_cuda_vec128 recv[2];
    decx::utils::_cuda_vec128 res;

    uint32_t _FFT_domain_dex = tidy % _Bops_num;
    const uint32_t _lane_id = tidy / _Bops_num;

    if (tidy < _Bops_num * _paral && tidx < _pitchsrc_v2)
    {
#pragma unroll 2
        for (uint8_t i = 0; i < 2; ++i) {
            recv[i]._vf = src[(_FFT_domain_dex + _lane_id * _signal_pitch.x) * _pitchsrc_v2 + tidx];
            if (_div) { recv[i]._vf = decx::utils::cuda::__float_div4_1(recv[i]._vf, _signal_len); }
            _FFT_domain_dex += _Bops_num;
        }
    }

    _FFT_domain_dex = (tidy % _Bops_num) * 2 + _lane_id * _signal_pitch.y;

    if (tidy < _Bops_num * _paral && tidx < _pitchdst_v2) 
    {
        res._vf = decx::utils::cuda::__float_add4(recv[0]._vf, recv[1]._vf);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        res._vf = decx::utils::cuda::__float_sub4(recv[0]._vf, recv[1]._vf);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
    }
}

template __global__ void
decx::dsp::fft::GPUK::cu_FFT3_R2_1st_C2C_cplxf<true>(const float4* __restrict, float4* __restrict,
    const uint32_t, const uint2, const uint32_t, const uint32_t, const uint32_t);

template __global__ void
decx::dsp::fft::GPUK::cu_FFT3_R2_1st_C2C_cplxf<false>(const float4* __restrict, float4* __restrict,
    const uint32_t, const uint2, const uint32_t, const uint32_t, const uint32_t);



__global__ void 
decx::dsp::fft::GPUK::cu_FFT3_R2_C2C_cplxf(const float4* __restrict src, 
                                           float4* __restrict dst, 
                                           const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                           const uint32_t signal_pitch, 
                                           const uint32_t _pitchsrc_v2, 
                                           const uint32_t _pitchdst_v2, 
                                           const uint32_t _paral)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 2;

    decx::utils::_cuda_vec128 recv[2];
    decx::utils::_cuda_vec128 res;

    uint32_t _FFT_domain_dex, warp_loc_id;
    const uint32_t _lane_id = tidy / _Bops_num;

    decx::utils::_cuda_vec64 W;

    if (tidy < _Bops_num * _paral && tidx < _pitchdst_v2)
    {
        _FFT_domain_dex = (tidy % _Bops_num) + _lane_id * signal_pitch;
#pragma unroll 2
        for (uint8_t i = 0; i < 2; ++i) {
            recv[i]._vf = src[_FFT_domain_dex * _pitchsrc_v2 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    warp_loc_id = (tidy % _Bops_num) % _kernel_info._store_pitch;

    W._cplxf32.construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[1]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[1]._vf, W._cplxf32);

    _FFT_domain_dex = ((tidy % _Bops_num) / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + warp_loc_id
        + _lane_id * signal_pitch;

    if (tidy < _Bops_num * _paral && tidx < _pitchdst_v2)
    {
        res._vf = decx::utils::cuda::__float_add4(recv[0]._vf, recv[1]._vf);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        _FFT_domain_dex += _kernel_info._store_pitch;

        res._vf = decx::utils::cuda::__float_sub4(recv[0]._vf, recv[1]._vf);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
    }
}
