/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "FFT3D_kernels.cuh"


template <bool _div> __global__ void 
decx::dsp::fft::GPUK::cu_FFT3_R4_1st_C2C_cplxf(const float4* __restrict src,
                                               float4* __restrict dst,
                                               const uint32_t _signal_len,
                                               const uint2 _signal_pitch,
                                               const uint32_t _pitchsrc_v2,
                                               const uint32_t _pitchdst_v2,
                                               const uint32_t _paral,
                                               const uint64_t _div_length)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _signal_len / 4;

    decx::utils::_cuda_vec128 recv[4];
    decx::utils::_cuda_vec128 res, tmp1, tmp2;

    const float _numer = __ull2float_rn(_div_length ? _div_length : _signal_len);
    uint32_t _FFT_dex_loc = (tidy % _Bops_num);
    const uint32_t _lane_id = tidy / _Bops_num;

    if (tidy < _Bops_num * _paral && tidx < _pitchsrc_v2)
    {
#pragma unroll 4
        for (uint8_t i = 0; i < 4; ++i) {
            recv[i]._vf = src[(_FFT_dex_loc + _lane_id * _signal_pitch.x) * _pitchsrc_v2 + tidx];
            if (_div) { recv[i]._vf = decx::utils::cuda::__float_div4_1(recv[i]._vf, _numer); }
            _FFT_dex_loc += _Bops_num;
        }
    }

    _FFT_dex_loc = (tidy % _Bops_num) * 4 + _lane_id * _signal_pitch.y;

    if (tidy < _Bops_num * _paral && tidx < _pitchdst_v2) 
    {
        // Calculate the first and third output
        tmp1._vf = decx::utils::cuda::__float_add4(recv[0]._vf, recv[2]._vf);
        tmp2._vf = decx::utils::cuda::__float_add4(recv[1]._vf, recv[3]._vf);

        // Store the first output
        res._vf = decx::utils::cuda::__float_add4(tmp1._vf, tmp2._vf);
        dst[_FFT_dex_loc * _pitchdst_v2 + tidx] = res._vf;
        _FFT_dex_loc += 2;
        
        // Store the third output
        res._vf = decx::utils::cuda::__float_sub4(tmp1._vf, tmp2._vf);
        dst[_FFT_dex_loc * _pitchdst_v2 + tidx] = res._vf;
        --_FFT_dex_loc;

        // Calculate and store the second output
        res._arrcplxf2[0].real = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[0].real, recv[3]._arrcplxf2[0].image), 
                                           __fadd_rn(recv[1]._arrcplxf2[0].image, recv[2]._arrcplxf2[0].real));
        res._arrcplxf2[0].image = __fadd_rn(__fsub_rn(recv[0]._arrcplxf2[0].image, recv[2]._arrcplxf2[0].image), 
                                            __fsub_rn(recv[1]._arrcplxf2[0].real, recv[3]._arrcplxf2[0].real));
        res._arrcplxf2[1].real = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[1].real, recv[3]._arrcplxf2[1].image), 
                                           __fadd_rn(recv[1]._arrcplxf2[1].image, recv[2]._arrcplxf2[1].real));
        res._arrcplxf2[1].image = __fadd_rn(__fsub_rn(recv[0]._arrcplxf2[1].image, recv[2]._arrcplxf2[1].image), 
                                            __fsub_rn(recv[1]._arrcplxf2[1].real, recv[3]._arrcplxf2[1].real));
        dst[_FFT_dex_loc * _pitchdst_v2 + tidx] = res._vf;
        _FFT_dex_loc += 2;

        res._arrcplxf2[0].real = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[0].real, recv[1]._arrcplxf2[0].image), 
                                           __fadd_rn(recv[2]._arrcplxf2[0].real, recv[3]._arrcplxf2[0].image));
        res._arrcplxf2[0].image = __fsub_rn(__fsub_rn(recv[0]._arrcplxf2[0].image, recv[1]._arrcplxf2[0].real), 
                                            __fsub_rn(recv[2]._arrcplxf2[0].image, recv[3]._arrcplxf2[0].real));
        res._arrcplxf2[1].real = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[1].real, recv[1]._arrcplxf2[1].image), 
                                           __fadd_rn(recv[2]._arrcplxf2[1].real, recv[3]._arrcplxf2[1].image));
        res._arrcplxf2[1].image = __fsub_rn(__fsub_rn(recv[0]._arrcplxf2[1].image, recv[1]._arrcplxf2[1].real), 
                                            __fsub_rn(recv[2]._arrcplxf2[1].image, recv[3]._arrcplxf2[1].real));
        dst[_FFT_dex_loc * _pitchdst_v2 + tidx] = res._vf;
    }
}

template __global__ void
decx::dsp::fft::GPUK::cu_FFT3_R4_1st_C2C_cplxf<true>(const float4* __restrict, float4* __restrict,
    const uint32_t, const uint2, const uint32_t, const uint32_t, const uint32_t, const uint64_t);

template __global__ void
decx::dsp::fft::GPUK::cu_FFT3_R4_1st_C2C_cplxf<false>(const float4* __restrict, float4* __restrict,
    const uint32_t, const uint2, const uint32_t, const uint32_t, const uint32_t, const uint64_t);



__global__ void 
decx::dsp::fft::GPUK::cu_FFT3_R4_C2C_cplxf(const float4* __restrict src,
                                           float4* __restrict dst,
                                           const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                           const uint32_t _signal_pitch,
                                           const uint32_t _pitchsrc_v2,
                                           const uint32_t _pitchdst_v2,
                                           const uint32_t _paral)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 4;

    decx::utils::_cuda_vec128 recv[4];
    decx::utils::_cuda_vec128 res, tmp1, tmp2;

    uint32_t _FFT_dex_loc;
    const uint32_t _lane_id = tidy / _Bops_num;

    decx::utils::_cuda_vec64 W;

    if (tidy < _Bops_num * _paral && tidx < _pitchdst_v2)
    {
        _FFT_dex_loc = (tidy % _Bops_num) + _lane_id * _signal_pitch;
#pragma unroll 4
        for (uint8_t i = 0; i < 4; ++i) {
            recv[i]._vf = src[_FFT_dex_loc * _pitchsrc_v2 + tidx];
            _FFT_dex_loc += _Bops_num;
        }
    }

    const uint32_t warp_loc_id = (tidy % _Bops_num) % _kernel_info._store_pitch;

    W._cplxf32.dev_construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[1]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[1]._vf, W._cplxf32);

    W._cplxf32.dev_construct_with_phase(__fmul_rn(Four_Pi, __fdividef((float)warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[2]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[2]._vf, W._cplxf32);

    W._cplxf32.dev_construct_with_phase(__fmul_rn(Six_Pi, __fdividef((float)warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[3]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[3]._vf, W._cplxf32);

    _FFT_dex_loc = ((tidy % _Bops_num) / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + warp_loc_id 
                 + _lane_id * _signal_pitch;

    if (tidy < _Bops_num * _paral && tidx < _pitchdst_v2)
    {
        // Calculate the first and third output
        tmp1._vf = decx::utils::cuda::__float_add4(recv[0]._vf, recv[2]._vf);
        tmp2._vf = decx::utils::cuda::__float_add4(recv[1]._vf, recv[3]._vf);

        // Store the first output
        res._vf = decx::utils::cuda::__float_add4(tmp1._vf, tmp2._vf);
        dst[_FFT_dex_loc * _pitchdst_v2 + tidx] = res._vf;
        _FFT_dex_loc += (_kernel_info._store_pitch << 1);
        
        // Store the third output
        res._vf = decx::utils::cuda::__float_sub4(tmp1._vf, tmp2._vf);
        dst[_FFT_dex_loc * _pitchdst_v2 + tidx] = res._vf;

        // Calculate and store the second output
        res._arrcplxf2[0].real = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[0].real, recv[3]._arrcplxf2[0].image), 
                                           __fadd_rn(recv[1]._arrcplxf2[0].image, recv[2]._arrcplxf2[0].real));
        res._arrcplxf2[0].image = __fadd_rn(__fsub_rn(recv[0]._arrcplxf2[0].image, recv[2]._arrcplxf2[0].image), 
                                            __fsub_rn(recv[1]._arrcplxf2[0].real, recv[3]._arrcplxf2[0].real));
        res._arrcplxf2[1].real = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[1].real, recv[3]._arrcplxf2[1].image), 
                                           __fadd_rn(recv[1]._arrcplxf2[1].image, recv[2]._arrcplxf2[1].real));
        res._arrcplxf2[1].image = __fadd_rn(__fsub_rn(recv[0]._arrcplxf2[1].image, recv[2]._arrcplxf2[1].image), 
                                            __fsub_rn(recv[1]._arrcplxf2[1].real, recv[3]._arrcplxf2[1].real));
        _FFT_dex_loc -= (_kernel_info._store_pitch);
        dst[_FFT_dex_loc * _pitchdst_v2 + tidx] = res._vf;

        res._arrcplxf2[0].real = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[0].real, recv[1]._arrcplxf2[0].image), 
                                           __fadd_rn(recv[2]._arrcplxf2[0].real, recv[3]._arrcplxf2[0].image));
        res._arrcplxf2[0].image = __fsub_rn(__fsub_rn(recv[0]._arrcplxf2[0].image, recv[1]._arrcplxf2[0].real), 
                                            __fsub_rn(recv[2]._arrcplxf2[0].image, recv[3]._arrcplxf2[0].real));
        res._arrcplxf2[1].real = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[1].real, recv[1]._arrcplxf2[1].image), 
                                           __fadd_rn(recv[2]._arrcplxf2[1].real, recv[3]._arrcplxf2[1].image));
        res._arrcplxf2[1].image = __fsub_rn(__fsub_rn(recv[0]._arrcplxf2[1].image, recv[1]._arrcplxf2[1].real), 
                                            __fsub_rn(recv[2]._arrcplxf2[1].image, recv[3]._arrcplxf2[1].real));

        _FFT_dex_loc += (_kernel_info._store_pitch << 1);
        dst[_FFT_dex_loc * _pitchdst_v2 + tidx] = res._vf;
    }
}