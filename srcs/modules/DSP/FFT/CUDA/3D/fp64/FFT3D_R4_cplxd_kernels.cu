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


#include "../FFT3D_kernels.cuh"



template <bool _div> __global__ void 
decx::dsp::fft::GPUK::cu_FFT3_R4_1st_C2C_cplxd(const double2* __restrict src,
                                               double2* __restrict dst,
                                               const uint32_t _signal_len,
                                               const uint2 _signal_pitch,
                                               const uint32_t _pitchsrc_v1,
                                               const uint32_t _pitchdst_v1,
                                               const uint32_t _paral,
                                               const uint64_t _div_length)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _signal_len / 4;

    decx::utils::_cuda_vec128 recv[4];
    decx::utils::_cuda_vec128 res, tmp1, tmp2;

    const double _numer = __ull2double_rn(_div_length ? _div_length : _signal_len);
    uint32_t _FFT_dex_loc = (tidy % _Bops_num);
    const uint32_t _lane_id = tidy / _Bops_num;

    if (tidy < _Bops_num * _paral && tidx < _pitchsrc_v1)
    {
#pragma unroll 4
        for (uint8_t i = 0; i < 4; ++i) {
            recv[i]._vd = src[(_FFT_dex_loc + _lane_id * _signal_pitch.x) * _pitchsrc_v1 + tidx];
            if (_div) { recv[i]._vd = decx::utils::cuda::__double_div2_1(recv[i]._vd, _numer); }
            _FFT_dex_loc += _Bops_num;
        }
    }

    _FFT_dex_loc = (tidy % _Bops_num) * 4 + _lane_id * _signal_pitch.y;

    if (tidy < _Bops_num * _paral && tidx < _pitchdst_v1) 
    {
        // Calculate the first and third output
        tmp1._vd = decx::utils::cuda::__double_add2(recv[0]._vd, recv[2]._vd);
        tmp2._vd = decx::utils::cuda::__double_add2(recv[1]._vd, recv[3]._vd);

        // Store the first output
        res._vd = decx::utils::cuda::__double_add2(tmp1._vd, tmp2._vd);
        dst[_FFT_dex_loc * _pitchdst_v1 + tidx] = res._vd;
        _FFT_dex_loc += 2;
        
        // Store the third output
        res._vd = decx::utils::cuda::__double_sub2(tmp1._vd, tmp2._vd);
        dst[_FFT_dex_loc * _pitchdst_v1 + tidx] = res._vd;
        --_FFT_dex_loc;

        // Calculate and store the second output
        res._cplxd.real = __dsub_rn(__dadd_rn(recv[0]._cplxd.real, recv[3]._cplxd.image), 
                                    __dadd_rn(recv[1]._cplxd.image, recv[2]._cplxd.real));
        res._cplxd.image = __dadd_rn(__dsub_rn(recv[0]._cplxd.image, recv[2]._cplxd.image), 
                                     __dsub_rn(recv[1]._cplxd.real, recv[3]._cplxd.real));
        
        dst[_FFT_dex_loc * _pitchdst_v1 + tidx] = res._vd;
        _FFT_dex_loc += 2;

        res._cplxd.real = __dsub_rn(__dadd_rn(recv[0]._cplxd.real, recv[1]._cplxd.image), 
                                    __dadd_rn(recv[2]._cplxd.real, recv[3]._cplxd.image));
        res._cplxd.image = __dsub_rn(__dsub_rn(recv[0]._cplxd.image, recv[1]._cplxd.real), 
                                     __dsub_rn(recv[2]._cplxd.image, recv[3]._cplxd.real));
        
        dst[_FFT_dex_loc * _pitchdst_v1 + tidx] = res._vd;
    }
}

template __global__ void
decx::dsp::fft::GPUK::cu_FFT3_R4_1st_C2C_cplxd<true>(const double2* __restrict, double2* __restrict,
    const uint32_t, const uint2, const uint32_t, const uint32_t, const uint32_t, const uint64_t);

template __global__ void
decx::dsp::fft::GPUK::cu_FFT3_R4_1st_C2C_cplxd<false>(const double2* __restrict, double2* __restrict,
    const uint32_t, const uint2, const uint32_t, const uint32_t, const uint32_t, const uint64_t);



__global__ void 
decx::dsp::fft::GPUK::cu_FFT3_R4_C2C_cplxd(const double2* __restrict src,
                                           double2* __restrict dst,
                                           const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                           const uint32_t _signal_pitch,
                                           const uint32_t _pitchsrc_v1,
                                           const uint32_t _pitchdst_v1,
                                           const uint32_t _paral)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 4;

    decx::utils::_cuda_vec128 recv[4];
    decx::utils::_cuda_vec128 res, tmp1, tmp2;

    uint32_t _FFT_dex_loc;
    const uint32_t _lane_id = tidy / _Bops_num;

    de::CPd W;

    if (tidy < _Bops_num * _paral && tidx < _pitchdst_v1)
    {
        _FFT_dex_loc = (tidy % _Bops_num) + _lane_id * _signal_pitch;
#pragma unroll 4
        for (uint8_t i = 0; i < 4; ++i) {
            recv[i]._vd = src[_FFT_dex_loc * _pitchsrc_v1 + tidx];
            _FFT_dex_loc += _Bops_num;
        }
    }

    const uint32_t warp_loc_id = (tidy % _Bops_num) % _kernel_info._store_pitch;

    W.construct_with_phase(__dmul_rn(Two_Pi, __ddiv_rn((double)warp_loc_id, (double)_kernel_info._warp_proc_len)));
    recv[1]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[1]._cplxd, W);

    W.construct_with_phase(__dmul_rn(Four_Pi, __ddiv_rn((double)warp_loc_id, (double)_kernel_info._warp_proc_len)));
    recv[2]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[2]._cplxd, W);

    W.construct_with_phase(__dmul_rn(Six_Pi, __ddiv_rn((double)warp_loc_id, (double)_kernel_info._warp_proc_len)));
    recv[3]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(recv[3]._cplxd, W);

    _FFT_dex_loc = ((tidy % _Bops_num) / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + warp_loc_id 
                 + _lane_id * _signal_pitch;

    if (tidy < _Bops_num * _paral && tidx < _pitchdst_v1)
    {
        // Calculate the first and third output
        tmp1._vd = decx::utils::cuda::__double_add2(recv[0]._vd, recv[2]._vd);
        tmp2._vd = decx::utils::cuda::__double_add2(recv[1]._vd, recv[3]._vd);

        // Store the first output
        res._vd = decx::utils::cuda::__double_add2(tmp1._vd, tmp2._vd);
        dst[_FFT_dex_loc * _pitchdst_v1 + tidx] = res._vd;
        _FFT_dex_loc += (_kernel_info._store_pitch << 1);
        
        // Store the third output
        res._vd = decx::utils::cuda::__double_sub2(tmp1._vd, tmp2._vd);
        dst[_FFT_dex_loc * _pitchdst_v1 + tidx] = res._vd;

        // Calculate and store the second output
        res._cplxd.real = __dsub_rn(__dadd_rn(recv[0]._cplxd.real, recv[3]._cplxd.image), 
                                    __dadd_rn(recv[1]._cplxd.image, recv[2]._cplxd.real));
        res._cplxd.image = __dadd_rn(__dsub_rn(recv[0]._cplxd.image, recv[2]._cplxd.image), 
                                     __dsub_rn(recv[1]._cplxd.real, recv[3]._cplxd.real));
        _FFT_dex_loc -= (_kernel_info._store_pitch);
        dst[_FFT_dex_loc * _pitchdst_v1 + tidx] = res._vd;

        res._cplxd.real = __dsub_rn(__dadd_rn(recv[0]._cplxd.real, recv[1]._cplxd.image), 
                                    __dadd_rn(recv[2]._cplxd.real, recv[3]._cplxd.image));
        res._cplxd.image = __dsub_rn(__dsub_rn(recv[0]._cplxd.image, recv[1]._cplxd.real), 
                                     __dsub_rn(recv[2]._cplxd.image, recv[3]._cplxd.real));
        _FFT_dex_loc += (_kernel_info._store_pitch << 1);
        dst[_FFT_dex_loc * _pitchdst_v1 + tidx] = res._vd;
    }
}
