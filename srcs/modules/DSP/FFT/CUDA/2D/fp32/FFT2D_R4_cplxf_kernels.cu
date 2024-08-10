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
__global__ void 
decx::dsp::fft::GPUK::cu_FFT2_R4_1st_R2C_cplxf(const float2* __restrict src,
                                               float4* __restrict dst,
                                               const uint32_t _signal_len,
                                               const uint32_t _pitchsrc_v2,
                                               const uint32_t _pitchdst_v2)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _signal_len / 4;

    decx::utils::_cuda_vec64 recv[4];
    decx::utils::_cuda_vec128 res;

    uint32_t _FFT_domain_dex = tidy;

    if (tidy < _Bops_num && tidx < _pitchsrc_v2)
    {
#pragma unroll 4
        for (uint8_t i = 0; i < 4; ++i) {
            recv[i]._vf2 = src[_FFT_domain_dex * _pitchsrc_v2 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    _FFT_domain_dex = tidy * 4;

    if (tidy < _Bops_num && tidx < _pitchdst_v2)
    {
        // 1st
        res._vf = decx::utils::vec4_set1_fp32(0.f);
        res._arrcplxf2[0].real = __fadd_rn(__fadd_rn(recv[0]._vf2.x, recv[1]._vf2.x),
                                              __fadd_rn(recv[2]._vf2.x, recv[3]._vf2.x));
        res._arrcplxf2[1].real = __fadd_rn(__fadd_rn(recv[0]._vf2.y, recv[1]._vf2.y),
                                              __fadd_rn(recv[2]._vf2.y, recv[3]._vf2.y));
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        // 2nd
        res._arrcplxf2[0].real = __fsub_rn(recv[0]._vf2.x, recv[2]._vf2.x);
        res._arrcplxf2[1].real = __fsub_rn(recv[0]._vf2.y, recv[2]._vf2.y);

        res._arrcplxf2[0].image = __fsub_rn(recv[1]._vf2.x, recv[3]._vf2.x);
        res._arrcplxf2[1].image = __fsub_rn(recv[1]._vf2.y, recv[3]._vf2.y);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        // 3rd
        res._vf = decx::utils::vec4_set1_fp32(0.f);
        res._arrcplxf2[0].real = __fadd_rn(__fsub_rn(recv[0]._vf2.x, recv[1]._vf2.x), __fsub_rn(recv[2]._vf2.x, recv[3]._vf2.x));
        res._arrcplxf2[1].real = __fadd_rn(__fsub_rn(recv[0]._vf2.y, recv[1]._vf2.y), __fsub_rn(recv[2]._vf2.y, recv[3]._vf2.y));
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        // 4th 
        res._arrcplxf2[0].real = __fsub_rn(recv[0]._vf2.x, recv[2]._vf2.x);
        res._arrcplxf2[1].real = __fsub_rn(recv[0]._vf2.y, recv[2]._vf2.y);

        res._arrcplxf2[0].image = __fsub_rn(recv[3]._vf2.x, recv[1]._vf2.x);
        res._arrcplxf2[1].image = __fsub_rn(recv[3]._vf2.y, recv[1]._vf2.y);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
    }
}



// [32 * 2, 8] = [64, 8]
__global__ void 
decx::dsp::fft::GPUK::cu_FFT2_R4_1st_R2C_uc8_cplxf(const ushort* __restrict src,
                                               float4* __restrict dst,
                                               const uint32_t _signal_len,
                                               const uint32_t _pitchsrc_v2,
                                               const uint32_t _pitchdst_v2)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _signal_len / 4;

    uchar2 recv[4];
    decx::utils::_cuda_vec128 res;

    uint32_t _FFT_domain_dex = tidy;

    if (tidy < _Bops_num && tidx < _pitchsrc_v2)
    {
#pragma unroll 4
        for (uint8_t i = 0; i < 4; ++i) {
            *((ushort*)&recv[i]) = src[_FFT_domain_dex * _pitchsrc_v2 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    _FFT_domain_dex = tidy * 4;

    if (tidy < _Bops_num && tidx < _pitchdst_v2)
    {
        // 1st
        res._vf = decx::utils::vec4_set1_fp32(0.f);
        res._arrcplxf2[0].real = __fadd_rn(__fadd_rn(recv[0].x, recv[1].x),
                                              __fadd_rn(recv[2].x, recv[3].x));
        res._arrcplxf2[1].real = __fadd_rn(__fadd_rn(recv[0].y, recv[1].y),
                                              __fadd_rn(recv[2].y, recv[3].y));
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        // 2nd
        res._arrcplxf2[0].real = __fsub_rn(recv[0].x, recv[2].x);
        res._arrcplxf2[1].real = __fsub_rn(recv[0].y, recv[2].y);

        res._arrcplxf2[0].image = __fsub_rn(recv[1].x, recv[3].x);
        res._arrcplxf2[1].image = __fsub_rn(recv[1].y, recv[3].y);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        // 3rd
        res._vf = decx::utils::vec4_set1_fp32(0.f);
        res._arrcplxf2[0].real = __fadd_rn(__fsub_rn(recv[0].x, recv[1].x), __fsub_rn(recv[2].x, recv[3].x));
        res._arrcplxf2[1].real = __fadd_rn(__fsub_rn(recv[0].y, recv[1].y), __fsub_rn(recv[2].y, recv[3].y));
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        ++_FFT_domain_dex;

        // 4th 
        res._arrcplxf2[0].real = __fsub_rn(recv[0].x, recv[2].x);
        res._arrcplxf2[1].real = __fsub_rn(recv[0].y, recv[2].y);

        res._arrcplxf2[0].image = __fsub_rn(recv[3].x, recv[1].x);
        res._arrcplxf2[1].image = __fsub_rn(recv[3].y, recv[1].y);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
    }
}



template<bool _div> __global__ void 
decx::dsp::fft::GPUK::cu_FFT2_R4_1st_C2C_cplxf(const float4* __restrict src,
                                               float4* __restrict dst,
                                               const uint32_t _signal_len,
                                               const uint32_t _pitchsrc_v2,
                                               const uint32_t _pitchdst_v2,
                                               const uint64_t _div_length)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _signal_len / 4;

    decx::utils::_cuda_vec128 recv[4];
    decx::utils::_cuda_vec128 res, tmp1, tmp2;

    const float _numer = __ull2float_rn(_div_length ? _div_length : _signal_len);
    uint32_t _FFT_domain_dex = tidy;

    if (tidy < _Bops_num && tidx < _pitchsrc_v2)
    {
#pragma unroll 4
        for (uint8_t i = 0; i < 4; ++i) {
            recv[i]._vf = src[_FFT_domain_dex * _pitchsrc_v2 + tidx];
            if (_div) { recv[i]._vf = decx::utils::cuda::__float_div4_1(recv[i]._vf, _numer); }
            _FFT_domain_dex += _Bops_num;
        }
    }

    _FFT_domain_dex = tidy * 4;

    if (tidy < _Bops_num && tidx < _pitchdst_v2) 
    {
        // Calculate the first and third output
        tmp1._vf = decx::utils::cuda::__float_add4(recv[0]._vf, recv[2]._vf);
        tmp2._vf = decx::utils::cuda::__float_add4(recv[1]._vf, recv[3]._vf);

        // Store the first output
        res._vf = decx::utils::cuda::__float_add4(tmp1._vf, tmp2._vf);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        _FFT_domain_dex += 2;
        
        // Store the third output
        res._vf = decx::utils::cuda::__float_sub4(tmp1._vf, tmp2._vf);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        --_FFT_domain_dex;

        // Calculate and store the second output
        res._arrcplxf2[0].real = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[0].real, recv[3]._arrcplxf2[0].image), 
                                           __fadd_rn(recv[1]._arrcplxf2[0].image, recv[2]._arrcplxf2[0].real));
        res._arrcplxf2[0].image = __fadd_rn(__fsub_rn(recv[0]._arrcplxf2[0].image, recv[2]._arrcplxf2[0].image), 
                                            __fsub_rn(recv[1]._arrcplxf2[0].real, recv[3]._arrcplxf2[0].real));
        res._arrcplxf2[1].real = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[1].real, recv[3]._arrcplxf2[1].image), 
                                           __fadd_rn(recv[1]._arrcplxf2[1].image, recv[2]._arrcplxf2[1].real));
        res._arrcplxf2[1].image = __fadd_rn(__fsub_rn(recv[0]._arrcplxf2[1].image, recv[2]._arrcplxf2[1].image), 
                                            __fsub_rn(recv[1]._arrcplxf2[1].real, recv[3]._arrcplxf2[1].real));
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        _FFT_domain_dex += 2;

        res._arrcplxf2[0].real = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[0].real, recv[1]._arrcplxf2[0].image), 
                                           __fadd_rn(recv[2]._arrcplxf2[0].real, recv[3]._arrcplxf2[0].image));
        res._arrcplxf2[0].image = __fsub_rn(__fsub_rn(recv[0]._arrcplxf2[0].image, recv[1]._arrcplxf2[0].real), 
                                            __fsub_rn(recv[2]._arrcplxf2[0].image, recv[3]._arrcplxf2[0].real));
        res._arrcplxf2[1].real = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[1].real, recv[1]._arrcplxf2[1].image), 
                                           __fadd_rn(recv[2]._arrcplxf2[1].real, recv[3]._arrcplxf2[1].image));
        res._arrcplxf2[1].image = __fsub_rn(__fsub_rn(recv[0]._arrcplxf2[1].image, recv[1]._arrcplxf2[1].real), 
                                            __fsub_rn(recv[2]._arrcplxf2[1].image, recv[3]._arrcplxf2[1].real));
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
    }
}

template __global__ void
decx::dsp::fft::GPUK::cu_FFT2_R4_1st_C2C_cplxf<true>(const float4* __restrict, float4* __restrict,
    const uint32_t, const uint32_t, const uint32_t, const uint64_t _div_length);

template __global__ void
decx::dsp::fft::GPUK::cu_FFT2_R4_1st_C2C_cplxf<false>(const float4* __restrict, float4* __restrict,
    const uint32_t, const uint32_t, const uint32_t, const uint64_t _div_length);



template <bool _conj> __global__ void 
decx::dsp::fft::GPUK::cu_FFT2_R4_C2C_cplxf(const float4* __restrict src,
                                           float4* __restrict dst,
                                           const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                           const uint32_t _pitchsrc_v2,
                                           const uint32_t _pitchdst_v2)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 4;

    decx::utils::_cuda_vec128 recv[4];
    decx::utils::_cuda_vec128 res, tmp1, tmp2;

    uint32_t _FFT_domain_dex, warp_loc_id;

    decx::utils::_cuda_vec64 W;

    warp_loc_id = tidy % _kernel_info._store_pitch;

    if (tidy < _Bops_num && tidx < _pitchdst_v2)
    {
        _FFT_domain_dex = tidy;
#pragma unroll 4
        for (uint8_t i = 0; i < 4; ++i) {
            recv[i]._vf = src[_FFT_domain_dex * _pitchsrc_v2 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    W._cplxf32.construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[1]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[1]._vf, W._cplxf32);

    W._cplxf32.construct_with_phase(__fmul_rn(Four_Pi, __fdividef((float)warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[2]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[2]._vf, W._cplxf32);

    W._cplxf32.construct_with_phase(__fmul_rn(Six_Pi, __fdividef((float)warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[3]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[3]._vf, W._cplxf32);

    _FFT_domain_dex = (tidy / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + warp_loc_id;

    if (tidy < _Bops_num && tidx < _pitchdst_v2)
    {
        // Calculate the first and third output
        tmp1._vf = decx::utils::cuda::__float_add4(recv[0]._vf, recv[2]._vf);
        tmp2._vf = decx::utils::cuda::__float_add4(recv[1]._vf, recv[3]._vf);

        // Store the first output
        res._vf = decx::utils::cuda::__float_add4(tmp1._vf, tmp2._vf);
        if (_conj) { res = decx::dsp::fft::GPUK::_complex4_conjugate_fp32(res); }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
        _FFT_domain_dex += (_kernel_info._store_pitch << 1);
        
        // Store the third output
        res._vf = decx::utils::cuda::__float_sub4(tmp1._vf, tmp2._vf);
        if (_conj) { res = decx::dsp::fft::GPUK::_complex4_conjugate_fp32(res); }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;

        // Calculate and store the second output
        res._arrcplxf2[0].real = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[0].real, recv[3]._arrcplxf2[0].image), 
                                           __fadd_rn(recv[1]._arrcplxf2[0].image, recv[2]._arrcplxf2[0].real));
        res._arrcplxf2[0].image = __fadd_rn(__fsub_rn(recv[0]._arrcplxf2[0].image, recv[2]._arrcplxf2[0].image), 
                                            __fsub_rn(recv[1]._arrcplxf2[0].real, recv[3]._arrcplxf2[0].real));
        res._arrcplxf2[1].real = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[1].real, recv[3]._arrcplxf2[1].image), 
                                           __fadd_rn(recv[1]._arrcplxf2[1].image, recv[2]._arrcplxf2[1].real));
        res._arrcplxf2[1].image = __fadd_rn(__fsub_rn(recv[0]._arrcplxf2[1].image, recv[2]._arrcplxf2[1].image), 
                                            __fsub_rn(recv[1]._arrcplxf2[1].real, recv[3]._arrcplxf2[1].real));
        _FFT_domain_dex -= (_kernel_info._store_pitch);
        if (_conj) { res = decx::dsp::fft::GPUK::_complex4_conjugate_fp32(res); }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;

        res._arrcplxf2[0].real = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[0].real, recv[1]._arrcplxf2[0].image), 
                                           __fadd_rn(recv[2]._arrcplxf2[0].real, recv[3]._arrcplxf2[0].image));
        res._arrcplxf2[0].image = __fsub_rn(__fsub_rn(recv[0]._arrcplxf2[0].image, recv[1]._arrcplxf2[0].real), 
                                            __fsub_rn(recv[2]._arrcplxf2[0].image, recv[3]._arrcplxf2[0].real));
        res._arrcplxf2[1].real = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[1].real, recv[1]._arrcplxf2[1].image), 
                                           __fadd_rn(recv[2]._arrcplxf2[1].real, recv[3]._arrcplxf2[1].image));
        res._arrcplxf2[1].image = __fsub_rn(__fsub_rn(recv[0]._arrcplxf2[1].image, recv[1]._arrcplxf2[1].real), 
                                            __fsub_rn(recv[2]._arrcplxf2[1].image, recv[3]._arrcplxf2[1].real));

        _FFT_domain_dex += (_kernel_info._store_pitch << 1);
        if (_conj) { res = decx::dsp::fft::GPUK::_complex4_conjugate_fp32(res); }
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf;
    }
}

template __global__ void decx::dsp::fft::GPUK::cu_FFT2_R4_C2C_cplxf<true>(const float4* __restrict, float4* __restrict,
    const decx::dsp::fft::FKI_4_2DK, const uint32_t, const uint32_t);

template __global__ void decx::dsp::fft::GPUK::cu_FFT2_R4_C2C_cplxf<false>(const float4* __restrict, float4* __restrict,
    const decx::dsp::fft::FKI_4_2DK, const uint32_t, const uint32_t);



__global__ void 
decx::dsp::fft::GPUK::cu_FFT2_R4_C2R_cplxf_u8(const float4* __restrict src,
                                              uchar2* __restrict dst,
                                              const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                              const uint32_t _pitchsrc_v2,
                                              const uint32_t _pitchdst_v2)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 4;

    decx::utils::_cuda_vec128 recv[4], tmp;
    decx::utils::_cuda_vec128 tmp1, tmp2;
    decx::utils::_cuda_vec64 res;
    de::CPf W;

    uint32_t _FFT_domain_dex = tidy;
    uint32_t _warp_loc_id = tidy % _kernel_info._store_pitch;

    if (tidy < _Bops_num && tidx < _pitchsrc_v2)
    {
#pragma unroll 4
        for (uint8_t i = 0; i < 4; ++i) {
            recv[i]._vf = src[_FFT_domain_dex * _pitchsrc_v2 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    W.construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)_warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[1]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[1]._vf, W);

    W.construct_with_phase(__fmul_rn(Four_Pi, __fdividef((float)_warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[2]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[2]._vf, W);

    W.construct_with_phase(__fmul_rn(Six_Pi, __fdividef((float)_warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[3]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[3]._vf, W);

    _FFT_domain_dex = (tidy / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + _warp_loc_id;

    if (tidy < _Bops_num && tidx < _pitchdst_v2)
    {
        // Calculate the first and third output
        tmp1._vf = decx::utils::cuda::__float_add4(recv[0]._vf, recv[2]._vf);
        tmp2._vf = decx::utils::cuda::__float_add4(recv[1]._vf, recv[3]._vf);

        // Store the first output
        res._vf2.x = __fadd_rn(tmp1._vf.x, tmp2._vf.x);
        res._vf2.y = __fadd_rn(tmp1._vf.z, tmp2._vf.z);

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = make_uchar2(res._vf2.x, res._vf2.y);
        _FFT_domain_dex += (_kernel_info._store_pitch << 1);
        
        // Store the third output
        res._vf2.x = __fsub_rn(tmp1._vf.x, tmp2._vf.x);
        res._vf2.y = __fsub_rn(tmp1._vf.z, tmp2._vf.z);

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = make_uchar2(res._vf2.x, res._vf2.y);

        // Calculate and store the second output
        res._vf2.x = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[0].real, recv[3]._arrcplxf2[0].image), 
                               __fadd_rn(recv[1]._arrcplxf2[0].image, recv[2]._arrcplxf2[0].real));
        res._vf2.y = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[1].real, recv[3]._arrcplxf2[1].image), 
                               __fadd_rn(recv[1]._arrcplxf2[1].image, recv[2]._arrcplxf2[1].real));
        _FFT_domain_dex -= (_kernel_info._store_pitch);

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = make_uchar2(res._vf2.x, res._vf2.y);

        res._vf2.x = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[0].real, recv[1]._arrcplxf2[0].image), 
                               __fadd_rn(recv[2]._arrcplxf2[0].real, recv[3]._arrcplxf2[0].image));
        res._vf2.y = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[1].real, recv[1]._arrcplxf2[1].image), 
                               __fadd_rn(recv[2]._arrcplxf2[1].real, recv[3]._arrcplxf2[1].image));

        _FFT_domain_dex += (_kernel_info._store_pitch << 1);

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = make_uchar2(res._vf2.x, res._vf2.y);
    }
}




__global__ void 
decx::dsp::fft::GPUK::cu_FFT2_R4_C2R_cplxf_fp32(const float4* __restrict src,
                                              float2* __restrict dst,
                                              const decx::dsp::fft::FKI_4_2DK _kernel_info,
                                              const uint32_t _pitchsrc_v2,
                                              const uint32_t _pitchdst_v2)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _Bops_num = _kernel_info._signal_len / 4;

    decx::utils::_cuda_vec128 recv[4], tmp;
    decx::utils::_cuda_vec128 tmp1, tmp2;
    decx::utils::_cuda_vec64 res;
    de::CPf W;

    uint32_t _FFT_domain_dex = tidy;
    uint32_t _warp_loc_id = tidy % _kernel_info._store_pitch;

    if (tidy < _Bops_num && tidx < _pitchsrc_v2)
    {
#pragma unroll 4
        for (uint8_t i = 0; i < 4; ++i) {
            recv[i]._vf = src[_FFT_domain_dex * _pitchsrc_v2 + tidx];
            _FFT_domain_dex += _Bops_num;
        }
    }

    W.construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)_warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[1]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[1]._vf, W);

    W.construct_with_phase(__fmul_rn(Four_Pi, __fdividef((float)_warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[2]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[2]._vf, W);

    W.construct_with_phase(__fmul_rn(Six_Pi, __fdividef((float)_warp_loc_id, (float)_kernel_info._warp_proc_len)));
    recv[3]._vf = decx::dsp::fft::GPUK::_complex_2mul1_fp32(recv[3]._vf, W);

    _FFT_domain_dex = (tidy / _kernel_info._store_pitch) * _kernel_info._warp_proc_len + _warp_loc_id;

    if (tidy < _Bops_num && tidx < _pitchdst_v2)
    {
        // Calculate the first and third output
        tmp1._vf = decx::utils::cuda::__float_add4(recv[0]._vf, recv[2]._vf);
        tmp2._vf = decx::utils::cuda::__float_add4(recv[1]._vf, recv[3]._vf);

        // Store the first output
        res._vf2.x = __fadd_rn(tmp1._vf.x, tmp2._vf.x);
        res._vf2.y = __fadd_rn(tmp1._vf.z, tmp2._vf.z);

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf2;
        _FFT_domain_dex += (_kernel_info._store_pitch << 1);
        
        // Store the third output
        res._vf2.x = __fsub_rn(tmp1._vf.x, tmp2._vf.x);
        res._vf2.y = __fsub_rn(tmp1._vf.z, tmp2._vf.z);

        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf2;

        // Calculate and store the second output
        res._vf2.x = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[0].real, recv[3]._arrcplxf2[0].image), 
                               __fadd_rn(recv[1]._arrcplxf2[0].image, recv[2]._arrcplxf2[0].real));
        res._vf2.y = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[1].real, recv[3]._arrcplxf2[1].image), 
                               __fadd_rn(recv[1]._arrcplxf2[1].image, recv[2]._arrcplxf2[1].real));
        _FFT_domain_dex -= (_kernel_info._store_pitch);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf2;

        res._vf2.x = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[0].real, recv[1]._arrcplxf2[0].image), 
                               __fadd_rn(recv[2]._arrcplxf2[0].real, recv[3]._arrcplxf2[0].image));
        res._vf2.y = __fsub_rn(__fadd_rn(recv[0]._arrcplxf2[1].real, recv[1]._arrcplxf2[1].image), 
                               __fadd_rn(recv[2]._arrcplxf2[1].real, recv[3]._arrcplxf2[1].image));

        _FFT_domain_dex += (_kernel_info._store_pitch << 1);
        dst[_FFT_domain_dex * _pitchdst_v2 + tidx] = res._vf2;
    }
}