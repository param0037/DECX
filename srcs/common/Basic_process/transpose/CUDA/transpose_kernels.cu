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


#include "transpose_kernels.cuh"
#ifdef _DECX_DSP_CUDA_
#include "../../../../modules/DSP/FFT/FFT_commons.h"
#endif


__global__
void decx::blas::GPUK::cu_transpose2D_b8(const double2* src,
                                       double2 *dst, 
                                       const uint32_t pitchsrc_v2,        // in double2 (de::CPf x2)
                                       const uint32_t pitchdst_v2,        // in double2 (de::CPf x2)
                                       const uint2 dst_dims)   // in de::CPf
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    uint64_t dex = (tidy << 3) * pitchsrc_v2 + tidx;

    __shared__ double2 _buffer[64][32 + 1];

    double2 _regs[8], _transp[4];
    for (uint8_t i = 0; i < 8; ++i) {
        _regs[i] = decx::utils::vec2_set1_fp64(0.0);
    }
    // Load source data to registers
    if (tidx < pitchsrc_v2) {
        for (uint8_t i = 0; i < 8; ++i) {
            if ((tidy << 3) + i < dst_dims.x) { _regs[i] = src[dex + pitchsrc_v2 * i]; }
        }
    }
    // On-register transpose (lane 0)
    _transp[0].x = _regs[0].x;      _transp[0].y = _regs[1].x;
    _transp[1].x = _regs[2].x;      _transp[1].y = _regs[3].x;
    _transp[2].x = _regs[4].x;      _transp[2].y = _regs[5].x;
    _transp[3].x = _regs[6].x;      _transp[3].y = _regs[7].x;
    // Store to shared memory, also in transposed form (lane 0)
    for (uint8_t i = 0; i < 4; ++i) {
        _buffer[threadIdx.x * 2][threadIdx.y * 4 + i] = _transp[i];
    }
    // On-register transpose (lane 1)
    _transp[0].x = _regs[0].y;      _transp[0].y = _regs[1].y;
    _transp[1].x = _regs[2].y;      _transp[1].y = _regs[3].y;
    _transp[2].x = _regs[4].y;      _transp[2].y = _regs[5].y;
    _transp[3].x = _regs[6].y;      _transp[3].y = _regs[7].y;
    // Store to shared memory, also in transposed form (lane 1)
    for (uint8_t i = 0; i < 4; ++i) {
        _buffer[threadIdx.x * 2 + 1][threadIdx.y * 4 + i] = _transp[i];
    }

    __syncthreads();

    for (uint8_t i = 0; i < 8; ++i) {
        _regs[i] = _buffer[threadIdx.y * 8 + i][threadIdx.x];
    }

    tidx = threadIdx.x + blockIdx.y * blockDim.x;
    tidy = threadIdx.y + blockIdx.x * blockDim.y;

    dex = tidy * 8 * pitchdst_v2 + tidx;

    if (tidx < pitchdst_v2) {
        for (uint8_t i = 0; i < 8; ++i) {
            if (tidy * 8 + i < dst_dims.y) { dst[dex + pitchdst_v2 * i] = _regs[i]; }
        }
    }
}



__global__
void decx::blas::GPUK::cu_transpose2D_b16(const double2* src,
                                        double2 *dst, 
                                        const uint32_t pitchsrc_v1,        // in double2 (de::CPd x1)
                                        const uint32_t pitchdst_v1,        // in double2 (de::CPd x1)
                                        const uint2 dst_dims)   // in de::CPd
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    uint64_t dex = (tidy << 2) * pitchsrc_v1 + tidx;

    __shared__ double2 _buffer[32][32 + 1];
    decx::utils::_cuda_vec128 _regs[4];

    for (uint8_t i = 0; i < 4; ++i) {
        _regs[i]._vd = decx::utils::vec2_set1_fp64(0.0);
    }
    // Load source data to registers
    if (tidx < pitchsrc_v1) {
        for (uint8_t i = 0; i < 4; ++i) {
            if ((tidy << 2) + i < dst_dims.x) { 
                _regs[i]._vd = src[dex + pitchsrc_v1 * i];
            }
        }
    }

    _buffer[threadIdx.x][threadIdx.y * 4] = _regs[0]._vd;
    _buffer[threadIdx.x][threadIdx.y * 4 + 1] = _regs[1]._vd;
    _buffer[threadIdx.x][threadIdx.y * 4 + 2] = _regs[2]._vd;
    _buffer[threadIdx.x][threadIdx.y * 4 + 3] = _regs[3]._vd;

    __syncthreads();

    for (uint8_t i = 0; i < 4; ++i) {
        _regs[i]._vd = _buffer[threadIdx.y * 4 + i][threadIdx.x];
    }

    tidx = threadIdx.x + blockIdx.y * blockDim.x;
    tidy = threadIdx.y + blockIdx.x * blockDim.y;

    dex = tidy * 4 * pitchdst_v1 + tidx;

    if (tidx < pitchdst_v1) {
        for (uint8_t i = 0; i < 4; ++i) {
            if (tidy * 4 + i < dst_dims.y) { dst[dex + pitchdst_v1 * i] = _regs[i]._vd; }
        }
    }
}


#ifdef _DECX_DSP_CUDA_
__global__
void decx::blas::GPUK::cu_transpose2D_b8_for_FFT(const double2* src,
                                               double2 *dst, 
                                               const uint32_t pitchsrc_v2,        // in double2 (de::CPf x2)
                                               const uint32_t pitchdst_v2,        // in double2 (de::CPf x2)
                                               const uint2 dst_dims)   // in de::CPf
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    uint64_t dex = (tidy << 3) * pitchsrc_v2 + tidx;

    __shared__ double2 _buffer[64][32 + 1];
    decx::utils::_cuda_vec128 W2;

    decx::utils::_cuda_vec128 _regs[8];
    double2 _transp[4];
    for (uint8_t i = 0; i < 8; ++i) {
        _regs[i]._vd = decx::utils::vec2_set1_fp64(0.0);
    }
    // Load source data to registers
    if (tidx < pitchsrc_v2) {
        for (uint8_t i = 0; i < 8; ++i) {
            if ((tidy << 3) + i < dst_dims.x) { 
                _regs[i]._vd = src[dex + pitchsrc_v2 * i]; 

                W2._arrcplxf2[0].construct_with_phase(__fdividef(__fmul_rn(Two_Pi, (tidx << 1) * ((tidy << 3) + i)), (dst_dims.x * dst_dims.y)));
                W2._arrcplxf2[1].construct_with_phase(__fdividef(__fmul_rn(Two_Pi, ((tidx << 1) + 1) * ((tidy << 3) + i)), (dst_dims.x * dst_dims.y)));
                _regs[i]._vf = decx::dsp::fft::GPUK::_complex_mul2_fp32(_regs[i]._vf, W2._vf);
            }
        }
    }
    // On-register transpose (lane 0)
    _transp[0].x = _regs[0]._vd.x;      _transp[0].y = _regs[1]._vd.x;
    _transp[1].x = _regs[2]._vd.x;      _transp[1].y = _regs[3]._vd.x;
    _transp[2].x = _regs[4]._vd.x;      _transp[2].y = _regs[5]._vd.x;
    _transp[3].x = _regs[6]._vd.x;      _transp[3].y = _regs[7]._vd.x;
    // Store to shared memory, also in transposed form (lane 0)
    for (uint8_t i = 0; i < 4; ++i) {
        _buffer[threadIdx.x * 2][threadIdx.y * 4 + i] = _transp[i];
    }
    // On-register transpose (lane 1)
    _transp[0].x = _regs[0]._vd.y;      _transp[0].y = _regs[1]._vd.y;
    _transp[1].x = _regs[2]._vd.y;      _transp[1].y = _regs[3]._vd.y;
    _transp[2].x = _regs[4]._vd.y;      _transp[2].y = _regs[5]._vd.y;
    _transp[3].x = _regs[6]._vd.y;      _transp[3].y = _regs[7]._vd.y;
    // Store to shared memory, also in transposed form (lane 1)
    for (uint8_t i = 0; i < 4; ++i) {
        _buffer[threadIdx.x * 2 + 1][threadIdx.y * 4 + i] = _transp[i];
    }

    __syncthreads();

    for (uint8_t i = 0; i < 8; ++i) {
        _regs[i]._vd = _buffer[threadIdx.y * 8 + i][threadIdx.x];
    }

    tidx = threadIdx.x + blockIdx.y * blockDim.x;
    tidy = threadIdx.y + blockIdx.x * blockDim.y;

    dex = tidy * 8 * pitchdst_v2 + tidx;

    if (tidx < pitchdst_v2) {
        for (uint8_t i = 0; i < 8; ++i) {
            if (tidy * 8 + i < dst_dims.y) { dst[dex + pitchdst_v2 * i] = _regs[i]._vd; }
        }
    }
}


__global__
void decx::blas::GPUK::cu_transpose2D_b16_for_FFT(const double2* src,
                                               double2 *dst, 
                                               const uint32_t pitchsrc_v1,        // in double2 (de::CPd x1)
                                               const uint32_t pitchdst_v1,        // in double2 (de::CPd x1)
                                               const uint2 dst_dims)   // in de::CPd
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    uint64_t dex = (tidy << 2) * pitchsrc_v1 + tidx;

    __shared__ double2 _buffer[32][32 + 1];
    de::CPd W;
    decx::utils::_cuda_vec128 _regs[4];

    for (uint8_t i = 0; i < 4; ++i) {
        _regs[i]._vd = decx::utils::vec2_set1_fp64(0.0);
    }
    // Load source data to registers
    if (tidx < pitchsrc_v1) {
        for (uint8_t i = 0; i < 4; ++i) {
            if ((tidy << 2) + i < dst_dims.x) { 
                _regs[i]._vd = src[dex + pitchsrc_v1 * i]; 

                W.construct_with_phase(__ddiv_rn(__dmul_rn(Two_Pi, tidx * ((tidy << 2) + i)), (dst_dims.x * dst_dims.y)));
                _regs[i]._cplxd = decx::dsp::fft::GPUK::_complex_mul_fp64(_regs[i]._cplxd, W);
            }
        }
    }

    _buffer[threadIdx.x][threadIdx.y * 4] = _regs[0]._vd;
    _buffer[threadIdx.x][threadIdx.y * 4 + 1] = _regs[1]._vd;
    _buffer[threadIdx.x][threadIdx.y * 4 + 2] = _regs[2]._vd;
    _buffer[threadIdx.x][threadIdx.y * 4 + 3] = _regs[3]._vd;

    __syncthreads();

    for (uint8_t i = 0; i < 4; ++i) {
        _regs[i]._vd = _buffer[threadIdx.y * 4 + i][threadIdx.x];
    }

    tidx = threadIdx.x + blockIdx.y * blockDim.x;
    tidy = threadIdx.y + blockIdx.x * blockDim.y;

    dex = tidy * 4 * pitchdst_v1 + tidx;

    if (tidx < pitchdst_v1) {
        for (uint8_t i = 0; i < 4; ++i) {
            if (tidy * 4 + i < dst_dims.y) { dst[dex + pitchdst_v1 * i] = _regs[i]._vd; }
        }
    }
}

#endif


__global__
void decx::blas::GPUK::cu_transpose2D_b4(const float2* src,
                                       float2 *dst, 
                                       const uint32_t pitchsrc_v2,        // in double2 (de::CPf x2)
                                       const uint32_t pitchdst_v2,        // in double2 (de::CPf x2)
                                       const uint2 dst_dims)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    uint64_t dex = (tidy << 3) * pitchsrc_v2 + tidx;

    __shared__ float2 _buffer[64][32 + 1];

    float2 _regs[8], _transp[4];
    for (uint8_t i = 0; i < 8; ++i) {
        _regs[i] = decx::utils::vec2_set1_fp32(0.0);
    }
    // Load source data to registers
    if (tidx < pitchsrc_v2) {
        for (uint8_t i = 0; i < 8; ++i) {
            if ((tidy << 3) + i < dst_dims.x) { _regs[i] = src[dex + pitchsrc_v2 * i]; }
        }
    }
    // On-register transpose (lane 0)
    _transp[0].x = _regs[0].x;      _transp[0].y = _regs[1].x;
    _transp[1].x = _regs[2].x;      _transp[1].y = _regs[3].x;
    _transp[2].x = _regs[4].x;      _transp[2].y = _regs[5].x;
    _transp[3].x = _regs[6].x;      _transp[3].y = _regs[7].x;
    // Store to shared memory, also in transposed form (lane 0)
    for (uint8_t i = 0; i < 4; ++i) {
        _buffer[threadIdx.x * 2][threadIdx.y * 4 + i] = _transp[i];
    }
    // On-register transpose (lane 1)
    _transp[0].x = _regs[0].y;      _transp[0].y = _regs[1].y;
    _transp[1].x = _regs[2].y;      _transp[1].y = _regs[3].y;
    _transp[2].x = _regs[4].y;      _transp[2].y = _regs[5].y;
    _transp[3].x = _regs[6].y;      _transp[3].y = _regs[7].y;
    // Store to shared memory, also in transposed form (lane 1)
    for (uint8_t i = 0; i < 4; ++i) {
        _buffer[threadIdx.x * 2 + 1][threadIdx.y * 4 + i] = _transp[i];
    }

    __syncthreads();

    for (uint8_t i = 0; i < 8; ++i) {
        _regs[i] = _buffer[threadIdx.y * 8 + i][threadIdx.x];
    }

    tidx = threadIdx.x + blockIdx.y * blockDim.x;
    tidy = threadIdx.y + blockIdx.x * blockDim.y;

    dex = tidy * 8 * pitchdst_v2 + tidx;

    if (tidx < pitchdst_v2) {
        for (uint8_t i = 0; i < 8; ++i) {
            if (tidy * 8 + i < dst_dims.y) { dst[dex + pitchdst_v2 * i] = _regs[i]; }
        }
    }
}


__global__ void decx::blas::GPUK::
cu_transpose2D_b2(const float4* __restrict src,
                  float4* __restrict dst, 
                  const uint32_t pitchsrc_v8,        // in double2 (de::Half x8)
                  const uint32_t pitchdst_v8,        // in double2 (de::Half x8)
                  const uint2 dst_dims)
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    uint64_t dex = (tidy << 3) * pitchsrc_v8 + tidx;

    __shared__ float4 _buffer[128][128 / 8 + 1];

    decx::utils::_cuda_vec128 _regs[8], _transp[4];
    for (uint8_t i = 0; i < 8; ++i) {
        _regs[i]._vf = decx::utils::vec4_set1_fp32(0);
    }
    // Load source data to registers
    if (tidx < pitchsrc_v8) {
        for (uint8_t i = 0; i < 8; ++i) {
            if ((tidy << 3) + i < dst_dims.x) { _regs[i]._vf = src[dex + pitchsrc_v8 * i]; }
        }
    }
#pragma unroll
    for (uint32_t i = 0; i < 4; ++i){
        uint32_t _backup = _regs[i * 2]._vui.x;
        _regs[i * 2]._vui.x = __byte_perm(_regs[i * 2]._vui.x, _regs[i * 2 + 1]._vui.x, 0x5410);
        _regs[i * 2 + 1]._vui.x = __byte_perm(_backup, _regs[i * 2 + 1]._vui.x, 0x7632);

        _backup = _regs[i * 2]._vui.y;
        _regs[i * 2]._vui.y = __byte_perm(_regs[i * 2]._vui.y, _regs[i * 2 + 1]._vui.y, 0x5410);
        _regs[i * 2 + 1]._vui.y = __byte_perm(_backup, _regs[i * 2 + 1]._vui.y, 0x7632);

        _backup = _regs[i * 2]._vui.z;
        _regs[i * 2]._vui.z = __byte_perm(_regs[i * 2]._vui.z, _regs[i * 2 + 1]._vui.z, 0x5410);
        _regs[i * 2 + 1]._vui.z = __byte_perm(_backup, _regs[i * 2 + 1]._vui.z, 0x7632);

        _backup = _regs[i * 2]._vui.w;
        _regs[i * 2]._vui.w = __byte_perm(_regs[i * 2]._vui.w, _regs[i * 2 + 1]._vui.w, 0x5410);
        _regs[i * 2 + 1]._vui.w = __byte_perm(_backup, _regs[i * 2 + 1]._vui.w, 0x7632);
    }
    // On-register transpose (lane 0)
    _transp[0]._arrui[0] = _regs[0]._arrui[0];      _transp[1]._arrui[0] = _regs[1]._arrui[0];
    _transp[0]._arrui[1] = _regs[2]._arrui[0];      _transp[1]._arrui[1] = _regs[3]._arrui[0];
    _transp[0]._arrui[2] = _regs[4]._arrui[0];      _transp[1]._arrui[2] = _regs[5]._arrui[0];
    _transp[0]._arrui[3] = _regs[6]._arrui[0];      _transp[1]._arrui[3] = _regs[7]._arrui[0];

    _transp[2]._arrui[0] = _regs[0]._arrui[1];      _transp[3]._arrui[0] = _regs[1]._arrui[1];
    _transp[2]._arrui[1] = _regs[2]._arrui[1];      _transp[3]._arrui[1] = _regs[3]._arrui[1];
    _transp[2]._arrui[2] = _regs[4]._arrui[1];      _transp[3]._arrui[2] = _regs[5]._arrui[1];
    _transp[2]._arrui[3] = _regs[6]._arrui[1];      _transp[3]._arrui[3] = _regs[7]._arrui[1];

    // Store to shared memory, also in transposed form (lane 0)
    for (uint8_t i = 0; i < 4; ++i) {
        _buffer[threadIdx.x + i * blockDim.x][threadIdx.y] = _transp[i]._vf;
    }

    // On-register transpose (lane 1)
    _transp[0]._arrui[0] = _regs[0]._arrui[2];      _transp[1]._arrui[0] = _regs[1]._arrui[2];
    _transp[0]._arrui[1] = _regs[2]._arrui[2];      _transp[1]._arrui[1] = _regs[3]._arrui[2];
    _transp[0]._arrui[2] = _regs[4]._arrui[2];      _transp[1]._arrui[2] = _regs[5]._arrui[2];
    _transp[0]._arrui[3] = _regs[6]._arrui[2];      _transp[1]._arrui[3] = _regs[7]._arrui[2];

    _transp[2]._arrui[0] = _regs[0]._arrui[3];      _transp[3]._arrui[0] = _regs[1]._arrui[3];
    _transp[2]._arrui[1] = _regs[2]._arrui[3];      _transp[3]._arrui[1] = _regs[3]._arrui[3];
    _transp[2]._arrui[2] = _regs[4]._arrui[3];      _transp[3]._arrui[2] = _regs[5]._arrui[3];
    _transp[2]._arrui[3] = _regs[6]._arrui[3];      _transp[3]._arrui[3] = _regs[7]._arrui[3];

    // Store to shared memory, also in transposed form (lane 0)
    for (uint8_t i = 4; i < 8; ++i) {
         _buffer[threadIdx.x + i * blockDim.x][threadIdx.y] = _transp[i - 4]._vf;
    }

    __syncthreads();

#pragma unroll
    for (uint8_t i = 0; i < 8; ++i) {
        _regs[i]._vf = _buffer[threadIdx.y + i * 16][threadIdx.x];
    }

    tidx = threadIdx.x + blockIdx.y * blockDim.x;
    tidy = threadIdx.y + blockIdx.x * blockDim.y;

    dex = tidy * 8 * pitchdst_v8 + tidx;

    if (tidx < pitchdst_v8) {
        for (uint8_t i = 0; i < 8; ++i) {
            if (tidy * 8 + i < dst_dims.y) { dst[dex + pitchdst_v8 * i] = _regs[i]._vf; }
        }
    }
}



__global__
void decx::blas::GPUK::cu_transpose2D_b1(const uint32_t* src,
                                       uint32_t *dst, 
                                       const uint32_t pitchsrc_v4,        // in uchar4 (int32_t)
                                       const uint32_t pitchdst_v4,        // in uchar4 (int32_t)
                                       const uint2 dst_dims)              // in uchar
{
    uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    uint64_t dex = (tidy << 4) * pitchsrc_v4 + tidx;

    __shared__ uint32_t _buffer[128][32 + 1];

    uint32_t _regs[16];
    uint32_t tmp[2];

    for (uint8_t i = 0; i < 16; ++i) { _regs[i] = 0x00; }
    // Load source data to registers
    if (tidx < pitchsrc_v4) {
        for (uint8_t i = 0; i < 16; ++i) {
            if ((tidy << 4) + i < dst_dims.x) { _regs[i] = src[dex + pitchsrc_v4 * i]; }
        }
    }
    // On-register transpose
    // Local 2x2 transpose
    for (uint8_t i = 0; i < 8; ++i) {
        tmp[0] = __byte_perm(_regs[i * 2], _regs[i * 2 + 1], 0x6240);
        tmp[1] = __byte_perm(_regs[i * 2], _regs[i * 2 + 1], 0x7351);
        _regs[i * 2] = tmp[0];      _regs[i * 2 + 1] = tmp[1];
    }

    // 2x2 block transpose on registers
    for (uint8_t i = 0; i < 4; ++i) {
        tmp[0] = __byte_perm(_regs[i * 4], _regs[i * 4 + 2], 0x5410);
        tmp[1] = __byte_perm(_regs[i * 4], _regs[i * 4 + 2], 0x7632);
        _regs[i * 4] = tmp[0];          _regs[i * 4 + 2] = tmp[1];
        tmp[0] = __byte_perm(_regs[i * 4 + 1], _regs[i * 4 + 3], 0x5410);
        tmp[1] = __byte_perm(_regs[i * 4 + 1], _regs[i * 4 + 3], 0x7632);
        _regs[i * 4 + 1] = tmp[0];           _regs[i * 4 + 3] = tmp[1];
    }

    for (uint8_t i = 0; i < 4; ++i) {
        _buffer[threadIdx.x * 4 + i][threadIdx.y * 4] = _regs[i + 0];
        _buffer[threadIdx.x * 4 + i][threadIdx.y * 4 + 1] = _regs[i + 4];
        _buffer[threadIdx.x * 4 + i][threadIdx.y * 4 + 2] = _regs[i + 8];
        _buffer[threadIdx.x * 4 + i][threadIdx.y * 4 + 3] = _regs[i + 12];
    }

    __syncthreads();

    for (uint8_t i = 0; i < 16; ++i) {
        _regs[i] = _buffer[threadIdx.y * 16 + i][threadIdx.x];
    }

    tidx = threadIdx.x + blockIdx.y * blockDim.x;
    tidy = threadIdx.y + blockIdx.x * blockDim.y;

    dex = tidy * 16 * pitchdst_v4 + tidx;

    if (tidx < pitchdst_v4) {
        for (uint8_t i = 0; i < 16; ++i) {
            if (tidy * 16 + i < dst_dims.y) { dst[dex + pitchdst_v4 * i] = _regs[i]; }
        }
    }
}