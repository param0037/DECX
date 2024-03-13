/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "Conv2_uint8_kernels.cuh"
#include "../Conv2_kernel_defs.cuh"



__global__ void
decx::conv::GPUK::cu_Conv2_r64x8_uc8_kfp32(const float4* __restrict      src,
                                           const float* __restrict       kernel,
                                           float2* __restrict            dst, 
                                           const uint32_t                pitch_src,         // in float4
                                           const uint32_t                pitch_dst,         // in float2
                                           const uint2                   kernel_dims,
                                           const uint2                   kernel_shift,
                                           const uint2                   dst_dims)
{
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t idy = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ decx::utils::_cuda_vec64 src_frag[32][32 + sharedmem_offset * 2];

    float4 _accu0 = decx::utils::vec4_set1_fp32(0), 
           _accu1 = decx::utils::vec4_set1_fp32(0);

    decx::utils::_cuda_vec64 plane, plane_space[2];

    uint64_t glo_dex = idy * pitch_src + (threadIdx.x + blockDim.x / 2 * blockIdx.x);
    _accu0 = src[glo_dex];                 
    *((float4*)&src_frag[threadIdx.y][threadIdx.x * 2]) = _accu0;
    _accu0 = src[glo_dex + pitch_src * blockDim.x];
    *((float4*)&src_frag[threadIdx.y + 16][threadIdx.x * 2]) = _accu0;

    __syncthreads();

    _accu0 = decx::utils::vec4_set1_fp32(0);
    int dx, dy;
    for (int i = 0; i < kernel_dims.x * kernel_dims.y; ++i)
    {
        dx = (i / kernel_dims.x) + kernel_shift.x;        dy = ((i % kernel_dims.x) / 8) + (kernel_shift.y / 8);

        if ((i % kernel_dims.x) % 8 == 0) {
            plane_space[0] = src_frag[threadIdx.y + dx][threadIdx.x + dy];
            plane_space[1] = src_frag[threadIdx.y + dx][threadIdx.x + dy + 1];
            plane = plane_space[0];
        }
        else {
            plane = decx::conv::GPUK::reg64_shiftL_uint8(plane_space[0], plane_space[1], ((i % kernel_dims.x) % 8) * 8);
        }
        // calculate_accumulate
        const float tmp_ker = kernel[i];

        _accu0.x = __fmaf_rn(__uint2float_rn(plane._v_uint8[0]), tmp_ker, _accu0.x);
        _accu0.y = __fmaf_rn(__uint2float_rn(plane._v_uint8[1]), tmp_ker, _accu0.y);
        _accu0.z = __fmaf_rn(__uint2float_rn(plane._v_uint8[2]), tmp_ker, _accu0.z);
        _accu0.w = __fmaf_rn(__uint2float_rn(plane._v_uint8[3]), tmp_ker, _accu0.w);

        _accu1.x = __fmaf_rn(__uint2float_rn(plane._v_uint8[4]), tmp_ker, _accu1.x);
        _accu1.y = __fmaf_rn(__uint2float_rn(plane._v_uint8[5]), tmp_ker, _accu1.y);
        _accu1.z = __fmaf_rn(__uint2float_rn(plane._v_uint8[6]), tmp_ker, _accu1.z);
        _accu1.w = __fmaf_rn(__uint2float_rn(plane._v_uint8[7]), tmp_ker, _accu1.w);
    }

    plane = decx::conv::GPUK::cvt8_fp32_uint8(_accu0, _accu1);

    glo_dex = idy * pitch_dst + idx;
    if (idy < dst_dims.y && idx < dst_dims.x) {
        dst[glo_dex] = plane._vf2;
    }
}




__global__ void
decx::conv::GPUK::cu_Conv2_r64x8_uc8_fp32_kfp32(const float4* __restrict      src,
                                                const float* __restrict       kernel,
                                                float4* __restrict            dst, 
                                                const uint32_t                pitch_src,         // in float4
                                                const uint32_t                pitch_dst,         // in float4
                                                const uint2                   kernel_dims, 
                                                const uint2                   kernel_shift,
                                                const uint2                   dst_dims)
{
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t idy = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ decx::utils::_cuda_vec64 src_frag[32][32 + sharedmem_offset * 2];

    float4 _accu0 = decx::utils::vec4_set1_fp32(0), 
           _accu1 = decx::utils::vec4_set1_fp32(0);

    decx::utils::_cuda_vec64 plane, plane_space[2];

    size_t glo_dex = idy * pitch_src + (threadIdx.x + blockDim.x / 2 * blockIdx.x);
    _accu0 = src[glo_dex];                 
    *((float4*)&src_frag[threadIdx.y][threadIdx.x * 2]) = _accu0;
    _accu0 = src[glo_dex + pitch_src * blockDim.x];
    *((float4*)&src_frag[threadIdx.y + 16][threadIdx.x * 2]) = _accu0;

    __syncthreads();

    _accu0 = decx::utils::vec4_set1_fp32(0);
    int dx, dy;
    for (int i = 0; i < kernel_dims.x * kernel_dims.y; ++i)
    {
        dx = (i / kernel_dims.x) + kernel_shift.x;        dy = ((i % kernel_dims.x) / 8) + (kernel_shift.y / 8);

        if ((i % kernel_dims.x) % 8 == 0) {
            plane_space[0] = src_frag[threadIdx.y + dx][threadIdx.x + dy];
            plane_space[1] = src_frag[threadIdx.y + dx][threadIdx.x + dy + 1];
            plane = plane_space[0];
        }
        else {
            plane = decx::conv::GPUK::reg64_shiftL_uint8(plane_space[0], plane_space[1], ((i % kernel_dims.x) % 8) * 8);
        }
        // calculate_accumulate
        const float tmp_ker = kernel[i];

        _accu0.x = __fmaf_rn(__uint2float_rn(plane._v_uint8[0]), tmp_ker, _accu0.x);
        _accu0.y = __fmaf_rn(__uint2float_rn(plane._v_uint8[1]), tmp_ker, _accu0.y);
        _accu0.z = __fmaf_rn(__uint2float_rn(plane._v_uint8[2]), tmp_ker, _accu0.z);
        _accu0.w = __fmaf_rn(__uint2float_rn(plane._v_uint8[3]), tmp_ker, _accu0.w);

        _accu1.x = __fmaf_rn(__uint2float_rn(plane._v_uint8[4]), tmp_ker, _accu1.x);
        _accu1.y = __fmaf_rn(__uint2float_rn(plane._v_uint8[5]), tmp_ker, _accu1.y);
        _accu1.z = __fmaf_rn(__uint2float_rn(plane._v_uint8[6]), tmp_ker, _accu1.z);
        _accu1.w = __fmaf_rn(__uint2float_rn(plane._v_uint8[7]), tmp_ker, _accu1.w);
    }

    glo_dex = idy * pitch_dst + idx * 2;
    if (idy < dst_dims.y && idx < dst_dims.x) {
        dst[glo_dex] = _accu0;
        dst[glo_dex + 1] = _accu1;
    }
}



__global__ void
decx::conv::GPUK::cu_Conv2_r64x8_uc8_kfp32_LKH(const float4* __restrict         src,
                                               const float* __restrict          kernel,
                                               float2* __restrict               dst, 
                                               const uint                       pitch_src,         // in float4
                                               const uint                       pitch_dst,         // in float2
                                               const uint                       Wker, 
                                               const uint                       kernel_shift_W,
                                               const decx::conv::_conv2_LKH     LKH,
                                               const uint2                      dst_dims)
{
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t idy = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ decx::utils::_cuda_vec64 src_frag[32][32 + sharedmem_offset * 2];

    float4 _accu0 = decx::utils::vec4_set1_fp32(0), 
           _accu1 = decx::utils::vec4_set1_fp32(0),
           _IO_buffer;

    decx::utils::_cuda_vec64 plane, plane_space[2];

    uint total_frac_ker_len;

    size_t glo_dex = idy * pitch_src + (threadIdx.x + blockDim.x / 2 * blockIdx.x);
    uint ker_dex = 0;

    for (uint _L = 0; _L < LKH._loop + 1; ++_L)
    {
        _IO_buffer = src[glo_dex];
        *((float4*)&src_frag[threadIdx.y][threadIdx.x * 2]) = _IO_buffer;

        _IO_buffer = decx::utils::vec4_set1_fp32(0);
        if (_L == LKH._loop) {
            if (threadIdx.y < LKH._left - 1) {
                _IO_buffer = src[glo_dex + pitch_src * blockDim.y];
            }
        }
        else {
            _IO_buffer = src[glo_dex + pitch_src * blockDim.y];
        }
        *((float4*)&src_frag[threadIdx.y + 16][threadIdx.x * 2]) = _IO_buffer;

        __syncthreads();

        int dx, dy;
        total_frac_ker_len = (_L != LKH._loop) ? (17 * Wker) : (LKH._left * Wker);

        for (int i = 0; i < total_frac_ker_len; ++i)
        {
            dx = i / Wker;        dy = ((i % Wker) / 8) + (kernel_shift_W / 8);

            if ((i % Wker) % 8 == 0) {
                plane_space[0] = src_frag[threadIdx.y + dx][threadIdx.x + dy];
                plane_space[1] = src_frag[threadIdx.y + dx][threadIdx.x + dy + 1];
                plane = plane_space[0];
            }
            else {
                plane = decx::conv::GPUK::reg64_shiftL_uint8(plane_space[0], plane_space[1], ((i % Wker) % 8) * 8);
            }
            // calculate_accumulate
            const float tmp_ker = kernel[ker_dex];

            _accu0.x = __fmaf_rn(__uint2float_rn(plane._v_uint8[0]), tmp_ker, _accu0.x);
            _accu0.y = __fmaf_rn(__uint2float_rn(plane._v_uint8[1]), tmp_ker, _accu0.y);
            _accu0.z = __fmaf_rn(__uint2float_rn(plane._v_uint8[2]), tmp_ker, _accu0.z);
            _accu0.w = __fmaf_rn(__uint2float_rn(plane._v_uint8[3]), tmp_ker, _accu0.w);

            _accu1.x = __fmaf_rn(__uint2float_rn(plane._v_uint8[4]), tmp_ker, _accu1.x);
            _accu1.y = __fmaf_rn(__uint2float_rn(plane._v_uint8[5]), tmp_ker, _accu1.y);
            _accu1.z = __fmaf_rn(__uint2float_rn(plane._v_uint8[6]), tmp_ker, _accu1.z);
            _accu1.w = __fmaf_rn(__uint2float_rn(plane._v_uint8[7]), tmp_ker, _accu1.w);

            ++ker_dex;
        }

        __syncthreads();
        glo_dex += ((pitch_src << 4) + pitch_src);
    }

    plane = decx::conv::GPUK::cvt8_fp32_uint8(_accu0, _accu1);

    glo_dex = idy * pitch_dst + idx;
    if (idy < dst_dims.y && idx < dst_dims.x) {
        dst[glo_dex] = plane._vf2;
    }
}



__global__ void
decx::conv::GPUK::cu_Conv2_r64x8_uc8_fp32_kfp32_LKH(const float4* __restrict         src,
                                                    const float* __restrict          kernel,
                                                    float4* __restrict               dst, 
                                                    const uint                       pitch_src,         // in float4
                                                    const uint                       pitch_dst,         // in float2
                                                    const uint                       Wker, 
                                                    const uint                       kernel_shift_W,
                                                    const decx::conv::_conv2_LKH     LKH,
                                                    const uint2                      dst_dims)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ decx::utils::_cuda_vec64 src_frag[32][32 + sharedmem_offset * 2];

    float4 _accu0 = decx::utils::vec4_set1_fp32(0), 
           _accu1 = decx::utils::vec4_set1_fp32(0),
           _IO_buffer;

    decx::utils::_cuda_vec64 plane, plane_space[2];

    uint total_frac_ker_len;

    size_t glo_dex = idy * pitch_src + (threadIdx.x + blockDim.x / 2 * blockIdx.x);
    uint ker_dex = 0;

    for (uint _L = 0; _L < LKH._loop + 1; ++_L)
    {
        _IO_buffer = src[glo_dex];
        *((float4*)&src_frag[threadIdx.y][threadIdx.x * 2]) = _IO_buffer;

        _IO_buffer = decx::utils::vec4_set1_fp32(0);
        if (_L == LKH._loop) {
            if (threadIdx.y < LKH._left - 1) {
                _IO_buffer = src[glo_dex + pitch_src * blockDim.y];
            }
        }
        else {
            _IO_buffer = src[glo_dex + pitch_src * blockDim.y];
        }
        *((float4*)&src_frag[threadIdx.y + 16][threadIdx.x * 2]) = _IO_buffer;

        __syncthreads();

        int dx, dy;
        total_frac_ker_len = (_L != LKH._loop) ? (17 * Wker) : (LKH._left * Wker);

        for (int i = 0; i < total_frac_ker_len; ++i)
        {
            dx = i / Wker;        dy = ((i % Wker) / 8) + (kernel_shift_W / 8);

            if ((i % Wker) % 8 == 0) {
                plane_space[0] = src_frag[threadIdx.y + dx][threadIdx.x + dy];
                plane_space[1] = src_frag[threadIdx.y + dx][threadIdx.x + dy + 1];
                plane = plane_space[0];
            }
            else {
                plane = decx::conv::GPUK::reg64_shiftL_uint8(plane_space[0], plane_space[1], ((i % Wker) % 8) * 8);
            }
            // calculate_accumulate
            const float tmp_ker = kernel[ker_dex];

            _accu0.x = fmaf(__uint2float_rn(plane._v_uint8[0]), tmp_ker, _accu0.x);
            _accu0.y = fmaf(__uint2float_rn(plane._v_uint8[1]), tmp_ker, _accu0.y);
            _accu0.z = fmaf(__uint2float_rn(plane._v_uint8[2]), tmp_ker, _accu0.z);
            _accu0.w = fmaf(__uint2float_rn(plane._v_uint8[3]), tmp_ker, _accu0.w);

            _accu1.x = fmaf(__uint2float_rn(plane._v_uint8[4]), tmp_ker, _accu1.x);
            _accu1.y = fmaf(__uint2float_rn(plane._v_uint8[5]), tmp_ker, _accu1.y);
            _accu1.z = fmaf(__uint2float_rn(plane._v_uint8[6]), tmp_ker, _accu1.z);
            _accu1.w = fmaf(__uint2float_rn(plane._v_uint8[7]), tmp_ker, _accu1.w);

            ++ker_dex;
        }

        __syncthreads();
        glo_dex += ((pitch_src << 4) + pitch_src);
    }

    glo_dex = idy * pitch_dst + idx * 2;
    if (idy < dst_dims.y && idx < dst_dims.x) {
        dst[glo_dex] = _accu0;
        dst[glo_dex + 1] = _accu1;
    }
}