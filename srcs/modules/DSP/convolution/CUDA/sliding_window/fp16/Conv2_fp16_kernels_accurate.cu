/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "Conv2_fp16_kernels.cuh"


__global__ void 
decx::conv::GPUK::cu_hConv2_r8_within_accu(const float4* __restrict src,
                                           const __half*            kernel,
                                           float4* __restrict       dst,
                                           const uint               pitch_src,
                                           const uint               pitch_dst,
                                           const uint2              kernel_dims,
                                           const uint2              kernel_shift,
                                           const uint2              dst_dims)
{
#if __ABOVE_SM_53
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t idy = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ float4 src_frag[32][144 / 8 + 1];

    decx::utils::_cuda_vec128 reg_0, reg_1, _accu[2];
    const uint32_t _k_loop_w_v4 = _get_k_loop_w_v8(_MAX_SLIDE_CONV_R8_V8_FP16_, kernel_shift.y);
    const uint8_t _L = kernel_shift.y % 8;

    uint64_t glo_dex = idy * pitch_src + idx;

    for (uint8_t i = 0; i < 2; ++i) {
        reg_0._vf = src[glo_dex];
        src_frag[(i << 4) + threadIdx.y][threadIdx.x] = reg_0._vf;
        glo_dex += (pitch_src << 4);
    }

    if (threadIdx.x < 2) {
        glo_dex = idy * pitch_src + idx + 16;

        for (uint8_t i = 0; i < 2; ++i) {
            reg_0._vf = src[glo_dex];
            src_frag[(i << 4) + threadIdx.y][16 + threadIdx.x] = reg_0._vf;
            glo_dex += (pitch_src << 4);
        }
    }

    __syncthreads();

    _accu[0]._vf = decx::utils::vec4_set1_fp32(0.f);
    _accu[1]._vf = decx::utils::vec4_set1_fp32(0.f);

    for (int i = 0; i < kernel_dims.y; ++i)
    {
        const __half* _row_kernel_ptr = kernel + i * kernel_dims.x;
        uint8_t _row_offset = (kernel_shift.y >> 3);

        reg_0._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset];
        reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];

        const uint8_t _inc = decx::conv::GPUK::_pre_residual_v8_conv_fp16_accu(reg_0, reg_1, _accu, _row_kernel_ptr, _L);
        _row_offset += (bool)(kernel_shift.y % 8);
        _row_kernel_ptr += _inc;

        for (int j = 0; j < _k_loop_w_v4; ++j)
        {
            if (j > 0 || _L) {
                reg_0._vf = reg_1._vf;
                reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];
            }

            decx::conv::GPUK::_full_v8_conv_fp16_accu(reg_0, reg_1, _accu, _row_kernel_ptr);

            _row_kernel_ptr += 8;
            ++_row_offset;
        }

        if (_L) {
            reg_0._vf = reg_1._vf;
            reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];
            decx::conv::GPUK::_post_residual_v8_conv_fp16_accu(reg_0, reg_1, _accu, _row_kernel_ptr, 8 - _L);
        }
    }

    reg_0._arrh2[0] = __floats2half2_rn(_accu[0]._vf.x, _accu[0]._vf.y);
    reg_0._arrh2[1] = __floats2half2_rn(_accu[0]._vf.z, _accu[0]._vf.w);
    reg_0._arrh2[2] = __floats2half2_rn(_accu[1]._vf.x, _accu[1]._vf.y);
    reg_0._arrh2[3] = __floats2half2_rn(_accu[1]._vf.z, _accu[1]._vf.w);

    glo_dex = idy * pitch_dst + idx;
    if (idy < dst_dims.y && idx < dst_dims.x) {
        dst[glo_dex] = reg_0._vf;
    }
#endif
}





__global__ void 
decx::conv::GPUK::cu_hConv2_r16_within_accu(const float4* __restrict                src, 
                                            const __half* kernel,
                                            float4* __restrict                dst,
                                            const uint             pitch_src, 
                                            const uint             pitch_dst,
                                            const uint2             kernel_dims,
                                            const uint2             kernel_shift,
                                            const uint2            dst_dims)
{
#if __ABOVE_SM_53
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t idy = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ float4 src_frag[48][160 / 8 + 1];

    decx::utils::_cuda_vec128 reg_0, reg_1, _accu[2];
    const uint32_t _k_loop_w_v4 = _get_k_loop_w_v8(_MAX_SLIDE_CONV_R16_V8_FP16_, kernel_shift.y);
    const uint8_t _L = kernel_shift.y % 8;

    uint64_t glo_dex = idy * pitch_src + idx;

    for (uint8_t i = 0; i < 3; ++i) {
        reg_0._vf = src[glo_dex];
        src_frag[(i << 4) + threadIdx.y][threadIdx.x] = reg_0._vf;
        glo_dex += (pitch_src << 4);
    }

    if (threadIdx.x < 4) {
        glo_dex = idy * pitch_src + idx + 16;

        for (uint8_t i = 0; i < 3; ++i) {
            reg_0._vf = src[glo_dex];
            src_frag[(i << 4) + threadIdx.y][16 + threadIdx.x] = reg_0._vf;
            glo_dex += (pitch_src << 4);
        }
    }

    __syncthreads();

    _accu[0]._vf = decx::utils::vec4_set1_fp32(0.f);
    _accu[1]._vf = decx::utils::vec4_set1_fp32(0.f);

    for (int i = 0; i < kernel_dims.y; ++i)
    {
        const __half* _row_kernel_ptr = kernel + i * kernel_dims.x;
        uint8_t _row_offset = (kernel_shift.y >> 8);

        reg_0._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset];
        reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];

        const uint8_t _inc = decx::conv::GPUK::_pre_residual_v8_conv_fp16_accu(reg_0, reg_1, _accu, _row_kernel_ptr, _L);
        _row_offset += (bool)(kernel_shift.y % 8);
        _row_kernel_ptr += _inc;

        for (int j = 0; j < _k_loop_w_v4; ++j)
        {
            if (j > 0 || _L) {
                reg_0._vf = reg_1._vf;
                reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];
            }

            decx::conv::GPUK::_full_v8_conv_fp16_accu(reg_0, reg_1, _accu, _row_kernel_ptr);

            _row_kernel_ptr += 8;
            ++_row_offset;
        }

        if (_L) {
            reg_0._vf = reg_1._vf;
            reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];
            decx::conv::GPUK::_post_residual_v8_conv_fp16_accu(reg_0, reg_1, _accu, _row_kernel_ptr, 8 - _L);
        }
    }

    reg_0._arrh2[0] = __floats2half2_rn(_accu[0]._vf.x, _accu[0]._vf.y);
    reg_0._arrh2[1] = __floats2half2_rn(_accu[0]._vf.z, _accu[0]._vf.w);
    reg_0._arrh2[2] = __floats2half2_rn(_accu[1]._vf.x, _accu[1]._vf.y);
    reg_0._arrh2[3] = __floats2half2_rn(_accu[1]._vf.z, _accu[1]._vf.w);

    glo_dex = idy * pitch_dst + idx;
    if (idy < dst_dims.y && idx < dst_dims.x) {
        dst[glo_dex] = reg_0._vf;
    }
#endif
}




__global__ void 
decx::conv::GPUK::cu_hConv2_r816_within_accu(const float4* __restrict          src,
                                             const __half* kernel,
                                             float4* __restrict                dst,
                                             const uint             pitch_src, 
                                             const uint             pitch_dst,
                                             const uint2            kernel_dims,
                                             const uint2             kernel_shift,
                                             const uint2            dst_dims)
{
#if __ABOVE_SM_53
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t idy = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ float4 src_frag[32][160 / 8 + 1];

    decx::utils::_cuda_vec128 reg_0, reg_1, _accu[2];
    const uint32_t _k_loop_w_v4 = _get_k_loop_w_v8(_MAX_SLIDE_CONV_R16_V8_FP16_, kernel_shift.y);
    const uint8_t _L = kernel_shift.y % 8;

    uint64_t glo_dex = idy * pitch_src + idx;

    for (uint8_t i = 0; i < 2; ++i) {
        reg_0._vf = src[glo_dex];
        src_frag[(i << 4) + threadIdx.y][threadIdx.x] = reg_0._vf;
        glo_dex += (pitch_src << 4);
    }

    if (threadIdx.x < 4) {
        glo_dex = idy * pitch_src + idx + 16;

        for (uint8_t i = 0; i < 2; ++i) {
            reg_0._vf = src[glo_dex];
            src_frag[(i << 4) + threadIdx.y][16 + threadIdx.x] = reg_0._vf;
            glo_dex += (pitch_src << 4);
        }
    }

    __syncthreads();

    _accu[0]._vf = decx::utils::vec4_set1_fp32(0.f);
    _accu[1]._vf = decx::utils::vec4_set1_fp32(0.f);

    for (int i = 0; i < kernel_dims.y; ++i)
    {
        const __half* _row_kernel_ptr = kernel + i * kernel_dims.x;
        uint8_t _row_offset = (kernel_shift.y >> 8);

        reg_0._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset];
        reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];

        const uint8_t _inc = decx::conv::GPUK::_pre_residual_v8_conv_fp16_accu(reg_0, reg_1, _accu, _row_kernel_ptr, _L);
        _row_offset += (bool)(kernel_shift.y % 8);
        _row_kernel_ptr += _inc;

        for (int j = 0; j < _k_loop_w_v4; ++j)
        {
            if (j > 0 || _L) {
                reg_0._vf = reg_1._vf;
                reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];
            }

            decx::conv::GPUK::_full_v8_conv_fp16_accu(reg_0, reg_1, _accu, _row_kernel_ptr);

            _row_kernel_ptr += 8;
            ++_row_offset;
        }

        if (_L) {
            reg_0._vf = reg_1._vf;
            reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];
            decx::conv::GPUK::_post_residual_v8_conv_fp16_accu(reg_0, reg_1, _accu, _row_kernel_ptr, 8 - _L);
        }
    }

    reg_0._arrh2[0] = __floats2half2_rn(_accu[0]._vf.x, _accu[0]._vf.y);
    reg_0._arrh2[1] = __floats2half2_rn(_accu[0]._vf.z, _accu[0]._vf.w);
    reg_0._arrh2[2] = __floats2half2_rn(_accu[1]._vf.x, _accu[1]._vf.y);
    reg_0._arrh2[3] = __floats2half2_rn(_accu[1]._vf.z, _accu[1]._vf.w);

    glo_dex = idy * pitch_dst + idx;
    if (idy < dst_dims.y && idx < dst_dims.x) {
        dst[glo_dex] = reg_0._vf;
    }
#endif
}




__global__ void 
decx::conv::GPUK::cu_hConv2_r168_within_accu(const float4* __restrict          src,
                                             const __half* kernel,
                                             float4* __restrict                dst,
                                             const uint              pitch_src,
                                             const uint              pitch_dst,
                                             const uint2           kernel_dims,
                                             const uint2              kernel_shift,
                                             const uint2             dst_dims)
{
#if __ABOVE_SM_53
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t idy = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ float4 src_frag[48][144 / 8 + 1];

    decx::utils::_cuda_vec128 reg_0, reg_1, _accu[2];
    const uint32_t _k_loop_w_v4 = _get_k_loop_w_v8(_MAX_SLIDE_CONV_R8_V8_FP16_, kernel_shift.y);
    const uint8_t _L = kernel_shift.y % 8;

    uint64_t glo_dex = idy * pitch_src + idx;

    for (uint8_t i = 0; i < 3; ++i) {
        reg_0._vf = src[glo_dex];
        src_frag[(i << 4) + threadIdx.y][threadIdx.x] = reg_0._vf;
        glo_dex += (pitch_src << 4);
    }

    if (threadIdx.x < 2) {
        glo_dex = idy * pitch_src + idx + 16;

        for (uint8_t i = 0; i < 3; ++i) {
            reg_0._vf = src[glo_dex];
            src_frag[(i << 4) + threadIdx.y][16 + threadIdx.x] = reg_0._vf;
            glo_dex += (pitch_src << 4);
        }
    }

    __syncthreads();

    _accu[0]._vf = decx::utils::vec4_set1_fp32(0.f);
    _accu[1]._vf = decx::utils::vec4_set1_fp32(0.f);

    for (int i = 0; i < kernel_dims.y; ++i)
    {
        const __half* _row_kernel_ptr = kernel + i * kernel_dims.x;
        uint8_t _row_offset = (kernel_shift.y >> 8);

        reg_0._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset];
        reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];

        const uint8_t _inc = decx::conv::GPUK::_pre_residual_v8_conv_fp16_accu(reg_0, reg_1, _accu, _row_kernel_ptr, _L);
        _row_offset += (bool)(kernel_shift.y % 8);
        _row_kernel_ptr += _inc;

        for (int j = 0; j < _k_loop_w_v4; ++j)
        {
            if (j > 0 || _L) {
                reg_0._vf = reg_1._vf;
                reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];
            }

            decx::conv::GPUK::_full_v8_conv_fp16_accu(reg_0, reg_1, _accu, _row_kernel_ptr);

            _row_kernel_ptr += 8;
            ++_row_offset;
        }

        if (_L) {
            reg_0._vf = reg_1._vf;
            reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];
            decx::conv::GPUK::_post_residual_v8_conv_fp16_accu(reg_0, reg_1, _accu, _row_kernel_ptr, 8 - _L);
        }
    }

    reg_0._arrh2[0] = __floats2half2_rn(_accu[0]._vf.x, _accu[0]._vf.y);
    reg_0._arrh2[1] = __floats2half2_rn(_accu[0]._vf.z, _accu[0]._vf.w);
    reg_0._arrh2[2] = __floats2half2_rn(_accu[1]._vf.x, _accu[1]._vf.y);
    reg_0._arrh2[3] = __floats2half2_rn(_accu[1]._vf.z, _accu[1]._vf.w);

    glo_dex = idy * pitch_dst + idx;
    if (idy < dst_dims.y && idx < dst_dims.x) {
        dst[glo_dex] = reg_0._vf;
    }
#endif
}