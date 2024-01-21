/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "Conv2_fp32_kernels.cuh"
#include "../../../../../core/utils/decx_cuda_math_functions.cuh"


#define _MAX_SLIDE_CONV_R16_V4_FP32_ 8
#define _MAX_SLIDE_CONV_R8_V4_FP32_ 4


#define _get_k_loop_w_v4(_max_slide, kernel_shift_W) (_max_slide - 2 * decx::utils::ceil<uint32_t>(kernel_shift_W, 4))

#define _fma_v4_v1_v4_fp32_shifted(_base, reg_0, _accu, kernel_val) {                               \
    for (uint8_t j = 0; j < 4; ++j) {                                                               \
        _accu._arrf[j] = __fmaf_rn(reg_0._arrf[((_base) + j) & 0x3], kernel_val, _accu._arrf[j]);   \
    }                                                                                               \
}


namespace decx
{
namespace conv 
{
namespace GPUK
{
    __device__ __inline__ static uint8_t
    _pre_residual_v4_conv_fp32(decx::utils::_cuda_vec128& reg_0, decx::utils::_cuda_vec128& reg_1, decx::utils::_cuda_vec128& _accu,
        const float* kernel, const uint8_t _offset)
    {
        float tmp_ker;

        if (_offset == 0) {
            tmp_ker = kernel[0];
            _accu._vf = decx::utils::cuda::__fmaf_v4_v1_v4(reg_0._vf, tmp_ker, _accu._vf);

            return 1;
        }
        for (uint8_t k = 0; k < 4; ++k) {
            tmp_ker = (k < _offset - 1) ? 0 : kernel[k - _offset + 1];
            reg_0._arrf[k] = reg_1._arrf[k];
            _fma_v4_v1_v4_fp32_shifted(k + 1, reg_0, _accu, tmp_ker);
        }

        return ((4 - _offset) + 1);
    }



    __device__ __inline__ static void
    _post_residual_v4_conv_fp32(decx::utils::_cuda_vec128& reg_0, decx::utils::_cuda_vec128& reg_1, decx::utils::_cuda_vec128& _accu,
            const float* kernel, const uint8_t _left)
    {
        float tmp_ker;
        
        for (uint8_t i = 1; i < 5; ++i) {
            tmp_ker = (i - 1 < _left) ? kernel[i - 1] : 0;
            reg_0._arrf[i - 1] = reg_1._arrf[i - 1];
            _fma_v4_v1_v4_fp32_shifted(i, reg_0, _accu, tmp_ker);
        }
    }


    __device__ __inline__ static void
    _full_v4_conv_fp32(decx::utils::_cuda_vec128& reg_0, decx::utils::_cuda_vec128& reg_1, decx::utils::_cuda_vec128& _accu, const float* kernel)
    {
        float tmp_ker;

        for (uint8_t i = 1; i < 5; ++i) 
        {
            tmp_ker = kernel[i - 1];
            reg_0._arrf[i - 1] = reg_1._arrf[i - 1];
            _fma_v4_v1_v4_fp32_shifted(i, reg_0, _accu, tmp_ker);
        }
    }
}
}
}



__global__ void 
decx::conv::GPUK::cu_sConv2_r8_within(const float4* __restrict   src, 
                                      const float* __restrict    kernel,
                                      float4* __restrict         dst,
                                      const uint32_t             pitch_src, 
                                      const uint32_t             pitch_dst,
                                      const uint2                kernel_dims,
                                      const uint2                kernel_shift,
                                      const uint2                dst_dims)
{
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t idy = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ float4 src_frag[32][80 / 4 + sharedmem_offset];

    decx::utils::_cuda_vec128 reg_0, reg_1, _accu;
    const uint32_t _k_loop_w_v4 = _get_k_loop_w_v4(_MAX_SLIDE_CONV_R8_V4_FP32_, kernel_shift.y);
    const uint8_t _L = kernel_shift.y % 4;

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

    _accu._vf = decx::utils::vec4_set1_fp32(0.f);

    for (int i = 0; i < kernel_dims.y; ++i)
    {
        const float* _row_kernel_ptr = kernel + i * kernel_dims.x;
        uint8_t _row_offset = (kernel_shift.y >> 2);

        reg_0._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset];
        reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];

        const uint8_t _inc = decx::conv::GPUK::_pre_residual_v4_conv_fp32(reg_0, reg_1, _accu, _row_kernel_ptr, _L);
        _row_offset += (bool)(kernel_shift.y % 4);
        _row_kernel_ptr += _inc;

        for (int j = 0; j < _k_loop_w_v4; ++j)
        {
            if (j > 0 || _L) {
                reg_0._vf = reg_1._vf;
                reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];
            }

            decx::conv::GPUK::_full_v4_conv_fp32(reg_0, reg_1, _accu, _row_kernel_ptr);

            _row_kernel_ptr += 4;
            ++_row_offset;
        }

        if (_L) {
            reg_0._vf = reg_1._vf;
            reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];
            decx::conv::GPUK::_post_residual_v4_conv_fp32(reg_0, reg_1, _accu, _row_kernel_ptr, 4 - _L);
        }
    }

    glo_dex = idy * pitch_dst + idx;
    if (idy < dst_dims.y && idx < dst_dims.x) {
        dst[glo_dex] = _accu._vf;
    }
}



__global__ void 
decx::conv::GPUK::cu_sConv2_r16_within(const float4* __restrict src, 
                                       const float* __restrict  kernel,
                                       float4* __restrict       dst,
                                       const uint32_t           pitch_src, 
                                       const uint32_t           pitch_dst,
                                       const uint2              kernel_dims,
                                       const uint2              kernel_shift,
                                       const uint2              dst_dims)
{
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t idy = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ float4 src_frag[48][96 / 4 + sharedmem_offset];

    decx::utils::_cuda_vec128 reg_0, reg_1, _accu;
    const uint32_t _k_loop_w_v4 = _get_k_loop_w_v4(_MAX_SLIDE_CONV_R16_V4_FP32_, kernel_shift.y);
    const uint8_t _L = kernel_shift.y % 4;

    uint64_t glo_dex = idy * pitch_src + idx;

    for (uint8_t i = 0; i < 3; ++i) {
        reg_0._vf = src[glo_dex];
        src_frag[(i << 4) + threadIdx.y][threadIdx.x] = reg_0._vf;
        glo_dex += (pitch_src << 4);
    }

    if (threadIdx.x < 8) {
        glo_dex = idy * pitch_src + idx + 16;

        for (uint8_t i = 0; i < 3; ++i) {
            reg_0._vf = src[glo_dex];
            src_frag[(i << 4) + threadIdx.y][16 + threadIdx.x] = reg_0._vf;
            glo_dex += (pitch_src << 4);
        }
    }

    __syncthreads();

    _accu._vf = decx::utils::vec4_set1_fp32(0.f);

    for (int i = 0; i < kernel_dims.y; ++i)
    {
        const float* _row_kernel_ptr = kernel + i * kernel_dims.x;
        uint8_t _row_offset = (kernel_shift.y >> 2);

        reg_0._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset];
        reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];

        const uint8_t _inc = decx::conv::GPUK::_pre_residual_v4_conv_fp32(reg_0, reg_1, _accu, _row_kernel_ptr, _L);
        _row_offset += (bool)(kernel_shift.y % 4);
        _row_kernel_ptr += _inc;

        for (int j = 0; j < _k_loop_w_v4; ++j)
        {
            if (j > 0 || _L) {
                reg_0._vf = reg_1._vf;
                reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];
            }

            decx::conv::GPUK::_full_v4_conv_fp32(reg_0, reg_1, _accu, _row_kernel_ptr);

            _row_kernel_ptr += 4;
            ++_row_offset;
        }

        if (kernel_shift.y % 4) {
            reg_0._vf = reg_1._vf;
            reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];
            decx::conv::GPUK::_post_residual_v4_conv_fp32(reg_0, reg_1, _accu, _row_kernel_ptr, 4 - (kernel_shift.y % 4));
        }
    }

    glo_dex = idy * pitch_dst + idx;
    if (idy < dst_dims.y && idx < dst_dims.x) {
        dst[glo_dex] = _accu._vf;
    }
}


// ------------------------------------------------------ kernrels ------------------------------------------------------

__global__ void 
decx::conv::GPUK::cu_sConv2_r816_within(const float4* __restrict    src, 
                                        const float* __restrict     kernel,
                                        float4* __restrict          dst,
                                        const uint32_t              pitch_src, 
                                        const uint32_t              pitch_dst,
                                        const uint2                 kernel_dims,
                                        const uint2                 kernel_shift,
                                        const uint2                 dst_dims)
{
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t idy = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ float4 src_frag[32][96 / 4 + sharedmem_offset];

    decx::utils::_cuda_vec128 reg_0, reg_1, _accu;
    const uint32_t _k_loop_w_v4 = _get_k_loop_w_v4(_MAX_SLIDE_CONV_R16_V4_FP32_, kernel_shift.y);
    const uint8_t _L = kernel_shift.y % 4;

    uint64_t glo_dex = idy * pitch_src + idx;

    for (uint8_t i = 0; i < 2; ++i) {
        reg_0._vf = src[glo_dex];
        src_frag[(i << 4) + threadIdx.y][threadIdx.x] = reg_0._vf;
        glo_dex += (pitch_src << 4);
    }

    if (threadIdx.x < 8) {
        glo_dex = idy * pitch_src + idx + 16;

        for (uint8_t i = 0; i < 2; ++i) {
            reg_0._vf = src[glo_dex];
            src_frag[(i << 4) + threadIdx.y][16 + threadIdx.x] = reg_0._vf;
            glo_dex += (pitch_src << 4);
        }
    }

    __syncthreads();

    _accu._vf = decx::utils::vec4_set1_fp32(0.f);

    for (int i = 0; i < kernel_dims.y; ++i)
    {
        const float* _row_kernel_ptr = kernel + i * kernel_dims.x;
        uint8_t _row_offset = (kernel_shift.y >> 2);

        reg_0._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset];
        reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];

        const uint8_t _inc = decx::conv::GPUK::_pre_residual_v4_conv_fp32(reg_0, reg_1, _accu, _row_kernel_ptr, _L);
        _row_offset += (bool)(kernel_shift.y % 4);
        _row_kernel_ptr += _inc;

        for (int j = 0; j < _k_loop_w_v4; ++j)
        {
            if (j > 0 || _L) {
                reg_0._vf = reg_1._vf;
                reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];
            }

            decx::conv::GPUK::_full_v4_conv_fp32(reg_0, reg_1, _accu, _row_kernel_ptr);

            _row_kernel_ptr += 4;
            ++_row_offset;
        }

        if (_L) {
            reg_0._vf = reg_1._vf;
            reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];
            decx::conv::GPUK::_post_residual_v4_conv_fp32(reg_0, reg_1, _accu, _row_kernel_ptr, 4 - _L);
        }
    }

    glo_dex = idy * pitch_dst + idx;
    if (idy < dst_dims.y && idx < dst_dims.x) {
        dst[glo_dex] = _accu._vf;
    }
}





__global__ void 
decx::conv::GPUK::cu_sConv2_r168_within(const float4* __restrict    src, 
                                        const float* __restrict     kernel,
                                        float4* __restrict          dst,
                                        const uint32_t              pitch_src, 
                                        const uint32_t              pitch_dst,
                                        const uint2                 kernel_dims,
                                        const uint2                 kernel_shift,
                                        const uint2                 dst_dims)
{
    uint32_t idx = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t idy = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ float4 src_frag[48][80 / 4 + sharedmem_offset];

    decx::utils::_cuda_vec128 reg_0, reg_1, _accu;
    const uint32_t _k_loop_w_v4 = _get_k_loop_w_v4(_MAX_SLIDE_CONV_R8_V4_FP32_, kernel_shift.y);
    const uint8_t _L = kernel_shift.y % 4;

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

    _accu._vf = decx::utils::vec4_set1_fp32(0.f);

    for (int i = 0; i < kernel_dims.y; ++i)
    {
        const float* _row_kernel_ptr = kernel + i * kernel_dims.x;
        uint8_t _row_offset = (kernel_shift.y >> 2);

        reg_0._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset];
        reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];

        const uint8_t _inc = decx::conv::GPUK::_pre_residual_v4_conv_fp32(reg_0, reg_1, _accu, _row_kernel_ptr, _L);
        _row_offset += (bool)(kernel_shift.y % 4);
        _row_kernel_ptr += _inc;

        for (int j = 0; j < _k_loop_w_v4; ++j)
        {
            if (j > 0 || _L) {
                reg_0._vf = reg_1._vf;
                reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];
            }

            decx::conv::GPUK::_full_v4_conv_fp32(reg_0, reg_1, _accu, _row_kernel_ptr);

            _row_kernel_ptr += 4;
            ++_row_offset;
        }

        if (_L) {
            reg_0._vf = reg_1._vf;
            reg_1._vf = src_frag[kernel_shift.x + threadIdx.y + i][threadIdx.x + _row_offset + 1];
            decx::conv::GPUK::_post_residual_v4_conv_fp32(reg_0, reg_1, _accu, _row_kernel_ptr, 4 - _L);
        }
    }

    glo_dex = idy * pitch_dst + idx;
    if (idy < dst_dims.y && idx < dst_dims.x) {
        dst[glo_dex] = _accu._vf;
    }
}