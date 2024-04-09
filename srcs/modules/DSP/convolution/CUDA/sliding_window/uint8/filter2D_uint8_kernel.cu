/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "../filter2D_kernel.cuh"


namespace decx
{
namespace dsp {
namespace GPUK {
    __device__ __inline__ static void 
    _conv_u8_sw8_v8(decx::utils::_cuda_vec64& reg0, 
                    decx::utils::_cuda_vec64& reg1, 
                    decx::utils::_cuda_vec128 accu[2], 
                    const float* _kernel_row_ptr)
    {
#pragma unroll 8
        for (uint8_t i = 0; i < 8; ++i) 
        {
            reg0._v_uint8[i] = reg1._v_uint8[i];

            accu[0]._arrf[0] = __fmaf_rn(__int2float_rn(reg0._v_uint8[(uint8_t)((i+1)&7)]),
                _kernel_row_ptr[i], accu[0]._arrf[0]);
            accu[0]._arrf[1] = __fmaf_rn(__int2float_rn(reg0._v_uint8[(uint8_t)((i+2)&7)]),
                _kernel_row_ptr[i], accu[0]._arrf[1]);
            accu[0]._arrf[2] = __fmaf_rn(__int2float_rn(reg0._v_uint8[(uint8_t)((i+3)&7)]),
                _kernel_row_ptr[i], accu[0]._arrf[2]);
            accu[0]._arrf[3] = __fmaf_rn(__int2float_rn(reg0._v_uint8[(uint8_t)((i+4)&7)]),
                _kernel_row_ptr[i], accu[0]._arrf[3]);
            accu[1]._arrf[0] = __fmaf_rn(__int2float_rn(reg0._v_uint8[(uint8_t)((i+5)&7)]),
                _kernel_row_ptr[i], accu[1]._arrf[0]);
            accu[1]._arrf[1] = __fmaf_rn(__int2float_rn(reg0._v_uint8[(uint8_t)((i+6)&7)]),
                _kernel_row_ptr[i], accu[1]._arrf[1]);
            accu[1]._arrf[2] = __fmaf_rn(__int2float_rn(reg0._v_uint8[(uint8_t)((i+7)&7)]),
                _kernel_row_ptr[i], accu[1]._arrf[2]);
            accu[1]._arrf[3] = __fmaf_rn(__int2float_rn(reg0._v_uint8[(uint8_t)((i+8)&7)]),
                _kernel_row_ptr[i], accu[1]._arrf[3]);
        }
    }

}
}
}



template <uint32_t _ext_w> __global__ void 
decx::dsp::GPUK::cu_filter2D_NB_u8_Kfp32_fp32(const double* __restrict src, 
                                             const float* __restrict kernel, 
                                             float4* __restrict dst,
                                             const uint32_t pitchsrc_v8, 
                                             const uint32_t pitchdst_v8, 
                                             const uint3 kernel_dims, 
                                             const uint2 conv_area)
{
    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    uint64_t dex_src = tidx + tidy * (uint64_t)pitchsrc_v8;

    const uint2 _ldg_bound_v8 = make_uint2(decx::utils::ceil<uint32_t>(conv_area.x + kernel_dims.x - 1, 8),
        conv_area.y + kernel_dims.y - 1);

    constexpr uint32_t _ext_w_v8 = _ext_w / 8;
    constexpr uint32_t _row_cover_x_v8 = _CU_FILTER2D_FP32_BLOCK_X_ * 8 / 8;
    __shared__ double _row[_CU_FILTER2D_FP32_BLOCK_Y_][_row_cover_x_v8 + _ext_w_v8 + 1];

    const uint32_t _k_loop_w_v8 = decx::utils::ceil<uint32_t>(kernel_dims.x - 1, 8);
    const uint32_t _k_loop_w_L = (kernel_dims.x - 1) % 8 ? (kernel_dims.x - 1) % 8 : 8;
    decx::utils::_cuda_vec64 _recv[2];
    _recv[0]._vf2 = decx::utils::vec2_set1_fp32(0);
    _recv[1]._vf2 = decx::utils::vec2_set1_fp32(0);

    // Load from global L2
    if (tidy < _ldg_bound_v8.y) {
        if (tidx < _ldg_bound_v8.x) _recv[0]._fp64 = src[dex_src];
    }
    _row[threadIdx.y][threadIdx.x] = _recv[0]._fp64;
    if (tidy < _ldg_bound_v8.y) {
        if (tidx + _row_cover_x_v8 < _ldg_bound_v8.x)  _recv[1]._fp64 = src[dex_src + _row_cover_x_v8];
    }
    if (threadIdx.x < _ext_w_v8) {
        _row[threadIdx.y][threadIdx.x + _row_cover_x_v8] = _recv[1]._fp64;
    }
    // End of load from global L2

    __syncthreads();

    decx::utils::_cuda_vec128 _accu[2];
    _accu[0]._vf = decx::utils::vec4_set1_fp32(0);
    _accu[1]._vf = decx::utils::vec4_set1_fp32(0);

    for (uint32_t i = 0; i < kernel_dims.y; ++i) 
    {
        // The updated fresh data is mapped to the last warp anyway, so don't need to sync threads after the refresh.
        const uint8_t mapping_shmem_idy = (uint8_t)((threadIdx.y + i) & 7);

        if (i > 0) {
            if (threadIdx.y == _CU_FILTER2D_FP32_BLOCK_Y_ - 1) {    // Let the last warp to load from the new row
                if (tidx < _ldg_bound_v8.x && tidy + i < _ldg_bound_v8.y) {
                    _recv[0]._fp64 = src[dex_src + i * pitchsrc_v8];
                }
                _row[mapping_shmem_idy][threadIdx.x] = _recv[0]._fp64;
                _recv[0]._vf2 = decx::utils::vec2_set1_fp32(0);
                if (tidx + _row_cover_x_v8 < _ldg_bound_v8.x && tidy + i < _ldg_bound_v8.y) {
                    _recv[0]._fp64 = src[dex_src + i * pitchsrc_v8 + _row_cover_x_v8];
                }
                if (threadIdx.x < _ext_w_v8) {
                    _row[mapping_shmem_idy][threadIdx.x + _row_cover_x_v8] = _recv[0]._fp64;
                }
            }
        }

        const float* _row_kernel_ptr = kernel + i * kernel_dims.z;

        _recv[0]._fp64 = _row[mapping_shmem_idy][threadIdx.x];
        _accu[0]._vf = decx::utils::cuda::__fmaf_v4_v1_v4_u8(*((uchar4*)&_recv[0]._vui2.x), _row_kernel_ptr[0], _accu[0]._vf);
        _accu[1]._vf = decx::utils::cuda::__fmaf_v4_v1_v4_u8(*((uchar4*)&_recv[0]._vui2.y), _row_kernel_ptr[0], _accu[1]._vf);
        ++_row_kernel_ptr;

        for (uint32_t j = 0; j < _k_loop_w_v8; ++j)
        {
            _recv[1]._fp64 = _row[mapping_shmem_idy][threadIdx.x + j + 1];
            decx::dsp::GPUK::_conv_u8_sw8_v8(_recv[0], _recv[1], _accu, _row_kernel_ptr);
            _row_kernel_ptr += 8;
        }
        __syncthreads();
    }

    uint64_t dex_dst = tidx + tidy * (uint64_t)pitchdst_v8;
    if (tidx < decx::utils::ceil<uint32_t>(conv_area.x, 8) && tidy < conv_area.y) {
        dst[dex_dst * 2] = _accu[0]._vf;
        dst[dex_dst * 2 + 1] = _accu[1]._vf;
    }
}


template __global__ void decx::dsp::GPUK::cu_filter2D_NB_u8_Kfp32_fp32<32>(const double* __restrict, const float* __restrict,
    float4* __restrict, const uint32_t, const uint32_t, const uint3, const uint2);

template __global__ void decx::dsp::GPUK::cu_filter2D_NB_u8_Kfp32_fp32<24>(const double* __restrict, const float* __restrict,
    float4* __restrict, const uint32_t, const uint32_t, const uint3, const uint2);

template __global__ void decx::dsp::GPUK::cu_filter2D_NB_u8_Kfp32_fp32<16>(const double* __restrict, const float* __restrict,
    float4* __restrict, const uint32_t, const uint32_t, const uint3, const uint2);

template __global__ void decx::dsp::GPUK::cu_filter2D_NB_u8_Kfp32_fp32<8>(const double* __restrict, const float* __restrict,
    float4* __restrict, const uint32_t, const uint32_t, const uint3, const uint2);



template <uint32_t _ext_w> __global__ void 
decx::dsp::GPUK::cu_filter2D_NB_u8_Kfp32_u8(const double* __restrict src, 
                                           const float* __restrict kernel, 
                                           double* __restrict dst,
                                           const uint32_t pitchsrc_v8, 
                                           const uint32_t pitchdst_v8, 
                                           const uint3 kernel_dims, 
                                           const uint2 conv_area)
{
    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    uint64_t dex_src = tidx + tidy * (uint64_t)pitchsrc_v8;

    const uint2 _ldg_bound_v8 = make_uint2(decx::utils::ceil<uint32_t>(conv_area.x + kernel_dims.x - 1, 8),
        conv_area.y + kernel_dims.y - 1);

    constexpr uint32_t _ext_w_v8 = _ext_w / 8;
    constexpr uint32_t _row_cover_x_v8 = _CU_FILTER2D_FP32_BLOCK_X_ * 8 / 8;
    __shared__ double _row[_CU_FILTER2D_FP32_BLOCK_Y_][_row_cover_x_v8 + _ext_w_v8 + 1];

    const uint32_t _k_loop_w_v8 = decx::utils::ceil<uint32_t>(kernel_dims.x - 1, 8);
    const uint32_t _k_loop_w_L = (kernel_dims.x - 1) % 8 ? (kernel_dims.x - 1) % 8 : 8;
    decx::utils::_cuda_vec64 _recv[2];
    _recv[0]._vf2 = decx::utils::vec2_set1_fp32(0);
    _recv[1]._vf2 = decx::utils::vec2_set1_fp32(0);

    // Load from global L2
    if (tidy < _ldg_bound_v8.y) {
        if (tidx < _ldg_bound_v8.x) _recv[0]._fp64 = src[dex_src];
    }
    _row[threadIdx.y][threadIdx.x] = _recv[0]._fp64;
    if (tidy < _ldg_bound_v8.y) {
        if (tidx + _row_cover_x_v8 < _ldg_bound_v8.x)  _recv[1]._fp64 = src[dex_src + _row_cover_x_v8];
    }
    if (threadIdx.x < _ext_w_v8) {
        _row[threadIdx.y][threadIdx.x + _row_cover_x_v8] = _recv[1]._fp64;
    }
    // End of load from global L2

    __syncthreads();

    decx::utils::_cuda_vec128 _accu[2];
    _accu[0]._vf = decx::utils::vec4_set1_fp32(0);
    _accu[1]._vf = decx::utils::vec4_set1_fp32(0);

    for (uint32_t i = 0; i < kernel_dims.y; ++i) 
    {
        // The updated fresh data is mapped to the last warp anyway, so don't need to sync threads after the refresh.
        const uint8_t mapping_shmem_idy = (uint8_t)((threadIdx.y + i) & 7);

        if (i > 0) {
            if (threadIdx.y == _CU_FILTER2D_FP32_BLOCK_Y_ - 1) {    // Let the last warp to load from the new row
                if (tidx < _ldg_bound_v8.x && tidy + i < _ldg_bound_v8.y) {
                    _recv[0]._fp64 = src[dex_src + i * pitchsrc_v8];
                }
                _row[mapping_shmem_idy][threadIdx.x] = _recv[0]._fp64;
                _recv[0]._vf2 = decx::utils::vec2_set1_fp32(0);
                if (tidx + _row_cover_x_v8 < _ldg_bound_v8.x && tidy + i < _ldg_bound_v8.y) {
                    _recv[0]._fp64 = src[dex_src + i * pitchsrc_v8 + _row_cover_x_v8];
                }
                if (threadIdx.x < _ext_w_v8) {
                    _row[mapping_shmem_idy][threadIdx.x + _row_cover_x_v8] = _recv[0]._fp64;
                }
            }
        }

        const float* _row_kernel_ptr = kernel + i * kernel_dims.z;

        _recv[0]._fp64 = _row[mapping_shmem_idy][threadIdx.x];
        _accu[0]._vf = decx::utils::cuda::__fmaf_v4_v1_v4_u8(*((uchar4*)&_recv[0]._vui2.x), _row_kernel_ptr[0], _accu[0]._vf);
        _accu[1]._vf = decx::utils::cuda::__fmaf_v4_v1_v4_u8(*((uchar4*)&_recv[0]._vui2.y), _row_kernel_ptr[0], _accu[1]._vf);
        ++_row_kernel_ptr;

        for (uint32_t j = 0; j < _k_loop_w_v8; ++j)
        {
            _recv[1]._fp64 = _row[mapping_shmem_idy][threadIdx.x + j + 1];
            decx::dsp::GPUK::_conv_u8_sw8_v8(_recv[0], _recv[1], _accu, _row_kernel_ptr);
            _row_kernel_ptr += 8;
        }
        __syncthreads();
    }

    uint64_t dex_dst = tidx + tidy * (uint64_t)pitchdst_v8;
    if (tidx < decx::utils::ceil<uint32_t>(conv_area.x, 8) && tidy < conv_area.y) 
    {
        _recv[0]._v_uint8[0] = __float2int_rn(_accu[0]._vf.x);
        _recv[0]._v_uint8[1] = __float2int_rn(_accu[0]._vf.y);
        _recv[0]._v_uint8[2] = __float2int_rn(_accu[0]._vf.z);
        _recv[0]._v_uint8[3] = __float2int_rn(_accu[0]._vf.w);
        _recv[0]._v_uint8[4] = __float2int_rn(_accu[1]._vf.x);
        _recv[0]._v_uint8[5] = __float2int_rn(_accu[1]._vf.y);
        _recv[0]._v_uint8[6] = __float2int_rn(_accu[1]._vf.z);
        _recv[0]._v_uint8[7] = __float2int_rn(_accu[1]._vf.w);
        dst[dex_dst] = _recv[0]._fp64;
    }
}

template __global__ void decx::dsp::GPUK::cu_filter2D_NB_u8_Kfp32_u8<32>(const double* __restrict, const float* __restrict,
    double* __restrict, const uint32_t, const uint32_t, const uint3, const uint2);

template __global__ void decx::dsp::GPUK::cu_filter2D_NB_u8_Kfp32_u8<24>(const double* __restrict, const float* __restrict,
    double* __restrict, const uint32_t, const uint32_t, const uint3, const uint2);

template __global__ void decx::dsp::GPUK::cu_filter2D_NB_u8_Kfp32_u8<16>(const double* __restrict, const float* __restrict,
    double* __restrict, const uint32_t, const uint32_t, const uint3, const uint2);

template __global__ void decx::dsp::GPUK::cu_filter2D_NB_u8_Kfp32_u8<8>(const double* __restrict, const float* __restrict,
    double* __restrict, const uint32_t, const uint32_t, const uint3, const uint2);



template <uint32_t _ext_w> __global__ void 
decx::dsp::GPUK::cu_filter2D_BC_u8_Kfp32_fp32(const double* __restrict src,
                                             const float* __restrict kernel, 
                                             float4* __restrict dst,
                                             const uint32_t pitchsrc_v8, 
                                             const uint32_t pitchdst_v8, 
                                             const uint3 kernel_dims, 
                                             const uint2 conv_area)
{
    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _ldg_bound_x = decx::utils::ceil<uint32_t>(conv_area.x + kernel_dims.x - 1, 8);
    const uint32_t _KH_half = (kernel_dims.y >> 1);
    int64_t dex_src = tidx + tidy * (int64_t)pitchsrc_v8 - _KH_half * pitchsrc_v8;

    constexpr uint32_t _ext_w_v8 = _ext_w / 8;
    constexpr uint32_t _row_cover_x_v8 = _CU_FILTER2D_FP32_BLOCK_X_ * 8 / 8;
    __shared__ double _row[_CU_FILTER2D_FP32_BLOCK_Y_][_row_cover_x_v8 + _ext_w_v8 + 1];

    const uint32_t _k_loop_w_v8 = decx::utils::ceil<uint32_t>(kernel_dims.x - 1, 8);
    const uint32_t _k_loop_w_L = (kernel_dims.x - 1) % 8 ? (kernel_dims.x - 1) % 8 : 8;
    decx::utils::_cuda_vec64 _recv[2];
    _recv[0]._vf2 = decx::utils::vec2_set1_fp32(0);
    _recv[1]._vf2 = decx::utils::vec2_set1_fp32(0);

    // Load from global L2
    if (tidy < conv_area.y + _KH_half && tidy > _KH_half) {
        if (tidx < _ldg_bound_x) _recv[0]._fp64 = src[dex_src];
    }
    _row[threadIdx.y][threadIdx.x] = _recv[0]._fp64;
    if (tidy < conv_area.y + _KH_half && tidy > _KH_half) {
        if (tidx + _row_cover_x_v8 < _ldg_bound_x)  _recv[1]._fp64 = src[dex_src + _row_cover_x_v8];
    }
    if (threadIdx.x < _ext_w_v8) {
        _row[threadIdx.y][threadIdx.x + _row_cover_x_v8] = _recv[1]._fp64;
    }
    // End of load from global L2

    __syncthreads();

    decx::utils::_cuda_vec128 _accu[2];
    _accu[0]._vf = decx::utils::vec4_set1_fp32(0);
    _accu[1]._vf = decx::utils::vec4_set1_fp32(0);

    for (int32_t i = 0; i < kernel_dims.y; ++i)
    {
        // The updated fresh data is mapped to the last warp anyway, so don't need to sync threads after the refresh.
        const uint8_t mapping_shmem_idy = (uint8_t)((threadIdx.y + i) & 7);

        if (tidy + i > _KH_half && tidy + i < conv_area.y + _KH_half)
        {
            if (i > 0) {
                if (threadIdx.y == _CU_FILTER2D_FP32_BLOCK_Y_ - 1) {    // Let the last warp to load from the new row
                    if (tidx < _ldg_bound_x) {
                        _recv[0]._fp64 = src[dex_src + i * pitchsrc_v8];
                    }
                    _row[mapping_shmem_idy][threadIdx.x] = _recv[0]._fp64;
                    _recv[0]._vf2 = decx::utils::vec2_set1_fp32(0);
                    if (tidx + _row_cover_x_v8 < _ldg_bound_x) {
                        _recv[0]._fp64 = src[dex_src + i * pitchsrc_v8 + _row_cover_x_v8];
                    }
                    if (threadIdx.x < _ext_w_v8) {
                        _row[mapping_shmem_idy][threadIdx.x + _row_cover_x_v8] = _recv[0]._fp64;
                    }
                }
            }

            const float* _row_kernel_ptr = kernel + i * kernel_dims.z;

            _recv[0]._fp64 = _row[mapping_shmem_idy][threadIdx.x];
            _accu[0]._vf = decx::utils::cuda::__fmaf_v4_v1_v4_u8(*((uchar4*)&_recv[0]._vui2.x), _row_kernel_ptr[0], _accu[0]._vf);
            _accu[1]._vf = decx::utils::cuda::__fmaf_v4_v1_v4_u8(*((uchar4*)&_recv[0]._vui2.y), _row_kernel_ptr[0], _accu[1]._vf);
            ++_row_kernel_ptr;

            for (uint32_t j = 0; j < _k_loop_w_v8; ++j)
            {
                _recv[1]._fp64 = _row[mapping_shmem_idy][threadIdx.x + j + 1];
                decx::dsp::GPUK::_conv_u8_sw8_v8(_recv[0], _recv[1], _accu, _row_kernel_ptr);
                _row_kernel_ptr += 8;
            }
        }
        __syncthreads();
    }

    uint64_t dex_dst = tidx + tidy * (uint64_t)pitchdst_v8;
    if (tidx < decx::utils::ceil<uint32_t>(conv_area.x, 8) && tidy < conv_area.y) {
        dst[dex_dst * 2] = _accu[0]._vf;
        dst[dex_dst * 2 + 1] = _accu[1]._vf;
    }
}

template __global__ void decx::dsp::GPUK::cu_filter2D_BC_u8_Kfp32_fp32<32>(const double* __restrict, const float* __restrict,
    float4* __restrict, const uint32_t, const uint32_t, const uint3, const uint2);

template __global__ void decx::dsp::GPUK::cu_filter2D_BC_u8_Kfp32_fp32<24>(const double* __restrict, const float* __restrict,
    float4* __restrict, const uint32_t, const uint32_t, const uint3, const uint2);

template __global__ void decx::dsp::GPUK::cu_filter2D_BC_u8_Kfp32_fp32<16>(const double* __restrict, const float* __restrict,
    float4* __restrict, const uint32_t, const uint32_t, const uint3, const uint2);

template __global__ void decx::dsp::GPUK::cu_filter2D_BC_u8_Kfp32_fp32<8>(const double* __restrict, const float* __restrict,
    float4* __restrict, const uint32_t, const uint32_t, const uint3, const uint2);




template <uint32_t _ext_w> __global__ void 
decx::dsp::GPUK::cu_filter2D_BC_u8_Kfp32_u8(const double* __restrict src,
                                            const float* __restrict kernel, 
                                            double* __restrict dst,
                                            const uint32_t pitchsrc_v8, 
                                            const uint32_t pitchdst_v8, 
                                            const uint3 kernel_dims, 
                                            const uint2 conv_area)
{
    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _ldg_bound_x = decx::utils::ceil<uint32_t>(conv_area.x + kernel_dims.x - 1, 8);
    const uint32_t _KH_half = (kernel_dims.y >> 1);
    int64_t dex_src = tidx + tidy * (int64_t)pitchsrc_v8 - _KH_half * pitchsrc_v8;

    constexpr uint32_t _ext_w_v8 = _ext_w / 8;
    constexpr uint32_t _row_cover_x_v8 = _CU_FILTER2D_FP32_BLOCK_X_ * 8 / 8;
    __shared__ double _row[_CU_FILTER2D_FP32_BLOCK_Y_][_row_cover_x_v8 + _ext_w_v8 + 1];

    const uint32_t _k_loop_w_v8 = decx::utils::ceil<uint32_t>(kernel_dims.x - 1, 8);
    const uint32_t _k_loop_w_L = (kernel_dims.x - 1) % 8 ? (kernel_dims.x - 1) % 8 : 8;
    decx::utils::_cuda_vec64 _recv[2];
    _recv[0]._vf2 = decx::utils::vec2_set1_fp32(0);
    _recv[1]._vf2 = decx::utils::vec2_set1_fp32(0);

    // Load from global L2
    if (tidy < conv_area.y + _KH_half && tidy > _KH_half) {
        if (tidx < _ldg_bound_x) _recv[0]._fp64 = src[dex_src];
    }
    _row[threadIdx.y][threadIdx.x] = _recv[0]._fp64;
    if (tidy < conv_area.y + _KH_half && tidy > _KH_half) {
        if (tidx + _row_cover_x_v8 < _ldg_bound_x)  _recv[1]._fp64 = src[dex_src + _row_cover_x_v8];
    }
    if (threadIdx.x < _ext_w_v8) {
        _row[threadIdx.y][threadIdx.x + _row_cover_x_v8] = _recv[1]._fp64;
    }
    // End of load from global L2

    __syncthreads();

    decx::utils::_cuda_vec128 _accu[2];
    _accu[0]._vf = decx::utils::vec4_set1_fp32(0);
    _accu[1]._vf = decx::utils::vec4_set1_fp32(0);

    for (int32_t i = 0; i < kernel_dims.y; ++i)
    {
        // The updated fresh data is mapped to the last warp anyway, so don't need to sync threads after the refresh.
        const uint8_t mapping_shmem_idy = (uint8_t)((threadIdx.y + i) & 7);

        if (tidy + i > _KH_half && tidy + i < conv_area.y + _KH_half)
        {
            if (i > 0) {
                if (threadIdx.y == _CU_FILTER2D_FP32_BLOCK_Y_ - 1) {    // Let the last warp to load from the new row
                    if (tidx < _ldg_bound_x) {
                        _recv[0]._fp64 = src[dex_src + i * pitchsrc_v8];
                    }
                    _row[mapping_shmem_idy][threadIdx.x] = _recv[0]._fp64;
                    _recv[0]._vf2 = decx::utils::vec2_set1_fp32(0);
                    if (tidx + _row_cover_x_v8 < _ldg_bound_x) {
                        _recv[0]._fp64 = src[dex_src + i * pitchsrc_v8 + _row_cover_x_v8];
                    }
                    if (threadIdx.x < _ext_w_v8) {
                        _row[mapping_shmem_idy][threadIdx.x + _row_cover_x_v8] = _recv[0]._fp64;
                    }
                }
            }

            const float* _row_kernel_ptr = kernel + i * kernel_dims.z;

            _recv[0]._fp64 = _row[mapping_shmem_idy][threadIdx.x];
            _accu[0]._vf = decx::utils::cuda::__fmaf_v4_v1_v4_u8(*((uchar4*)&_recv[0]._vui2.x), _row_kernel_ptr[0], _accu[0]._vf);
            _accu[1]._vf = decx::utils::cuda::__fmaf_v4_v1_v4_u8(*((uchar4*)&_recv[0]._vui2.y), _row_kernel_ptr[0], _accu[1]._vf);
            ++_row_kernel_ptr;

            for (uint32_t j = 0; j < _k_loop_w_v8; ++j)
            {
                _recv[1]._fp64 = _row[mapping_shmem_idy][threadIdx.x + j + 1];
                decx::dsp::GPUK::_conv_u8_sw8_v8(_recv[0], _recv[1], _accu, _row_kernel_ptr);
                _row_kernel_ptr += 8;
            }
        }
        __syncthreads();
    }

    uint64_t dex_dst = tidx + tidy * (uint64_t)pitchdst_v8;
    if (tidx < decx::utils::ceil<uint32_t>(conv_area.x, 8) && tidy < conv_area.y) 
    {
        _recv[0]._v_uint8[0] = __float2int_rn(_accu[0]._vf.x);
        _recv[0]._v_uint8[1] = __float2int_rn(_accu[0]._vf.y);
        _recv[0]._v_uint8[2] = __float2int_rn(_accu[0]._vf.z);
        _recv[0]._v_uint8[3] = __float2int_rn(_accu[0]._vf.w);
        _recv[0]._v_uint8[4] = __float2int_rn(_accu[1]._vf.x);
        _recv[0]._v_uint8[5] = __float2int_rn(_accu[1]._vf.y);
        _recv[0]._v_uint8[6] = __float2int_rn(_accu[1]._vf.z);
        _recv[0]._v_uint8[7] = __float2int_rn(_accu[1]._vf.w);
        dst[dex_dst] = _recv[0]._fp64;
    }
}

template __global__ void decx::dsp::GPUK::cu_filter2D_BC_u8_Kfp32_u8<32>(const double* __restrict, const float* __restrict,
    double* __restrict, const uint32_t, const uint32_t, const uint3, const uint2);

template __global__ void decx::dsp::GPUK::cu_filter2D_BC_u8_Kfp32_u8<24>(const double* __restrict, const float* __restrict,
    double* __restrict, const uint32_t, const uint32_t, const uint3, const uint2);

template __global__ void decx::dsp::GPUK::cu_filter2D_BC_u8_Kfp32_u8<16>(const double* __restrict, const float* __restrict,
    double* __restrict, const uint32_t, const uint32_t, const uint3, const uint2);

template __global__ void decx::dsp::GPUK::cu_filter2D_BC_u8_Kfp32_u8<8>(const double* __restrict, const float* __restrict,
    double* __restrict, const uint32_t, const uint32_t, const uint3, const uint2);

