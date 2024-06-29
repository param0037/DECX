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


#include "../filter2D_kernel.cuh"


namespace decx
{
namespace dsp {
    namespace GPUK {
        __device__ __inline__ static void 
        _conv_fp32_sw4_v4(decx::utils::_cuda_vec128& reg0, 
                            decx::utils::_cuda_vec128& reg1, 
                            decx::utils::_cuda_vec128& accu, 
                            const float* _kernel_row_ptr)
        {
#pragma unroll 4
            for (uint8_t i = 0; i < 4; ++i) 
            {
                reg0._arrf[i] = reg1._arrf[i];
#pragma unroll 4
                for (uint8_t j = 0; j < 4; ++j) 
                {
                    const uint8_t _map_idx_v4 = (uint8_t)((i + j + 1) & 3);
                    accu._arrf[j] = __fmaf_rn(reg0._arrf[_map_idx_v4], _kernel_row_ptr[i], accu._arrf[j]);
                }
            }
        }

    }
}
}



// block[32, 8]
// block_covered[32 * 4, 8]
template <uint32_t _ext_w> __global__ void
decx::dsp::GPUK::cu_filter2D_NB_fp32(const float4* __restrict  src, 
                                  const float* __restrict   kernel,
                                  float4* __restrict        dst,
                                  const uint32_t            pitchsrc_v4, 
                                  const uint32_t            pitchdst_v4, 
                                  const uint3               kernel_dims,      // [W, H, pitch]
                                  const uint2               conv_area)
{
    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    uint64_t dex_src = tidx + tidy * (uint64_t)pitchsrc_v4;

    const uint2 _ldg_bound_v4 = make_uint2(decx::utils::ceil<uint32_t>(conv_area.x + kernel_dims.x - 1, 4),
        conv_area.y + kernel_dims.y - 1);

    constexpr uint32_t _ext_w_v4 = _ext_w / 4;
    constexpr uint32_t _row_cover_x_v4 = _CU_FILTER2D_FP32_BLOCK_X_ * 4 / 4;
    __shared__ float4 _row[_CU_FILTER2D_FP32_BLOCK_Y_][_row_cover_x_v4 + _ext_w_v4 + 1];

    const uint32_t _k_loop_w_v4 = decx::utils::ceil<uint32_t>(kernel_dims.x - 1, 4);
    const uint32_t _k_loop_w_L = (kernel_dims.x - 1) % 4 ? (kernel_dims.x - 1) % 4 : 4;
    decx::utils::_cuda_vec128 _recv[2];
    _recv[0]._vf = decx::utils::vec4_set1_fp32(0);
    _recv[1]._vf = decx::utils::vec4_set1_fp32(0);

    // Load from global L2
    if (tidy < _ldg_bound_v4.y) {
        if (tidx < _ldg_bound_v4.x) _recv[0]._vf = src[dex_src];
    }
    _row[threadIdx.y][threadIdx.x] = _recv[0]._vf;
    if (tidy < _ldg_bound_v4.y) {
        if (tidx + _row_cover_x_v4 < _ldg_bound_v4.x)  _recv[1]._vf = src[dex_src + _row_cover_x_v4];
    }
    if (threadIdx.x < _ext_w_v4) {
        _row[threadIdx.y][threadIdx.x + _row_cover_x_v4] = _recv[1]._vf;
    }
    // End of load from global L2

    __syncthreads();

    decx::utils::_cuda_vec128 _accu;
    _accu._vf = decx::utils::vec4_set1_fp32(0);

    for (uint32_t i = 0; i < kernel_dims.y; ++i) 
    {
        // The updated fresh data is mapped to the last warp anyway, so don't need to sync threads after the refresh.
        const uint8_t mapping_shmem_idy = (uint8_t)((threadIdx.y + i) & 7);

        if (i > 0) {
            if (threadIdx.y == _CU_FILTER2D_FP32_BLOCK_Y_ - 1) {    // Let the last warp to load from the new row
                if (tidx < _ldg_bound_v4.x && tidy + i < _ldg_bound_v4.y) {
                    _recv[0]._vf = src[dex_src + i * pitchsrc_v4];
                }
                _row[mapping_shmem_idy][threadIdx.x] = _recv[0]._vf;
                _recv[0]._vf = decx::utils::vec4_set1_fp32(0);
                if (tidx + _row_cover_x_v4 < _ldg_bound_v4.x && tidy + i < _ldg_bound_v4.y) {
                    _recv[0]._vf = src[dex_src + i * pitchsrc_v4 + _row_cover_x_v4];
                }
                if (threadIdx.x < _ext_w_v4) {
                    _row[mapping_shmem_idy][threadIdx.x + _row_cover_x_v4] = _recv[0]._vf;
                }
            }
        }

        const float* _row_kernel_ptr = kernel + i * kernel_dims.z;

        _recv[0]._vf = _row[mapping_shmem_idy][threadIdx.x];
        _accu._vf = decx::utils::cuda::__fmaf_v4_v1_v4(_recv[0]._vf, _row_kernel_ptr[0], _accu._vf);
        ++_row_kernel_ptr;

        for (uint32_t j = 0; j < _k_loop_w_v4; ++j)
        {
            _recv[1]._vf = _row[mapping_shmem_idy][threadIdx.x + j + 1];
            decx::dsp::GPUK::_conv_fp32_sw4_v4(_recv[0], _recv[1], _accu, _row_kernel_ptr);
            _row_kernel_ptr += 4;
        }
        __syncthreads();
    }

    uint64_t dex_dst = tidx + tidy * (uint64_t)pitchdst_v4;
    if (tidx < decx::utils::ceil<uint32_t>(conv_area.x, 4) && tidy < conv_area.y) {
        dst[dex_dst] = _accu._vf;
    }
}


// block[32, 8]
// block_covered[32 * 4, 8]
template <uint32_t _ext_w> __global__ void
decx::dsp::GPUK::cu_filter2D_BC_fp32(const float4* __restrict  src, 
                                     const float* __restrict   kernel,
                                     float4* __restrict        dst,
                                     const uint32_t            pitchsrc_v4, 
                                     const uint32_t            pitchdst_v4, 
                                     const uint3               kernel_dims,      // [W, H, pitch]
                                     const uint2               conv_area)
{
    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _ldg_bound_x = decx::utils::ceil<uint32_t>(conv_area.x + kernel_dims.x - 1, 4);
    const uint32_t _KH_half = (kernel_dims.y >> 1);
    int64_t dex_src = tidx + tidy * (int64_t)pitchsrc_v4 - _KH_half * pitchsrc_v4;

    constexpr uint32_t _ext_w_v4 = _ext_w / 4;
    constexpr uint32_t _row_cover_x_v4 = _CU_FILTER2D_FP32_BLOCK_X_ * 4 / 4;
    __shared__ float4 _row[_CU_FILTER2D_FP32_BLOCK_Y_][_row_cover_x_v4 + _ext_w_v4 + 1];

    const uint32_t _k_loop_w_v4 = decx::utils::ceil<uint32_t>(kernel_dims.x - 1, 4);
    const uint32_t _k_loop_w_L = (kernel_dims.x - 1) % 4 ? (kernel_dims.x - 1) % 4 : 4;
    decx::utils::_cuda_vec128 _recv[2];
    _recv[0]._vf = decx::utils::vec4_set1_fp32(0);
    _recv[1]._vf = decx::utils::vec4_set1_fp32(0);

    // Load from global L2
    if (tidy < conv_area.y + _KH_half && tidy > _KH_half) {
        if (tidx < _ldg_bound_x) _recv[0]._vf = src[dex_src];
    }
    _row[threadIdx.y][threadIdx.x] = _recv[0]._vf;
    if (tidy < conv_area.y + _KH_half && tidy > _KH_half) {
        if (tidx + _row_cover_x_v4 < _ldg_bound_x)  _recv[1]._vf = src[dex_src + _row_cover_x_v4];
    }
    if (threadIdx.x < _ext_w_v4) {
        _row[threadIdx.y][threadIdx.x + _row_cover_x_v4] = _recv[1]._vf;
    }
    // End of load from global L2

    __syncthreads();

    decx::utils::_cuda_vec128 _accu;
    _accu._vf = decx::utils::vec4_set1_fp32(0);

    for (int32_t i = 0; i < kernel_dims.y; ++i)
    {
        // The updated fresh data is mapped to the last warp anyway, so don't need to sync threads after the refresh.
        const uint8_t mapping_shmem_idy = (uint8_t)((threadIdx.y + i) & 7);

        if (tidy + i > _KH_half && tidy + i < conv_area.y + _KH_half)
        {
            if (i > 0) {
                if (threadIdx.y == _CU_FILTER2D_FP32_BLOCK_Y_ - 1) {    // Let the last warp to load from the new row
                    if (tidx < _ldg_bound_x) {
                        _recv[0]._vf = src[dex_src + i * pitchsrc_v4];
                    }
                    _row[mapping_shmem_idy][threadIdx.x] = _recv[0]._vf;
                    _recv[0]._vf = decx::utils::vec4_set1_fp32(0);
                    if (tidx + _row_cover_x_v4 < _ldg_bound_x) {
                        _recv[0]._vf = src[dex_src + i * pitchsrc_v4 + _row_cover_x_v4];
                    }
                    if (threadIdx.x < _ext_w_v4) {
                        _row[mapping_shmem_idy][threadIdx.x + _row_cover_x_v4] = _recv[0]._vf;
                    }
                }
            }

            const float* _row_kernel_ptr = kernel + i * kernel_dims.z;

            _recv[0]._vf = _row[mapping_shmem_idy][threadIdx.x];
            _accu._vf = decx::utils::cuda::__fmaf_v4_v1_v4(_recv[0]._vf, _row_kernel_ptr[0], _accu._vf);
            ++_row_kernel_ptr;

            for (uint32_t j = 0; j < _k_loop_w_v4; ++j)
            {
                _recv[1]._vf = _row[mapping_shmem_idy][threadIdx.x + j + 1];
                decx::dsp::GPUK::_conv_fp32_sw4_v4(_recv[0], _recv[1], _accu, _row_kernel_ptr);
                _row_kernel_ptr += 4;
            }
        }
        __syncthreads();
    }

    uint64_t dex_dst = tidx + tidy * (uint64_t)pitchdst_v4;
    if (tidx < decx::utils::ceil<uint32_t>(conv_area.x, 4) && tidy < conv_area.y) {
        dst[dex_dst] = _accu._vf;
    }
}


_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 4);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 8);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 12);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 16);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 20);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 24);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 28);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 32);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 36);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 40);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 44);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 48);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 52);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 56);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 60);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 64);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 68);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 72);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 76);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 80);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 84);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 88);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 92);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 96);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 100);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 104);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 108);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 112);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 116);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 120);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 124);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp32, float4, float, float4, 128);


_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 4);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 8);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 12);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 16);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 20);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 24);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 28);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 32);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 36);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 40);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 44);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 48);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 52);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 56);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 60);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 64);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 68);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 72);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 76);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 80);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 84);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 88);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 92);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 96);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 100);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 104);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 108);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 112);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 116);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 120);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 124);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_fp32, float4, float, float4, 128);
