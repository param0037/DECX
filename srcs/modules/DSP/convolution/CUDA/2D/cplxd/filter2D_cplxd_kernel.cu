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

#include "../common/cuda_filter2D_planner.cuh"
#include "../common/filter2D_kernel.cuh"
#include "../../../../../../common/CUSV/CUDA_cpd64.cuh"


// block[32, 8]
// block_covered[32 * 2, 8]
template <uint32_t _ext_w> __global__ void
decx::dsp::GPUK::cu_filter2D_NB_cplxd(const double2* __restrict  src, 
                                      const de::CPd* __restrict  kernel,
                                      double2* __restrict        dst,
                                      const uint32_t             pitchsrc_v1, 
                                      const uint32_t             pitchdst_v1, 
                                      const uint3                kernel_dims,      // [W, H, pitch]
                                      const uint2                conv_area)
{
    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    uint64_t dex_src = tidx + tidy * (uint64_t)pitchsrc_v1;

    const uint2 _ldg_bound_v1 = make_uint2(conv_area.x + kernel_dims.x - 1,
                                           conv_area.y + kernel_dims.y - 1);

    __shared__ double2 _row[_CU_FILTER2D_FP32_BLOCK_Y_][_CU_FILTER2D_FP32_BLOCK_X_ + _ext_w + 1];

    decx::utils::_cuda_vec128 _reg;
    _reg._vd = decx::utils::vec2_set1_fp64(0);

    // Load from global L2
    if (tidy < _ldg_bound_v1.y) {
        if (tidx < _ldg_bound_v1.x) _reg._vd = src[dex_src];
    }
    _row[threadIdx.y][threadIdx.x] = _reg._vd;

    _reg._vd = decx::utils::vec2_set1_fp64(0);
    if (tidy < _ldg_bound_v1.y) {
        if (tidx + _CU_FILTER2D_FP32_BLOCK_X_ < _ldg_bound_v1.x)  _reg._vd = src[dex_src + _CU_FILTER2D_FP32_BLOCK_X_];
    }
    if (threadIdx.x < _ext_w) {
        _row[threadIdx.y][threadIdx.x + _CU_FILTER2D_FP32_BLOCK_X_] = _reg._vd;
    }
    // End of load from global L2

    __syncthreads();

    decx::utils::_cuda_vec128 _accu;
    _accu._vd = decx::utils::vec2_set1_fp64(0);

    for (uint32_t i = 0; i < kernel_dims.y; ++i) 
    {
        // The updated fresh data is mapped to the last warp anyway, so don't need to sync threads after the refresh.
        const uint8_t mapping_shmem_idy = (uint8_t)((threadIdx.y + i) & 7);

        if (i > 0) {
            if (threadIdx.y == _CU_FILTER2D_FP32_BLOCK_Y_ - 1) {    // Let the last warp to load from the new row
                if (tidx < _ldg_bound_v1.x && tidy + i < _ldg_bound_v1.y) {
                    _reg._vd = src[dex_src + i * pitchsrc_v1];
                }
                _row[mapping_shmem_idy][threadIdx.x] = _reg._vd;
                _reg._vd = decx::utils::vec2_set1_fp64(0);
                if (tidx + _CU_FILTER2D_FP32_BLOCK_X_ < _ldg_bound_v1.x && tidy + i < _ldg_bound_v1.y) {
                    _reg._vd = src[dex_src + i * pitchsrc_v1 + _CU_FILTER2D_FP32_BLOCK_X_];
                }
                if (threadIdx.x < _ext_w) {
                    _row[mapping_shmem_idy][threadIdx.x + _CU_FILTER2D_FP32_BLOCK_X_] = _reg._vd;
                }
            }
        }

        const de::CPd* _row_kernel_ptr = kernel + i * kernel_dims.z;

        for (uint32_t j = 0; j < kernel_dims.x; ++j)
        {
            _reg._vd = _row[mapping_shmem_idy][threadIdx.x + j];
            _accu._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(_reg._cplxd, _row_kernel_ptr[0], _accu._cplxd);
            ++_row_kernel_ptr;
        }
        __syncthreads();
    }

    uint64_t dex_dst = tidx + tidy * (uint64_t)pitchdst_v1;
    if (tidx < conv_area.x && tidy < conv_area.y) {
        dst[dex_dst] = _accu._vd;
    }
}



// block[32, 8]
// block_covered[32 * 4, 8]
template <uint32_t _ext_w> __global__ void
decx::dsp::GPUK::cu_filter2D_BC_cplxd(const double2* __restrict  src, 
                                      const de::CPd* __restrict   kernel,
                                      double2* __restrict        dst,
                                      const uint32_t            pitchsrc_v1, 
                                      const uint32_t            pitchdst_v1, 
                                      const uint3               kernel_dims,      // [W, H, pitch]
                                      const uint2               conv_area)
{
    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    const uint32_t _ldg_bound_x = conv_area.x + kernel_dims.x - 1;
    const uint32_t _KH_half = (kernel_dims.y >> 1);
    int64_t dex_src = tidx + tidy * (int64_t)pitchsrc_v1 - _KH_half * pitchsrc_v1;

    __shared__ double2 _row[_CU_FILTER2D_FP32_BLOCK_Y_][_CU_FILTER2D_FP32_BLOCK_X_ + _ext_w + 1];

    decx::utils::_cuda_vec128 _reg;
    _reg._vd = decx::utils::vec2_set1_fp64(0);

    // Load from global L2
    if (tidy < conv_area.y + _KH_half && tidy > _KH_half) {
        if (tidx < _ldg_bound_x) _reg._vd = src[dex_src];
    }
    _row[threadIdx.y][threadIdx.x] = _reg._vd;
    if (tidy < conv_area.y + _KH_half && tidy > _KH_half) {
        if (tidx + _CU_FILTER2D_FP32_BLOCK_X_ < _ldg_bound_x)  _reg._vd = src[dex_src + _CU_FILTER2D_FP32_BLOCK_X_];
    }
    if (threadIdx.x < _ext_w) {
        _row[threadIdx.y][threadIdx.x + _CU_FILTER2D_FP32_BLOCK_X_] = _reg._vd;
    }
    // End of load from global L2

    __syncthreads();

    decx::utils::_cuda_vec128 _accu;
    _accu._vd = decx::utils::vec2_set1_fp64(0);

    for (int32_t i = 0; i < kernel_dims.y; ++i)
    {
        // The updated fresh data is mapped to the last warp anyway, so don't need to sync threads after the refresh.
        const uint8_t mapping_shmem_idy = (uint8_t)((threadIdx.y + i) & 7);

        if (tidy + i > _KH_half && tidy + i < conv_area.y + _KH_half)
        {
            if (i > 0) {
                if (threadIdx.y == _CU_FILTER2D_FP32_BLOCK_Y_ - 1) {    // Let the last warp to load from the new row
                    if (tidx < _ldg_bound_x) {
                        _reg._vd = src[dex_src + i * pitchsrc_v1];
                    }
                    _row[mapping_shmem_idy][threadIdx.x] = _reg._vd;
                    _reg._vd = decx::utils::vec2_set1_fp64(0);
                    if (tidx + _CU_FILTER2D_FP32_BLOCK_X_ < _ldg_bound_x) {
                        _reg._vd = src[dex_src + i * pitchsrc_v1 + _CU_FILTER2D_FP32_BLOCK_X_];
                    }
                    if (threadIdx.x < _ext_w) {
                        _row[mapping_shmem_idy][threadIdx.x + _CU_FILTER2D_FP32_BLOCK_X_] = _reg._vd;
                    }
                }
            }

            const de::CPd* _row_kernel_ptr = kernel + i * kernel_dims.z;

            for (uint32_t j = 0; j < kernel_dims.x; ++j)
            {
                _reg._vd = _row[mapping_shmem_idy][threadIdx.x + j];
                _accu._cplxd = decx::dsp::fft::GPUK::_complex_fma_fp64(_reg._cplxd, _row_kernel_ptr[0], _accu._cplxd);
                ++_row_kernel_ptr;
            }
        }
        __syncthreads();
    }

    uint64_t dex_dst = tidx + tidy * (uint64_t)pitchdst_v1;
    if (tidx < conv_area.x && tidy < conv_area.y) {
        dst[dex_dst] = _accu._vd;
    }
}


_CU_FILTER2D_SPEC_(cu_filter2D_NB_cplxd, double2, de::CPd, double2, 2);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_cplxd, double2, de::CPd, double2, 4);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_cplxd, double2, de::CPd, double2, 6);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_cplxd, double2, de::CPd, double2, 8);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_cplxd, double2, de::CPd, double2, 10);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_cplxd, double2, de::CPd, double2, 12);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_cplxd, double2, de::CPd, double2, 14);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_cplxd, double2, de::CPd, double2, 16);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_cplxd, double2, de::CPd, double2, 18);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_cplxd, double2, de::CPd, double2, 20);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_cplxd, double2, de::CPd, double2, 22);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_cplxd, double2, de::CPd, double2, 24);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_cplxd, double2, de::CPd, double2, 26);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_cplxd, double2, de::CPd, double2, 28);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_cplxd, double2, de::CPd, double2, 30);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_cplxd, double2, de::CPd, double2, 32);

_CU_FILTER2D_SPEC_(cu_filter2D_BC_cplxd, double2, de::CPd, double2, 2);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_cplxd, double2, de::CPd, double2, 4);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_cplxd, double2, de::CPd, double2, 6);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_cplxd, double2, de::CPd, double2, 8);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_cplxd, double2, de::CPd, double2, 10);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_cplxd, double2, de::CPd, double2, 12);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_cplxd, double2, de::CPd, double2, 14);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_cplxd, double2, de::CPd, double2, 16);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_cplxd, double2, de::CPd, double2, 18);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_cplxd, double2, de::CPd, double2, 20);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_cplxd, double2, de::CPd, double2, 22);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_cplxd, double2, de::CPd, double2, 24);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_cplxd, double2, de::CPd, double2, 26);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_cplxd, double2, de::CPd, double2, 28);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_cplxd, double2, de::CPd, double2, 30);
_CU_FILTER2D_SPEC_(cu_filter2D_BC_cplxd, double2, de::CPd, double2, 32);
