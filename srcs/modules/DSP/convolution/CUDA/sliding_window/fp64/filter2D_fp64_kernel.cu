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
    namespace GPUK 
    {
        __device__ __inline__ static void 
        _conv_fp64_sw2_v2(decx::utils::_cuda_vec128& reg0, 
                          decx::utils::_cuda_vec128& reg1, 
                          decx::utils::_cuda_vec128& accu, 
                          const double* _kernel_row_ptr)
        {
            reg0._arrd[0] = reg1._arrd[0];
            accu._arrd[0] = __fma_rn(reg0._arrd[1], _kernel_row_ptr[0], accu._arrd[0]);
            accu._arrd[1] = __fma_rn(reg0._arrd[0], _kernel_row_ptr[0], accu._arrd[1]);

            reg0._arrd[1] = reg1._arrd[1];
            accu._arrd[0] = __fma_rn(reg0._arrd[0], _kernel_row_ptr[0], accu._arrd[0]);
            accu._arrd[1] = __fma_rn(reg0._arrd[1], _kernel_row_ptr[0], accu._arrd[1]);
        }

    }
}
}



// block[32, 8]
// block_covered[32 * 2, 8]
template <uint32_t _ext_w> __global__ void
decx::dsp::GPUK::cu_filter2D_NB_fp64(const double2* __restrict  src, 
                                     const double* __restrict   kernel,
                                     double2* __restrict        dst,
                                     const uint32_t             pitchsrc_v4, 
                                     const uint32_t             pitchdst_v4, 
                                     const uint3                kernel_dims,      // [W, H, pitch]
                                     const uint2                conv_area)
{
    const uint32_t tidx = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t tidy = threadIdx.y + blockIdx.y * blockDim.y;

    uint64_t dex_src = tidx + tidy * (uint64_t)pitchsrc_v4;

    const uint2 _ldg_bound_v2 = make_uint2(decx::utils::ceil<uint32_t>(conv_area.x + kernel_dims.x - 1, 2),
        conv_area.y + kernel_dims.y - 1);

    constexpr uint32_t _ext_w_v2 = _ext_w / 2;
    constexpr uint32_t _row_cover_x_v2 = _CU_FILTER2D_FP32_BLOCK_X_ * 2 / 2;
    __shared__ double2 _row[_CU_FILTER2D_FP32_BLOCK_Y_][_row_cover_x_v2 + _ext_w_v2 + 1];

    const uint32_t _k_loop_w_v2 = decx::utils::ceil<uint32_t>(kernel_dims.x - 1, 2);
    const uint32_t _k_loop_w_L = (kernel_dims.x - 1) % 2 ? (kernel_dims.x - 1) % 2 : 2;
    decx::utils::_cuda_vec128 _recv[2];
    _recv[0]._vd = decx::utils::vec2_set1_fp64(0);
    _recv[1]._vd = decx::utils::vec2_set1_fp64(0);

    // Load from global L2
    if (tidy < _ldg_bound_v2.y) {
        if (tidx < _ldg_bound_v2.x) _recv[0]._vd = src[dex_src];
    }
    _row[threadIdx.y][threadIdx.x] = _recv[0]._vd;
    if (tidy < _ldg_bound_v2.y) {
        if (tidx + _row_cover_x_v2 < _ldg_bound_v2.x)  _recv[1]._vd = src[dex_src + _row_cover_x_v2];
    }
    if (threadIdx.x < _ext_w_v2) {
        _row[threadIdx.y][threadIdx.x + _row_cover_x_v2] = _recv[1]._vd;
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
                if (tidx < _ldg_bound_v2.x && tidy + i < _ldg_bound_v2.y) {
                    _recv[0]._vd = src[dex_src + i * pitchsrc_v4];
                }
                _row[mapping_shmem_idy][threadIdx.x] = _recv[0]._vd;
                _recv[0]._vd = decx::utils::vec2_set1_fp64(0);
                if (tidx + _row_cover_x_v2 < _ldg_bound_v2.x && tidy + i < _ldg_bound_v2.y) {
                    _recv[0]._vd = src[dex_src + i * pitchsrc_v4 + _row_cover_x_v2];
                }
                if (threadIdx.x < _ext_w_v2) {
                    _row[mapping_shmem_idy][threadIdx.x + _row_cover_x_v2] = _recv[0]._vd;
                }
            }
        }

        const double* _row_kernel_ptr = kernel + i * kernel_dims.z;

        _recv[0]._vd = _row[mapping_shmem_idy][threadIdx.x];
        _accu._vd = decx::utils::cuda::__fma_v2_v1_v2(_recv[0]._vd, _row_kernel_ptr[0], _accu._vd);
        ++_row_kernel_ptr;

        for (uint32_t j = 0; j < _k_loop_w_v2; ++j)
        {
            _recv[1]._vd = _row[mapping_shmem_idy][threadIdx.x + j + 1];
            decx::dsp::GPUK::_conv_fp64_sw2_v2(_recv[0], _recv[1], _accu, _row_kernel_ptr);
            _row_kernel_ptr += 2;
        }
        __syncthreads();
    }

    uint64_t dex_dst = tidx + tidy * (uint64_t)pitchdst_v4;
    if (tidx < decx::utils::ceil<uint32_t>(conv_area.x, 2) && tidy < conv_area.y) {
        dst[dex_dst] = _accu._vd;
    }
}

_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 2);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 4);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 6);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 8);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 10);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 12);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 14);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 16);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 18);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 20);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 22);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 24);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 26);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 28);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 30);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 32);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 34);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 36);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 38);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 40);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 42);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 44);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 46);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 48);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 50);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 52);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 54);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 56);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 58);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 60);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 62);
_CU_FILTER2D_SPEC_(cu_filter2D_NB_fp64, double2, double, double2, 64);
