/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "im2col_fp32.cuh"
#include "../../../../core/utils/decx_cuda_vectypes_ops.cuh"


__global__ void 
decx::nn::GPUK::cu_im2col_NB_fp32(const float4* __restrict  src, 
                                  float4* __restrict        dst, 
                                  const uint2               conv2D_area, 
                                  const uint2               kernel_dims,
                                  const uint32_t            dpitch_src_v1, 
                                  const uint32_t            wpitch_src_v1, 
                                  const uint64_t            dst_row_size)
{
    uint64_t dex_src = 0, dex_dst = 0;

    const uint8_t _dp_v4_lane_size = dpitch_src_v1 / 4;
    const uint8_t _dst_blockDimx = blockDim.x / _dp_v4_lane_size;
    const uint8_t _lane_id = threadIdx.x / _dp_v4_lane_size;
    const uint8_t _lane_loc_id = threadIdx.x % _dp_v4_lane_size;

    const uint32_t dex_plane_x = _lane_id + blockIdx.x * _dst_blockDimx;
    const uint32_t dex_plane_y = threadIdx.y + blockIdx.y * _IM2COL_FP32_BLOCK_Y_;

    /*uint32_t dex_src_plane = _lane_loc_id + dex_plane_x * dpitch_src_v1;
    uint32_t dex_src_y = dex_plane_y;*/

    decx::utils::_cuda_vec128 _reg;
    _reg._vf = decx::utils::vec4_set1_fp32(0);

    // 256 (default)
    constexpr uint32_t block_size_1D = _IM2COL_FP32_BLOCK_X_ * _IM2COL_FP32_BLOCK_Y_;
    // 128 (usually)
    constexpr uint32_t warp_proc_v4_size = _WARP_SIZE_ * 4;
    // 64 (usually)
    constexpr uint32_t thread_per_row_shmem = _IM2COL_GET_THREAD_PER_ROW_(4);
    // 4 (usually)
    constexpr uint8_t STG_blockDim_y = _IM2COL_GET_STG_BLOCKDIM_Y_(thread_per_row_shmem);

    const uint8_t block_1D_loc_tid = threadIdx.x + blockDim.x * threadIdx.y;

    const uint32_t STG_dex_x = (block_1D_loc_tid % thread_per_row_shmem) + blockIdx.x * thread_per_row_shmem;
    uint32_t STG_dex_y = (block_1D_loc_tid / thread_per_row_shmem)/* + blockIdx.y * STG_blockDim_y*/;

    __shared__ float _shmem[4][block_size_1D + 4];

    for (uint32_t i = 0; i < kernel_dims.x; ++i) 
    {
        dex_src = _lane_loc_id + dex_plane_x * dpitch_src_v1 / 4 + dex_plane_y * wpitch_src_v1 * dpitch_src_v1;
        for (uint32_t j = 0; j < kernel_dims.y; ++j) 
        {
            if (dex_plane_x < conv2D_area.x && dex_plane_y < conv2D_area.y) {
                _reg._vf = src[dex_src];
            }
            
            _shmem[0][block_1D_loc_tid] = _reg._arrf[0];
            _shmem[1][block_1D_loc_tid] = _reg._arrf[1];
            _shmem[2][block_1D_loc_tid] = _reg._arrf[2];
            _shmem[3][block_1D_loc_tid] = _reg._arrf[3];

            __syncthreads();

            _reg._vf = ((float4*)_shmem[block_1D_loc_tid / thread_per_row_shmem])[(block_1D_loc_tid % thread_per_row_shmem)];

            if (dex_plane_x < conv2D_area.x && dex_plane_y < conv2D_area.y) {
                dst[STG_dex_x + STG_dex_y * dst_row_size / 4] = _reg._vf;
            }
            STG_dex_y += STG_blockDim_y;
            dex_src += dpitch_src_v1;
        }
        dex_src += wpitch_src_v1 * dpitch_src_v1;
    }


}




__global__ void 
decx::nn::GPUK::cu_im2col_NB_fp32_divKH(const float4* __restrict  src, 
                                  float4* __restrict        dst, 
                                  const uint2               conv2D_area, 
                                  const uint2               kernel_dims,
                                  const uint32_t            dpitch_src_v1, 
                                  const uint32_t            wpitch_src_v1, 
                                  const uint64_t            dst_row_size)
{
    uint64_t dex_src = 0, dex_dst = 0;

    const uint8_t _dp_v4_lane_size = dpitch_src_v1 / 4;
    const uint8_t _dst_blockDimx = blockDim.x / _dp_v4_lane_size;
    const uint8_t _lane_id = threadIdx.x / _dp_v4_lane_size;
    const uint8_t _lane_loc_id = threadIdx.x % _dp_v4_lane_size;

    const uint32_t dex_plane_x = _lane_id + blockIdx.x * _dst_blockDimx;
    const uint32_t dex_plane_y = blockIdx.z + threadIdx.y + blockIdx.y * _IM2COL_FP32_BLOCK_Y_;

    /*uint32_t dex_src_plane = _lane_loc_id + dex_plane_x * dpitch_src_v1;
    uint32_t dex_src_y = dex_plane_y;*/

    decx::utils::_cuda_vec128 _reg;
    _reg._vf = decx::utils::vec4_set1_fp32(0);

    // 256 (default)
    constexpr uint32_t block_size_1D = _IM2COL_FP32_BLOCK_X_ * _IM2COL_FP32_BLOCK_Y_;
    // 128 (usually)
    constexpr uint32_t warp_proc_v4_size = _WARP_SIZE_ * 4;
    // 64 (usually)
    constexpr uint32_t thread_per_row_shmem = _IM2COL_GET_THREAD_PER_ROW_(4);
    // 4 (usually)
    constexpr uint8_t STG_blockDim_y = _IM2COL_GET_STG_BLOCKDIM_Y_(thread_per_row_shmem);

    const uint8_t block_1D_loc_tid = threadIdx.x + blockDim.x * threadIdx.y;

    const uint32_t STG_dex_x = (block_1D_loc_tid % thread_per_row_shmem) + blockIdx.x * thread_per_row_shmem;
    uint32_t STG_dex_y = (block_1D_loc_tid / thread_per_row_shmem)/* + blockIdx.y * STG_blockDim_y*/;

    __shared__ float _shmem[4][block_size_1D + 4];

    for (uint32_t i = 0; i < kernel_dims.x; ++i) 
    {
        dex_src = _lane_loc_id + dex_plane_x * dpitch_src_v1 / 4 + dex_plane_y * wpitch_src_v1 * dpitch_src_v1;
        for (uint32_t j = 0; j < kernel_dims.y; ++j) 
        {
            if (dex_plane_x < conv2D_area.x && dex_plane_y < conv2D_area.y) {
                _reg._vf = src[dex_src];
            }
            
            _shmem[0][block_1D_loc_tid] = _reg._arrf[0];
            _shmem[1][block_1D_loc_tid] = _reg._arrf[1];
            _shmem[2][block_1D_loc_tid] = _reg._arrf[2];
            _shmem[3][block_1D_loc_tid] = _reg._arrf[3];

            __syncthreads();

            _reg._vf = ((float4*)_shmem[block_1D_loc_tid / thread_per_row_shmem])[(block_1D_loc_tid % thread_per_row_shmem)];

            if (dex_plane_x < conv2D_area.x && dex_plane_y < conv2D_area.y) {
                dst[STG_dex_x + STG_dex_y * dst_row_size / 4] = _reg._vf;
            }
            STG_dex_y += STG_blockDim_y;
            dex_src += dpitch_src_v1;
        }
        dex_src += wpitch_src_v1 * dpitch_src_v1;
    }


}