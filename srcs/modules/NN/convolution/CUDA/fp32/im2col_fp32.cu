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
decx::nn::GPUK::cu_im2col_D4_NB_fp32_divKH(const float4* __restrict  src, 
                                  float4* __restrict        dst, 
                                  const uint2               conv2D_area, 
                                  const uint2               kernel_dims,
                                  const uint32_t            wpitch_dst_v1, 
                                  const uint32_t            wpitch_src_v1, 
                                  const uint64_t            im2col_buf_pitch_v1)
{
    uint64_t dex_src = 0;

    const uint32_t dex_plane_src_x = threadIdx.x + blockIdx.x * blockDim.x;
    const uint32_t dex_plane_src_y = threadIdx.y + blockIdx.y * blockDim.y;

    const uint8_t STG_threadIdx_x = (threadIdx.x + blockDim.x * threadIdx.y) % _CUDA_WARP_SIZE_;
    //const uint8_t STG_threadIdx_y = ((threadIdx.x + blockDim.x * threadIdx.y) / _CUDA_WARP_SIZE_) % 4;
    const uint8_t STG_threadIdx_y = threadIdx.x / _CUDA_WARP_SIZE_;

    const uint32_t dex_plane_dst_x = STG_threadIdx_x + blockIdx.x * _CUDA_WARP_SIZE_;
    const uint32_t dex_plane_dst_y = STG_threadIdx_y;

    decx::utils::_cuda_vec128 _reg;

    const uint32_t STG_dex_x = dex_plane_dst_x + dex_plane_src_y * wpitch_dst_v1 / 4;
    //const uint32_t STG_dex_x = dex_plane_dst_x * 4 + dex_plane_src_y * wpitch_dst_v1/* / 4*/;
    uint32_t STG_dex_y = dex_plane_dst_y;

    __shared__ float _shmem[_IM2COL_D4_FP32_BLOCK_Y_ * 4][_IM2COL_D4_FP32_BLOCK_X_ + 4];

    for (uint32_t i = 0; i < kernel_dims.y; ++i) 
    {
        dex_src = dex_plane_src_x + (blockIdx.z + dex_plane_src_y + i) * wpitch_src_v1;
        for (uint32_t j = 0; j < kernel_dims.x; ++j) 
        {
            _reg._vf = decx::utils::vec4_set1_fp32(0);

            if (dex_plane_src_x < conv2D_area.x && dex_plane_src_y < conv2D_area.y) {
                _reg._vf = src[dex_src];
            }

            _shmem[threadIdx.y * 4 + 0][threadIdx.x] = _reg._arrf[0];
            _shmem[threadIdx.y * 4 + 1][threadIdx.x] = _reg._arrf[1];
            _shmem[threadIdx.y * 4 + 2][threadIdx.x] = _reg._arrf[2];
            _shmem[threadIdx.y * 4 + 3][threadIdx.x] = _reg._arrf[3];

            __syncthreads();

            _reg._vf = ((float4*)_shmem[threadIdx.y * 4 + STG_threadIdx_y])[STG_threadIdx_x];

            if (dex_plane_src_x < conv2D_area.x && dex_plane_src_y < conv2D_area.y) 
            {
                dst[STG_dex_x + (STG_dex_y + blockIdx.z * kernel_dims.x * 4) * im2col_buf_pitch_v1 / 4] = _reg._vf;
                /*_dst[STG_dex_x + (STG_dex_y + blockIdx.z * kernel_dims.x * 4) * im2col_buf_pitch_v1] = _reg._vf.x;
                _dst[STG_dex_x + (STG_dex_y + blockIdx.z * kernel_dims.x * 4) * im2col_buf_pitch_v1 + 1] = _reg._vf.y;
                _dst[STG_dex_x + (STG_dex_y + blockIdx.z * kernel_dims.x * 4) * im2col_buf_pitch_v1 + 2] = _reg._vf.z;
                _dst[STG_dex_x + (STG_dex_y + blockIdx.z * kernel_dims.x * 4) * im2col_buf_pitch_v1 + 3] = _reg._vf.w;*/
            }
            STG_dex_y += 4;
            dex_src += 1;

            __syncthreads();
        }
    }
}
