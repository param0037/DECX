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
decx::nn::GPUK::cu_im2col_DP4_NB_fp32(const float4* __restrict  src, 
                                      float4* __restrict        dst, 
                                      const uint2               conv2D_area, 
                                      const uint2               kernel_dims,
                                      const uint2               strides,
                                      const uint32_t            wpitch_dst_v1, 
                                      const uint32_t            wpitch_src_v1, 
                                      const uint64_t            im2col_buf_pitch_v1)
{
    constexpr uint32_t _LDG_blockDim_x = _IM2COL_GET_STG_BLOCKDIM_X_(_IM2COL_D4N_FP32_BLOCK_X_, 4);
    uint64_t dex_src = 0;

    const uint32_t dex_plane_src_x = (threadIdx.x + blockIdx.x * _LDG_blockDim_x);
    const uint32_t dex_plane_src_y = (threadIdx.y + blockIdx.y * blockDim.y);

    const uint8_t STG_threadIdx_x = (threadIdx.x + blockDim.x * threadIdx.y) % _CUDA_WARP_SIZE_;
    const uint8_t STG_threadIdx_y = threadIdx.x / _CUDA_WARP_SIZE_;

    const uint32_t dex_plane_dst_x = STG_threadIdx_x + blockIdx.x * _CUDA_WARP_SIZE_;
    uint32_t dex_plane_dst_y = STG_threadIdx_y;

    decx::utils::_cuda_vec128 _reg;

    const uint32_t STG_dex_x = dex_plane_dst_x + dex_plane_src_y * wpitch_dst_v1 / 4;
    
    __shared__ float _shmem[_IM2COL_D4N_FP32_BLOCK_Y_ * 4][_IM2COL_D4N_FP32_BLOCK_X_ + 4];

    for (uint32_t i = 0; i < decx::utils::ceil<uint32_t>(kernel_dims.y, gridDim.z); ++i) 
    {
        dex_src = dex_plane_src_x * strides.x + (blockIdx.z + dex_plane_src_y * strides.y + i) * wpitch_src_v1;
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

            if (dex_plane_dst_x < conv2D_area.x && dex_plane_src_y < conv2D_area.y) {
                dst[STG_dex_x + (dex_plane_dst_y + blockIdx.z * kernel_dims.x * 4) * im2col_buf_pitch_v1 / 4] = _reg._vf;
            }
            dex_plane_dst_y += 4;
            dex_src += 1;

            __syncthreads();
        }
    }
}


template <bool _boundless_T, bool _boundless_B> __global__ void
decx::nn::GPUK::cu_im2col_DP4_BC_fp32(const float4* __restrict  src, 
                                      float4* __restrict        dst, 
                                      const uint2               conv2D_area, 
                                      const uint2               kernel_dims,
                                      const uint2               strides,
                                      const uint32_t            wpitch_dst_v1, 
                                      const uint32_t            wpitch_src_v1, 
                                      const uint64_t            im2col_buf_pitch_v1)
{
    constexpr uint32_t _LDG_blockDim_x = _IM2COL_GET_STG_BLOCKDIM_X_(_IM2COL_D4N_FP32_BLOCK_X_, 4);
    int64_t dex_src = 0;

    const uint32_t dex_plane_src_x = (threadIdx.x + blockIdx.x * _LDG_blockDim_x);
    const uint32_t dex_plane_src_y = (threadIdx.y + blockIdx.y * blockDim.y);

    const uint8_t STG_threadIdx_x = (threadIdx.x + blockDim.x * threadIdx.y) % _CUDA_WARP_SIZE_;
    const uint8_t STG_threadIdx_y = threadIdx.x / _CUDA_WARP_SIZE_;

    const uint32_t dex_plane_dst_x = STG_threadIdx_x + blockIdx.x * _CUDA_WARP_SIZE_;
    uint32_t dex_plane_dst_y = STG_threadIdx_y;

    decx::utils::_cuda_vec128 _reg;

    const uint32_t& _half_KH = (kernel_dims.y / 2);

    const uint32_t STG_dex_x = dex_plane_dst_x + dex_plane_src_y * wpitch_dst_v1 / 4;
    
    __shared__ float _shmem[_IM2COL_D4N_FP32_BLOCK_Y_ * 4][_IM2COL_D4N_FP32_BLOCK_X_ + 4];

    for (uint32_t i = 0; i < decx::utils::ceil<uint32_t>(kernel_dims.y, gridDim.z); ++i) 
    {
        const uint32_t _global_coor_y = blockIdx.z + dex_plane_src_y * strides.y + i;
        
        if ((_boundless_T || _global_coor_y > _half_KH-1) && (_boundless_B || _global_coor_y < conv2D_area.y * strides.y + _half_KH))
        {
            dex_src = dex_plane_src_x * strides.x + _global_coor_y * wpitch_src_v1;
            
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

                if (dex_plane_dst_x < conv2D_area.x && dex_plane_src_y < conv2D_area.y) {
                    dst[STG_dex_x + (dex_plane_dst_y + blockIdx.z * kernel_dims.x * 4) * im2col_buf_pitch_v1 / 4] = _reg._vf;
                }
                dex_plane_dst_y += 4;
                dex_src += 1;

                __syncthreads();
            }   // end for
        }   // end if
    }
}


template __global__ void decx::nn::GPUK::cu_im2col_DP4_BC_fp32<false, false>(const float4* __restrict, float4* __restrict, const uint2, const uint2,
    const uint2, const uint32_t, const uint32_t, const uint64_t);

template __global__ void decx::nn::GPUK::cu_im2col_DP4_BC_fp32<true, false>(const float4* __restrict, float4* __restrict, const uint2, const uint2,
    const uint2, const uint32_t, const uint32_t, const uint64_t);

template __global__ void decx::nn::GPUK::cu_im2col_DP4_BC_fp32<false, true>(const float4* __restrict, float4* __restrict, const uint2, const uint2,
    const uint2, const uint32_t, const uint32_t, const uint64_t);



// block[32 * 4, 2] = [128, 2]
__global__ void 
decx::nn::GPUK::cu_im2col_DP8_NB_fp32(const float4* __restrict  src, 
                                      float2* __restrict        dst, 
                                      const uint2               conv2D_area, 
                                      const uint2               kernel_dims,
                                      const uint2               strides,
                                      const uint32_t            wpitch_dst_v1, 
                                      const uint32_t            wpitch_src_v1, 
                                      const uint64_t            im2col_buf_pitch_v1)
{
    constexpr uint32_t _LDG_blockDim_x = _IM2COL_GET_STG_BLOCKDIM_X_(_IM2COL_D4N_FP32_BLOCK_X_, 8);

    uint64_t dex_src = 0;

    const uchar2 _logical_ldgl = make_uchar2(threadIdx.x % 2, threadIdx.x / 2);
    const uchar2 _logical_gl2shmem = make_uchar2((threadIdx.x / 2) % _LDG_blockDim_x, threadIdx.x % 2);
    const uchar2 _logical_stgl = make_uchar2(threadIdx.x % _CUDA_WARP_SIZE_, threadIdx.x / _CUDA_WARP_SIZE_);

    const uint32_t dex_plane_src_x = (_logical_ldgl.y + blockIdx.x * _LDG_blockDim_x);
    const uint32_t dex_plane_src_y = (threadIdx.y + blockIdx.y * blockDim.y);

    uint32_t dex_plane_dst_y = _logical_stgl.y;

    decx::utils::_cuda_vec128 _reg;

    const uint32_t dex_plane_dst_x = _logical_stgl.x + _CUDA_WARP_SIZE_ * blockIdx.x;
    const uint32_t STG_dex_x = dex_plane_dst_x + dex_plane_src_y * wpitch_dst_v1 / 2;
    
    __shared__ float _shmem[_IM2COL_D4N_FP32_BLOCK_Y_ * 4][_IM2COL_D4N_FP32_BLOCK_X_ + 2];

    for (uint32_t i = 0; i < decx::utils::ceil<uint32_t>(kernel_dims.y, gridDim.z); ++i) 
    {
        dex_src = _logical_ldgl.x + (dex_plane_src_x * strides.x + (blockIdx.z + dex_plane_src_y * strides.y + i) * wpitch_src_v1) * 2;
        for (uint32_t j = 0; j < kernel_dims.x; ++j) 
        {
            _reg._vf = decx::utils::vec4_set1_fp32(0);

            if (dex_plane_src_x < conv2D_area.x && dex_plane_src_y < conv2D_area.y) {
                _reg._vf = src[dex_src];
            }

            _shmem[threadIdx.y * 4 + 0][_logical_gl2shmem.x + 64 * _logical_gl2shmem.y] = _reg._arrf[0];
            _shmem[threadIdx.y * 4 + 1][_logical_gl2shmem.x + 64 * _logical_gl2shmem.y] = _reg._arrf[1];
            _shmem[threadIdx.y * 4 + 2][_logical_gl2shmem.x + 64 * _logical_gl2shmem.y] = _reg._arrf[2];
            _shmem[threadIdx.y * 4 + 3][_logical_gl2shmem.x + 64 * _logical_gl2shmem.y] = _reg._arrf[3];

            __syncthreads();

            if (dex_plane_dst_x < conv2D_area.x && dex_plane_src_y < conv2D_area.y)
            {
                _reg._arrf2[0] = ((float2*)_shmem[threadIdx.y * 4 + _logical_stgl.y])[_logical_stgl.x];
                dst[STG_dex_x + (dex_plane_dst_y + blockIdx.z * kernel_dims.x * 8) * im2col_buf_pitch_v1 / 2] = _reg._arrf2[0];
                
                dex_plane_dst_y += 4;

                _reg._arrf2[1] = ((float2*)_shmem[threadIdx.y * 4 + _logical_stgl.y])[_logical_stgl.x + _CUDA_WARP_SIZE_];
                dst[STG_dex_x + (dex_plane_dst_y + blockIdx.z * kernel_dims.x * 8) * im2col_buf_pitch_v1 / 2] = _reg._arrf2[1];
                dex_plane_dst_y += 4;
            }

            dex_src += 2;

            __syncthreads();
        }
    }
}


// block[32 * 4, 2] = [128, 2]
template <bool _boundless_T, bool _boundless_B> __global__ void
decx::nn::GPUK::cu_im2col_DP8_BC_fp32(const float4* __restrict  src, 
                                      float2* __restrict        dst, 
                                      const uint2               conv2D_area, 
                                      const uint2               kernel_dims,
                                      const uint2               strides,
                                      const uint32_t            wpitch_dst_v1, 
                                      const uint32_t            wpitch_src_v1, 
                                      const uint64_t            im2col_buf_pitch_v1)
{
    constexpr uint32_t _LDG_blockDim_x = _IM2COL_GET_STG_BLOCKDIM_X_(_IM2COL_D4N_FP32_BLOCK_X_, 8);

    uint64_t dex_src = 0;

    const uchar2 _logical_ldgl = make_uchar2(threadIdx.x % 2, threadIdx.x / 2);
    const uchar2 _logical_gl2shmem = make_uchar2((threadIdx.x / 2) % _LDG_blockDim_x, threadIdx.x % 2);
    const uchar2 _logical_stgl = make_uchar2(threadIdx.x % _CUDA_WARP_SIZE_, threadIdx.x / _CUDA_WARP_SIZE_);

    const uint32_t dex_plane_src_x = (_logical_ldgl.y + blockIdx.x * _LDG_blockDim_x);
    const uint32_t dex_plane_src_y = (threadIdx.y + blockIdx.y * blockDim.y);

    uint32_t dex_plane_dst_y = _logical_stgl.y;

    decx::utils::_cuda_vec128 _reg;

    const uint32_t& _half_KH = (kernel_dims.y / 2);

    const uint32_t dex_plane_dst_x = _logical_stgl.x + _CUDA_WARP_SIZE_ * blockIdx.x;
    const uint32_t STG_dex_x = dex_plane_dst_x + dex_plane_src_y * wpitch_dst_v1 / 2;
    
    __shared__ float _shmem[_IM2COL_D4N_FP32_BLOCK_Y_ * 4][_IM2COL_D4N_FP32_BLOCK_X_ + 2];

    for (uint32_t i = 0; i < decx::utils::ceil<uint32_t>(kernel_dims.y, gridDim.z); ++i) 
    {
        const uint32_t _global_coor_y = blockIdx.z + dex_plane_src_y * strides.y + i;
        
        if ((_boundless_T || _global_coor_y > _half_KH - 1) && (_boundless_B || _global_coor_y < conv2D_area.y * strides.y + _half_KH))
        {
            dex_src = _logical_ldgl.x + (dex_plane_src_x * strides.x + _global_coor_y * wpitch_src_v1) * 2;

            for (uint32_t j = 0; j < kernel_dims.x; ++j)
            {
                _reg._vf = decx::utils::vec4_set1_fp32(0);
                if (dex_plane_src_x < conv2D_area.x && dex_plane_src_y < conv2D_area.y) {
                    _reg._vf = src[dex_src];
                }

                _shmem[threadIdx.y * 4 + 0][_logical_gl2shmem.x + 64 * _logical_gl2shmem.y] = _reg._arrf[0];
                _shmem[threadIdx.y * 4 + 1][_logical_gl2shmem.x + 64 * _logical_gl2shmem.y] = _reg._arrf[1];
                _shmem[threadIdx.y * 4 + 2][_logical_gl2shmem.x + 64 * _logical_gl2shmem.y] = _reg._arrf[2];
                _shmem[threadIdx.y * 4 + 3][_logical_gl2shmem.x + 64 * _logical_gl2shmem.y] = _reg._arrf[3];

                __syncthreads();

                if (dex_plane_dst_x < conv2D_area.x && dex_plane_src_y < conv2D_area.y)
                {
                    _reg._arrf2[0] = ((float2*)_shmem[threadIdx.y * 4 + _logical_stgl.y])[_logical_stgl.x];
                    dst[STG_dex_x + (dex_plane_dst_y + blockIdx.z * kernel_dims.x * 8) * im2col_buf_pitch_v1 / 2] = _reg._arrf2[0];

                    dex_plane_dst_y += 4;

                    _reg._arrf2[1] = ((float2*)_shmem[threadIdx.y * 4 + _logical_stgl.y])[_logical_stgl.x + _CUDA_WARP_SIZE_];
                    dst[STG_dex_x + (dex_plane_dst_y + blockIdx.z * kernel_dims.x * 8) * im2col_buf_pitch_v1 / 2] = _reg._arrf2[1];
                }

                dex_plane_dst_y += 4;
                dex_src += 2;

                __syncthreads();
            }   // end for
        }   // end if
    }
}

template __global__ void decx::nn::GPUK::cu_im2col_DP8_BC_fp32<false, false>(const float4* __restrict, float2* __restrict, const uint2, const uint2,
    const uint2, const uint32_t, const uint32_t, const uint64_t);

template __global__ void decx::nn::GPUK::cu_im2col_DP8_BC_fp32<true, false>(const float4* __restrict, float2* __restrict, const uint2, const uint2,
    const uint2, const uint32_t, const uint32_t, const uint64_t);

template __global__ void decx::nn::GPUK::cu_im2col_DP8_BC_fp32<false, true>(const float4* __restrict, float2* __restrict, const uint2, const uint2,
    const uint2, const uint32_t, const uint32_t, const uint64_t);




// block[32 * 6, 2] = [128, 2]
__global__ void 
decx::nn::GPUK::cu_im2col_DP12_NB_fp32(const float4* __restrict  src, 
                                      float2* __restrict        dst, 
                                      const uint2               conv2D_area, 
                                      const uint2               kernel_dims,
                                      const uint2               strides,
                                      const uint32_t            wpitch_dst_v1, 
                                      const uint32_t            wpitch_src_v1, 
                                      const uint64_t            im2col_buf_pitch_v1)
{
    constexpr uint32_t _LDG_blockDim_x = _IM2COL_GET_STG_BLOCKDIM_X_(_IM2COL_D12_FP32_BLOCK_X_, 12);

    uint64_t dex_src = 0;

    const uchar2 _logical_ldgl = make_uchar2(threadIdx.x % 3, threadIdx.x / 3);
    const uchar2 _logical_gl2shmem = make_uchar2((threadIdx.x / 3) % _LDG_blockDim_x, threadIdx.x % 3);
    const uchar3 _logical_stgl = make_uchar3(threadIdx.x % _CUDA_WARP_SIZE_, (threadIdx.x / _CUDA_WARP_SIZE_) % 3, (threadIdx.x / _CUDA_WARP_SIZE_) / 3);

    const uint32_t dex_plane_src_x = (_logical_ldgl.y + blockIdx.x * _LDG_blockDim_x);
    const uint32_t dex_plane_src_y = (threadIdx.y + blockIdx.y * blockDim.y);

    uint32_t dex_plane_dst_y = _logical_stgl.y + _logical_stgl.z * 3;

    decx::utils::_cuda_vec128 _reg;

    const uint32_t dex_plane_dst_x = _logical_stgl.x + _CUDA_WARP_SIZE_ * blockIdx.x;
    const uint32_t STG_dex_x = dex_plane_dst_x + dex_plane_src_y * wpitch_dst_v1 / 2;       // /2 because of float2
    
    __shared__ float _shmem[_IM2COL_D12_FP32_BLOCK_Y_ * 4][_IM2COL_D12_FP32_BLOCK_X_ + 2];

    for (uint32_t i = 0; i < decx::utils::ceil<uint32_t>(kernel_dims.y, gridDim.z); ++i) 
    {
        dex_src = _logical_ldgl.x + (dex_plane_src_x * strides.x + (blockIdx.z + dex_plane_src_y * strides.y + i) * wpitch_src_v1) * 3;
        for (uint32_t j = 0; j < kernel_dims.x; ++j) 
        {
            _reg._vf = decx::utils::vec4_set1_fp32(0);

            if (dex_plane_src_x < conv2D_area.x && dex_plane_src_y < conv2D_area.y) {
                _reg._vf = src[dex_src];
            }

            _shmem[threadIdx.y * 4 + 0][_logical_gl2shmem.x + 64 * _logical_gl2shmem.y] = _reg._arrf[0];
            _shmem[threadIdx.y * 4 + 1][_logical_gl2shmem.x + 64 * _logical_gl2shmem.y] = _reg._arrf[1];
            _shmem[threadIdx.y * 4 + 2][_logical_gl2shmem.x + 64 * _logical_gl2shmem.y] = _reg._arrf[2];
            _shmem[threadIdx.y * 4 + 3][_logical_gl2shmem.x + 64 * _logical_gl2shmem.y] = _reg._arrf[3];

            __syncthreads();

            if (dex_plane_dst_x < conv2D_area.x && dex_plane_src_y < conv2D_area.y)
            {
                _reg._arrf2[0] = ((float2*)_shmem[threadIdx.y * 4 + _logical_stgl.z])[_logical_stgl.x + _CUDA_WARP_SIZE_ * _logical_stgl.y];
                dst[STG_dex_x + (dex_plane_dst_y + blockIdx.z * kernel_dims.x * 12) * im2col_buf_pitch_v1 / 2] = _reg._arrf2[0];
                
                dex_plane_dst_y += 6;

                _reg._arrf2[1] = ((float2*)_shmem[threadIdx.y * 4 + 2 + _logical_stgl.z])[_logical_stgl.x + _CUDA_WARP_SIZE_ * _logical_stgl.y];
                dst[STG_dex_x + (dex_plane_dst_y + blockIdx.z * kernel_dims.x * 12) * im2col_buf_pitch_v1 / 2] = _reg._arrf2[1];
                dex_plane_dst_y += 6;
            }

            dex_src += 3;

            __syncthreads();
        }
    }
}



// block[32 * 4, 2] = [128, 2]
template <bool _boundless_T, bool _boundless_B> __global__ void
decx::nn::GPUK::cu_im2col_DP12_BC_fp32(const float4* __restrict  src, 
                                       float2* __restrict        dst, 
                                       const uint2               conv2D_area, 
                                       const uint2               kernel_dims,
                                       const uint2               strides,
                                       const uint32_t            wpitch_dst_v1, 
                                       const uint32_t            wpitch_src_v1, 
                                       const uint64_t            im2col_buf_pitch_v1)
{
    constexpr uint32_t _LDG_blockDim_x = _IM2COL_GET_STG_BLOCKDIM_X_(_IM2COL_D12_FP32_BLOCK_X_, 12);

    uint64_t dex_src = 0;

    const uchar2 _logical_ldgl = make_uchar2(threadIdx.x % 3, threadIdx.x / 3);
    const uchar2 _logical_gl2shmem = make_uchar2((threadIdx.x / 3) % _LDG_blockDim_x, threadIdx.x % 3);
    const uchar3 _logical_stgl = make_uchar3(threadIdx.x % _CUDA_WARP_SIZE_, (threadIdx.x / _CUDA_WARP_SIZE_) % 3, (threadIdx.x / _CUDA_WARP_SIZE_) / 3);

    const uint32_t dex_plane_src_x = (_logical_ldgl.y + blockIdx.x * _LDG_blockDim_x);
    const uint32_t dex_plane_src_y = (threadIdx.y + blockIdx.y * blockDim.y);

    uint32_t dex_plane_dst_y = _logical_stgl.y + _logical_stgl.z * 3;

    decx::utils::_cuda_vec128 _reg;

    const uint32_t& _half_KH = (kernel_dims.y / 2);

    const uint32_t dex_plane_dst_x = _logical_stgl.x + _CUDA_WARP_SIZE_ * blockIdx.x;
    const uint32_t STG_dex_x = dex_plane_dst_x + dex_plane_src_y * wpitch_dst_v1 / 2;
    
    __shared__ float _shmem[_IM2COL_D12_FP32_BLOCK_Y_ * 4][_IM2COL_D12_FP32_BLOCK_X_ + 2];

    for (uint32_t i = 0; i < decx::utils::ceil<uint32_t>(kernel_dims.y, gridDim.z); ++i) 
    {
        const uint32_t _global_coor_y = blockIdx.z + dex_plane_src_y * strides.y + i;
        
        if ((_boundless_T || _global_coor_y > _half_KH - 1) && (_boundless_B || _global_coor_y < conv2D_area.y * strides.y + _half_KH))
        {
            dex_src = _logical_ldgl.x + (dex_plane_src_x * strides.x + _global_coor_y * wpitch_src_v1) * 3;

            for (uint32_t j = 0; j < kernel_dims.x; ++j)
            {
                _reg._vf = decx::utils::vec4_set1_fp32(0);
                if (dex_plane_src_x < conv2D_area.x && dex_plane_src_y < conv2D_area.y) {
                    _reg._vf = src[dex_src];
                }

                _shmem[threadIdx.y * 4 + 0][_logical_gl2shmem.x + 64 * _logical_gl2shmem.y] = _reg._arrf[0];
                _shmem[threadIdx.y * 4 + 1][_logical_gl2shmem.x + 64 * _logical_gl2shmem.y] = _reg._arrf[1];
                _shmem[threadIdx.y * 4 + 2][_logical_gl2shmem.x + 64 * _logical_gl2shmem.y] = _reg._arrf[2];
                _shmem[threadIdx.y * 4 + 3][_logical_gl2shmem.x + 64 * _logical_gl2shmem.y] = _reg._arrf[3];

                __syncthreads();

                if (dex_plane_dst_x < conv2D_area.x && dex_plane_src_y < conv2D_area.y)
                {
                    _reg._arrf2[0] = ((float2*)_shmem[threadIdx.y * 4 + _logical_stgl.z])[_logical_stgl.x + _CUDA_WARP_SIZE_ * _logical_stgl.y];
                    dst[STG_dex_x + (dex_plane_dst_y + blockIdx.z * kernel_dims.x * 12) * im2col_buf_pitch_v1 / 2] = _reg._arrf2[0];

                    dex_plane_dst_y += 6;

                    _reg._arrf2[1] = ((float2*)_shmem[threadIdx.y * 4 + 2 + _logical_stgl.z])[_logical_stgl.x + _CUDA_WARP_SIZE_ * _logical_stgl.y];
                    dst[STG_dex_x + (dex_plane_dst_y + blockIdx.z * kernel_dims.x * 12) * im2col_buf_pitch_v1 / 2] = _reg._arrf2[1];
                }

                dex_plane_dst_y += 6;
                dex_src += 3;

                __syncthreads();
            }   // end for
        }   // end if
    }
}

template __global__ void decx::nn::GPUK::cu_im2col_DP12_BC_fp32<false, false>(const float4* __restrict, float2* __restrict, const uint2, const uint2,
    const uint2, const uint32_t, const uint32_t, const uint64_t);

template __global__ void decx::nn::GPUK::cu_im2col_DP12_BC_fp32<true, false>(const float4* __restrict, float2* __restrict, const uint2, const uint2,
    const uint2, const uint32_t, const uint32_t, const uint64_t);

template __global__ void decx::nn::GPUK::cu_im2col_DP12_BC_fp32<false, true>(const float4* __restrict, float2* __restrict, const uint2, const uint2,
    const uint2, const uint32_t, const uint32_t, const uint64_t);



// block[32 * 4, 2] = [128, 2]
__global__ void 
decx::nn::GPUK::cu_im2col_DP16_NB_fp32(const float4* __restrict  src, 
                                       float* __restrict         dst, 
                                       const uint2               conv2D_area, 
                                       const uint2               kernel_dims,
                                       const uint2               strides,
                                       const uint32_t            wpitch_dst_v1, 
                                       const uint32_t            wpitch_src_v1, 
                                       const uint64_t            im2col_buf_pitch_v1)
{
    constexpr uint32_t _LDG_blockDim_x = _IM2COL_GET_STG_BLOCKDIM_X_(_IM2COL_D4N_FP32_BLOCK_X_, 16);

    uint64_t dex_src = 0;

    const uchar2 _logical_ldgl = make_uchar2(threadIdx.x % 4, threadIdx.x / 4);
    const uchar2 _logical_gl2shmem = make_uchar2((threadIdx.x / 4) % _LDG_blockDim_x, threadIdx.x % 4);
    const uchar2 _logical_stgl = make_uchar2(threadIdx.x % _CUDA_WARP_SIZE_, threadIdx.x / _CUDA_WARP_SIZE_);

    const uint32_t dex_plane_src_x = (_logical_ldgl.y + blockIdx.x * _LDG_blockDim_x);
    const uint32_t dex_plane_src_y = (threadIdx.y + blockIdx.y * blockDim.y);

    uint32_t dex_plane_dst_y = _logical_stgl.y;

    decx::utils::_cuda_vec128 _reg;

    const uint32_t dex_plane_dst_x = _logical_stgl.x + _CUDA_WARP_SIZE_ * blockIdx.x;
    const uint32_t STG_dex_x = dex_plane_dst_x + dex_plane_src_y * wpitch_dst_v1;
    
    __shared__ float _shmem[_IM2COL_D4N_FP32_BLOCK_Y_ * 4][_IM2COL_D4N_FP32_BLOCK_X_ + 1];

    for (uint32_t i = 0; i < decx::utils::ceil<uint32_t>(kernel_dims.y, gridDim.z); ++i) 
    {
        dex_src = _logical_ldgl.x + (dex_plane_src_x * strides.x + (blockIdx.z + dex_plane_src_y * strides.y + i) * wpitch_src_v1) * 4;
        for (uint32_t j = 0; j < kernel_dims.x; ++j) 
        {
            _reg._vf = decx::utils::vec4_set1_fp32(0);

            if (dex_plane_src_x < conv2D_area.x && dex_plane_src_y < conv2D_area.y) {
                _reg._vf = src[dex_src];
            }

            _shmem[threadIdx.y * 4 + 0][_logical_gl2shmem.x + 32 * _logical_gl2shmem.y] = _reg._arrf[0];
            _shmem[threadIdx.y * 4 + 1][_logical_gl2shmem.x + 32 * _logical_gl2shmem.y] = _reg._arrf[1];
            _shmem[threadIdx.y * 4 + 2][_logical_gl2shmem.x + 32 * _logical_gl2shmem.y] = _reg._arrf[2];
            _shmem[threadIdx.y * 4 + 3][_logical_gl2shmem.x + 32 * _logical_gl2shmem.y] = _reg._arrf[3];

            __syncthreads();

            if (dex_plane_dst_x < conv2D_area.x && dex_plane_src_y < conv2D_area.y)
            {
#pragma unroll 4
                for (int k = 0; k < 4; ++k) {
                    _reg._arrf[k] = _shmem[threadIdx.y * 4 + _logical_stgl.y][_logical_stgl.x + _CUDA_WARP_SIZE_ * k];
                    
                    dst[STG_dex_x + (dex_plane_dst_y + blockIdx.z * kernel_dims.x * 16) * im2col_buf_pitch_v1] = _reg._arrf[k];

                    dex_plane_dst_y += 4;
                }
            }
            
            dex_src += 4;

            __syncthreads();
        }
    }
}




// block[32 * 4, 2] = [128, 2]
template <bool _boundless_T, bool _boundless_B> __global__ void
decx::nn::GPUK::cu_im2col_DP16_BC_fp32(const float4* __restrict  src, 
                                       float* __restrict         dst, 
                                       const uint2               conv2D_area, 
                                       const uint2               kernel_dims,
                                       const uint2               strides,
                                       const uint32_t            wpitch_dst_v1, 
                                       const uint32_t            wpitch_src_v1, 
                                       const uint64_t            im2col_buf_pitch_v1)
{
    constexpr uint32_t _LDG_blockDim_x = _IM2COL_GET_STG_BLOCKDIM_X_(_IM2COL_D4N_FP32_BLOCK_X_, 16);

    uint64_t dex_src = 0;

    const uchar2 _logical_ldgl = make_uchar2(threadIdx.x % 4, threadIdx.x / 4);
    const uchar2 _logical_gl2shmem = make_uchar2((threadIdx.x / 4) % _LDG_blockDim_x, threadIdx.x % 4);
    const uchar2 _logical_stgl = make_uchar2(threadIdx.x % _CUDA_WARP_SIZE_, threadIdx.x / _CUDA_WARP_SIZE_);

    const uint32_t dex_plane_src_x = (_logical_ldgl.y + blockIdx.x * _LDG_blockDim_x);
    const uint32_t dex_plane_src_y = (threadIdx.y + blockIdx.y * blockDim.y);

    uint32_t dex_plane_dst_y = _logical_stgl.y;

    decx::utils::_cuda_vec128 _reg;

    const uint32_t& _half_KH = (kernel_dims.y / 2);

    const uint32_t dex_plane_dst_x = _logical_stgl.x + _CUDA_WARP_SIZE_ * blockIdx.x;
    const uint32_t STG_dex_x = dex_plane_dst_x + dex_plane_src_y * wpitch_dst_v1;
    
    __shared__ float _shmem[_IM2COL_D4N_FP32_BLOCK_Y_ * 4][_IM2COL_D4N_FP32_BLOCK_X_ + 1];

    for (uint32_t i = 0; i < decx::utils::ceil<uint32_t>(kernel_dims.y, gridDim.z); ++i) 
    {
        const uint32_t _global_coor_y = blockIdx.z + dex_plane_src_y * strides.y + i;
        
        if ((_boundless_T || _global_coor_y > _half_KH - 1) && (_boundless_B || _global_coor_y < conv2D_area.y * strides.y + _half_KH))
        {
            dex_src = _logical_ldgl.x + (dex_plane_src_x * strides.x + _global_coor_y * wpitch_src_v1) * 4;

            for (uint32_t j = 0; j < kernel_dims.x; ++j)
            {
                _reg._vf = decx::utils::vec4_set1_fp32(0);

                if (dex_plane_src_x < conv2D_area.x && dex_plane_src_y < conv2D_area.y) {
                    _reg._vf = src[dex_src];
                }

                _shmem[threadIdx.y * 4 + 0][_logical_gl2shmem.x + 32 * _logical_gl2shmem.y] = _reg._arrf[0];
                _shmem[threadIdx.y * 4 + 1][_logical_gl2shmem.x + 32 * _logical_gl2shmem.y] = _reg._arrf[1];
                _shmem[threadIdx.y * 4 + 2][_logical_gl2shmem.x + 32 * _logical_gl2shmem.y] = _reg._arrf[2];
                _shmem[threadIdx.y * 4 + 3][_logical_gl2shmem.x + 32 * _logical_gl2shmem.y] = _reg._arrf[3];

                __syncthreads();

                if (dex_plane_dst_x < conv2D_area.x && dex_plane_src_y < conv2D_area.y)
                {
#pragma unroll 4
                    for (int k = 0; k < 4; ++k) {
                        _reg._arrf[k] = _shmem[threadIdx.y * 4 + _logical_stgl.y][_logical_stgl.x + _CUDA_WARP_SIZE_ * k];

                        dst[STG_dex_x + (dex_plane_dst_y + blockIdx.z * kernel_dims.x * 16) * im2col_buf_pitch_v1] = _reg._arrf[k];

                        dex_plane_dst_y += 4;
                    }
                }

                dex_src += 4;

                __syncthreads();
            }   // end for
        }   // end if
    }
}

template __global__ void decx::nn::GPUK::cu_im2col_DP16_BC_fp32<false, false>(const float4* __restrict, float* __restrict, const uint2, const uint2,
    const uint2, const uint32_t, const uint32_t, const uint64_t);

template __global__ void decx::nn::GPUK::cu_im2col_DP16_BC_fp32<true, false>(const float4* __restrict, float* __restrict, const uint2, const uint2,
    const uint2, const uint32_t, const uint32_t, const uint64_t);

template __global__ void decx::nn::GPUK::cu_im2col_DP16_BC_fp32<false, true>(const float4* __restrict, float* __restrict, const uint2, const uint2,
    const uint2, const uint32_t, const uint32_t, const uint64_t);
