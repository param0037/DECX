/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "im2col.cuh"
#include "../../../BLAS/basic_process/transpose/CUDA/transpose_kernel.cuh"



__global__ void 
decx::conv_I2R::GPUK::cu_Im2Col_v128_fp32(const float4* __restrict      src,
                                          float4* __restrict            dst,
                                          const int2                    thread_bound,
                                          const size_t                  Wpitch_src,
                                          const size_t                  pitch_dst,
                                          const int2                    ker_size,
                                          const int                     depth_v128)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;       // adj
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;       // stride

    size_t glo_dex_src, glo_dex_dst;
    uint ker_row, ker_col;

    float4 _IO[4];
    float tmp;

    for (int i = 0; i < depth_v128; ++i)
    {
        glo_dex_src = (size_t)tidy * Wpitch_src + (size_t)tidx * depth_v128 * 4 + i;
        
        if (tidy < thread_bound.y && tidx < thread_bound.x) {
            for (int ker_iter = 0; ker_iter < ker_size.x * ker_size.y; ++ker_iter)
            {
                ker_row = ker_iter / ker_size.x;
                ker_col = ker_iter % ker_size.x;

                _IO[0] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128];
                _IO[1] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128 + depth_v128];
                _IO[2] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128 + depth_v128 * 2];
                _IO[3] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128 + depth_v128 * 3];

                TRANSPOSE_MAT4X4(_IO, tmp);

                glo_dex_dst = ((size_t)tidy * thread_bound.x + (size_t)tidx) + pitch_dst * 4 * (i + ker_iter * depth_v128);
                dst[glo_dex_dst] = _IO[0];
                dst[glo_dex_dst + pitch_dst] = _IO[1];
                dst[glo_dex_dst + pitch_dst * 2] = _IO[2];
                dst[glo_dex_dst + pitch_dst * 3] = _IO[3];
            }
        }
    }    // looping along the depth
}




__global__ void 
decx::conv_I2R::GPUK::cu_Im2Col_v128_fp16(const float4* __restrict      src,
                                          float4* __restrict            dst,
                                          const int2                    thread_bound,
                                          const size_t                  Wpitch_src,
                                          const size_t                  pitch_dst,
                                          const int2                    ker_size,
                                          const int                     depth_v128)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;       // adj
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;       // stride

    size_t glo_dex_src, glo_dex_dst;
    uint ker_row, ker_col;

    float4 _IO[8];
    __half tmp;

    for (int i = 0; i < depth_v128; ++i)
    {
        glo_dex_src = (size_t)tidy * Wpitch_src + (size_t)tidx * depth_v128 * 8 + i;
        
        if (tidy < thread_bound.y && tidx < thread_bound.x) {
            for (int ker_iter = 0; ker_iter < ker_size.x * ker_size.y; ++ker_iter)
            {
                ker_row = ker_iter / ker_size.x;
                ker_col = ker_iter % ker_size.x;

                _IO[0] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128];
                _IO[1] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128 + depth_v128];
                _IO[2] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128 + depth_v128 * 2];
                _IO[3] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128 + depth_v128 * 3];

                _IO[4] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128 + depth_v128 * 4];
                _IO[5] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128 + depth_v128 * 5];
                _IO[6] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128 + depth_v128 * 6];
                _IO[7] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128 + depth_v128 * 7];

                TRANSPOSE_MAT8x8(_IO, tmp);

                glo_dex_dst = ((size_t)tidy * thread_bound.x + (size_t)tidx) + pitch_dst * 8 * (i + ker_iter * depth_v128);
                dst[glo_dex_dst] = _IO[0];
                dst[glo_dex_dst + pitch_dst] = _IO[1];
                dst[glo_dex_dst + pitch_dst * 2] = _IO[2];
                dst[glo_dex_dst + pitch_dst * 3] = _IO[3];
                dst[glo_dex_dst + pitch_dst * 4] = _IO[4];
                dst[glo_dex_dst + pitch_dst * 5] = _IO[5];
                dst[glo_dex_dst + pitch_dst * 6] = _IO[6];
                dst[glo_dex_dst + pitch_dst * 7] = _IO[7];
            }
        }
    }    // looping along the depth
}


__global__ void 
decx::conv_I2R::GPUK::cu_Im2Col_v128_stride_fp32(const float4* __restrict      src,
                                            float4* __restrict            dst,
                                            const uint2                   strideXY,
                                            const int2                    thread_bound,
                                            const size_t                  Wpitch_src,
                                            const size_t                  pitch_dst,
                                            const int2                    ker_size,
                                            const int                     depth_v128)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;       // adj
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;       // stride

    size_t glo_dex_src, glo_dex_dst;
    uint ker_row, ker_col;
    const uint32_t stride_depth_v128 = depth_v128 * strideXY.x;

    float4 _IO[4];
    float tmp;

    for (int i = 0; i < depth_v128; ++i)
    {
        glo_dex_src = (size_t)tidy * Wpitch_src * strideXY.y + (size_t)tidx * depth_v128 * 4 * strideXY.x + i;

        if (tidy < thread_bound.y && tidx < thread_bound.x) {
            for (int ker_iter = 0; ker_iter < ker_size.x * ker_size.y; ++ker_iter)
            {
                ker_row = ker_iter / ker_size.x;
                ker_col = ker_iter % ker_size.x;

                _IO[0] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128];
                _IO[1] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128 + stride_depth_v128];
                _IO[2] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128 + stride_depth_v128 * 2];
                _IO[3] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128 + stride_depth_v128 * 3];

                TRANSPOSE_MAT4X4(_IO, tmp);

                glo_dex_dst = ((size_t)tidy * thread_bound.x + (size_t)tidx) + pitch_dst * 4 * (i + ker_iter * depth_v128);
                dst[glo_dex_dst] = _IO[0];
                dst[glo_dex_dst + pitch_dst] = _IO[1];
                dst[glo_dex_dst + pitch_dst * 2] = _IO[2];
                dst[glo_dex_dst + pitch_dst * 3] = _IO[3];
            }
        }
    }    // looping along the depth
}




__global__ void 
decx::conv_I2R::GPUK::cu_Im2Col_v128_stride_fp16(const float4* __restrict      src,
                                                 float4* __restrict            dst,
                                                 const uint2                   strideXY,
                                                 const int2                    thread_bound,
                                                 const size_t                  Wpitch_src,
                                                 const size_t                  pitch_dst,
                                                 const int2                    ker_size,
                                                 const int                     depth_v128)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;       // adj
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;       // stride

    size_t glo_dex_src, glo_dex_dst;
    uint ker_row, ker_col;

    const uint32_t stride_depth_v128 = depth_v128 * strideXY.x;

    float4 _IO[8];
    __half tmp;

    for (int i = 0; i < depth_v128; ++i)
    {
        glo_dex_src = (size_t)tidy * Wpitch_src * strideXY.y + (size_t)tidx * depth_v128 * 8 * strideXY.x + i;
        
        if (tidy < thread_bound.y && tidx < thread_bound.x) {
            for (int ker_iter = 0; ker_iter < ker_size.x * ker_size.y; ++ker_iter)
            {
                ker_row = ker_iter / ker_size.x;
                ker_col = ker_iter % ker_size.x;

                _IO[0] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128];
                _IO[1] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128 + stride_depth_v128];
                _IO[2] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128 + stride_depth_v128 * 2];
                _IO[3] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128 + stride_depth_v128 * 3];

                _IO[4] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128 + stride_depth_v128 * 4];
                _IO[5] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128 + stride_depth_v128 * 5];
                _IO[6] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128 + stride_depth_v128 * 6];
                _IO[7] = src[glo_dex_src + ker_row * Wpitch_src + ker_col * depth_v128 + stride_depth_v128 * 7];

                TRANSPOSE_MAT8x8(_IO, tmp);

                glo_dex_dst = ((size_t)tidy * thread_bound.x + (size_t)tidx) + pitch_dst * 8 * (i + ker_iter * depth_v128);

                dst[glo_dex_dst] = _IO[0];
                dst[glo_dex_dst + pitch_dst] = _IO[1];
                dst[glo_dex_dst + pitch_dst * 2] = _IO[2];
                dst[glo_dex_dst + pitch_dst * 3] = _IO[3];
                dst[glo_dex_dst + pitch_dst * 4] = _IO[4];
                dst[glo_dex_dst + pitch_dst * 5] = _IO[5];
                dst[glo_dex_dst + pitch_dst * 6] = _IO[6];
                dst[glo_dex_dst + pitch_dst * 7] = _IO[7];
            }
        }
    }    // looping along the depth
}

