/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "im2col.cuh"


__global__ void 
decx::conv::GPUK::cu_sIm2Col_r8_within(float4* src,
                            float4* dst,
                            const int2                    kernel_shift,
                            const int2                    thread_bound,
                            const size_t                  Wpitch_src,
                            const size_t                  pitch_dst,
                            const int2                    ker_size,
                            const int                     depth)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;

    size_t glo_dex_src, glo_dex_dst;

    __shared__ float4 frag[32][80 + 1];
    float4 reg_0[4],
        // [0] : shmem_ker_i, [1] : shmem_ker_j, [2] : ker_i, [3] : ker_j
        reg_1[4];

    for (int i = 0; i < depth; ++i)
    {
        glo_dex_src = (size_t)tidx * Wpitch_src + tidy * depth * 4 + i;

        reg_0[0] = src[glo_dex_src];
        reg_0[1] = src[glo_dex_src + depth];
        reg_0[2] = src[glo_dex_src + depth * 2];
        reg_0[3] = src[glo_dex_src + depth * 3];

        glo_dex_src += 16 * Wpitch_src;

        reg_1[0] = src[glo_dex_src];
        reg_1[1] = src[glo_dex_src + depth];
        reg_1[2] = src[glo_dex_src + depth * 2];
        reg_1[3] = src[glo_dex_src + depth * 3];

        store_to_shmem_L_vec4;

        if (threadIdx.y < 4) {
            glo_dex_src = (size_t)tidx * Wpitch_src + (tidy * 4 + 64) * depth + i;
            reg_0[0] = src[glo_dex_src];
            reg_0[1] = src[glo_dex_src + depth];
            reg_0[2] = src[glo_dex_src + depth * 2];
            reg_0[3] = src[glo_dex_src + depth * 3];

            glo_dex_src += 16 * Wpitch_src;

            reg_1[0] = src[glo_dex_src];
            reg_1[1] = src[glo_dex_src + depth];
            reg_1[2] = src[glo_dex_src + depth * 2];
            reg_1[3] = src[glo_dex_src + depth * 3];

            store_to_shmem_R_vec4;
        }

        __syncthreads();

        // glo_dex_src is as the height of dst
        glo_dex_src = ((size_t)tidx * thread_bound.x + (size_t)tidy) * 4;

        if (tidx < thread_bound.y && tidy < thread_bound.x) {
            for (int ker_iter = 0; ker_iter < ker_size.x * ker_size.y; ++ker_iter)
            {
                *((int*)&reg_1[2]) = ker_iter / ker_size.x;
                *((int*)&reg_1[3]) = ker_iter % ker_size.x;
                *((int*)&reg_1[0]) = kernel_shift.x + *((int*)&reg_1[2]);
                *((int*)&reg_1[1]) = kernel_shift.y + *((int*)&reg_1[3]);

                glo_dex_dst = glo_dex_src * pitch_dst + ker_iter * depth + i;

                if (*((int*)&reg_1[3])) {    // not the beginnig of each row on kernel
                    reg_left_shift_float4_4(reg_0);
                    reg_0[3] = frag[threadIdx.x + *((int*)&reg_1[0])][4 * threadIdx.y + *((int*)&reg_1[1]) + 3];
                }
                else {                            // the beginnig of each row on kernel
                    reg_0[0] = frag[threadIdx.x + *((int*)&reg_1[0])][4 * threadIdx.y + *((int*)&reg_1[1])];
                    reg_0[1] = frag[threadIdx.x + *((int*)&reg_1[0])][4 * threadIdx.y + *((int*)&reg_1[1]) + 1];
                    reg_0[2] = frag[threadIdx.x + *((int*)&reg_1[0])][4 * threadIdx.y + *((int*)&reg_1[1]) + 2];
                    reg_0[3] = frag[threadIdx.x + *((int*)&reg_1[0])][4 * threadIdx.y + *((int*)&reg_1[1]) + 3];
                }

                dst[glo_dex_dst] = reg_0[0];            glo_dex_dst += pitch_dst;
                dst[glo_dex_dst] = reg_0[1];            glo_dex_dst += pitch_dst;
                dst[glo_dex_dst] = reg_0[2];            glo_dex_dst += pitch_dst;
                dst[glo_dex_dst] = reg_0[3];
            }
        }

        __syncthreads();
    }    // looping along the depth
}



__global__ void 
decx::conv::GPUK::cu_sIm2Col_r1_exact(float4*                       src,
                          float4*                       dst,
                          const int2                    thread_bound,
                          const size_t                  Wpitch_src,
                          const size_t                  pitch_dst,
                          const int2                    ker_size,
                          const int                     depth)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    
    size_t glo_dex_src, glo_dex_dst;
    __shared__ float4 frag[18][66 + 1];

    float4 reg_0[4],
        // [0] : shmem_ker_i, [1] : shmem_ker_j, [2] : ker_i, [3] : ker_j
        reg_1[4];
    
    for (int i = 0; i < depth; ++i)
    {
        glo_dex_src = (size_t)tidx * Wpitch_src + tidy * depth * 4 + i;

        reg_0[0] = src[glo_dex_src];                        // forward along W
        reg_0[1] = src[glo_dex_src + depth];                // forward along W
        reg_0[2] = src[glo_dex_src + depth * 2];            // forward along W
        reg_0[3] = src[glo_dex_src + depth * 3];            // forward along W

        // store to shared memory
        frag[threadIdx.x][4 * threadIdx.y] = reg_0[0];          frag[threadIdx.x][4 * threadIdx.y + 1] = reg_0[1];
        frag[threadIdx.x][4 * threadIdx.y + 2] = reg_0[2];      frag[threadIdx.x][4 * threadIdx.y + 3] = reg_0[3];

        glo_dex_src += 16 * Wpitch_src;

        if (threadIdx.x < 2) {
            reg_1[0] = src[glo_dex_src];                    // forward along W
            reg_1[1] = src[glo_dex_src + depth];            // forward along W
            reg_1[2] = src[glo_dex_src + depth * 2];        // forward along W
            reg_1[3] = src[glo_dex_src + depth * 3];        // forward along W

            // store to shared memory
            frag[16 + threadIdx.x][4 * threadIdx.y] = reg_1[0];             frag[16 + threadIdx.x][4 * threadIdx.y + 1] = reg_1[1];
            frag[16 + threadIdx.x][4 * threadIdx.y + 2] = reg_1[2];         frag[16 + threadIdx.x][4 * threadIdx.y + 3] = reg_1[3];
        }

        if (threadIdx.y < 2) {
            glo_dex_src = (size_t)tidx * Wpitch_src + (tidy * 4 + 64) * depth + i;
            reg_0[0] = src[glo_dex_src];
            
            // store to shared memory
            frag[threadIdx.x][threadIdx.y + 64] = reg_0[0];

            glo_dex_src += 16 * Wpitch_src;

            if (threadIdx.x < 2) {
                reg_1[0] = src[glo_dex_src];
                
                // store to shared memory
                frag[threadIdx.x + 16][threadIdx.y + 64] = reg_1[0];
            }
        }

        __syncthreads();

        // glo_dex_src is as the height of dst
        glo_dex_src = ((size_t)tidx * thread_bound.x + (size_t)tidy) * 4;

        if (tidx < thread_bound.y && tidy < thread_bound.x) {
            for (int ker_iter = 0; ker_iter < ker_size.x * ker_size.y; ++ker_iter)
            {
                *((int*)&reg_1[2]) = ker_iter / ker_size.x;
                *((int*)&reg_1[3]) = ker_iter % ker_size.x;
                int& dx = *((int*)&reg_1[2]);
                int& dy = *((int*)&reg_1[3]);

                glo_dex_dst = glo_dex_src * pitch_dst + ker_iter * depth + i;

                if (dy) {    // not the beginnig of each row on kernel
                    reg_left_shift_float4_4(reg_0);
                    reg_0[3] = frag[threadIdx.x + dx][4 * threadIdx.y + dy + 3];
                }
                else {                            // the beginnig of each row on kernel
                    reg_0[0] = frag[threadIdx.x + dx][4 * threadIdx.y + dy];
                    reg_0[1] = frag[threadIdx.x + dx][4 * threadIdx.y + dy + 1];
                    reg_0[2] = frag[threadIdx.x + dx][4 * threadIdx.y + dy + 2];
                    reg_0[3] = frag[threadIdx.x + dx][4 * threadIdx.y + dy + 3];
                }

                dst[glo_dex_dst] = reg_0[0];            glo_dex_dst += pitch_dst;
                dst[glo_dex_dst] = reg_0[1];            glo_dex_dst += pitch_dst;
                dst[glo_dex_dst] = reg_0[2];            glo_dex_dst += pitch_dst;
                dst[glo_dex_dst] = reg_0[3];
            }
        }

        __syncthreads();
    }    // looping along the depth
}



__global__ void 
decx::conv::GPUK::cu_sIm2Col_r2_exact(float4*                       src,
                          float4*                       dst,
                          const int2                    thread_bound,
                          const size_t                  Wpitch_src,
                          const size_t                  pitch_dst,
                          const int2                    ker_size,
                          const int                     depth)
{
    int tidx = threadIdx.x + blockDim.x * blockIdx.x;
    int tidy = threadIdx.y + blockDim.y * blockIdx.y;
    
    size_t glo_dex_src, glo_dex_dst;
    __shared__ float4 frag[20][68 + 1];

    float4 reg_0[4],
        // [0] : shmem_ker_i, [1] : shmem_ker_j, [2] : ker_i, [3] : ker_j
        reg_1[4];
    
    for (int i = 0; i < depth; ++i)
    {
        glo_dex_src = (size_t)tidx * Wpitch_src + tidy * depth * 4 + i;

        reg_0[0] = src[glo_dex_src];                        // forward along W
        reg_0[1] = src[glo_dex_src + depth];                // forward along W
        reg_0[2] = src[glo_dex_src + depth * 2];            // forward along W
        reg_0[3] = src[glo_dex_src + depth * 3];            // forward along W

        // store to shared memory
        frag[threadIdx.x][4 * threadIdx.y] = reg_0[0];          frag[threadIdx.x][4 * threadIdx.y + 1] = reg_0[1];
        frag[threadIdx.x][4 * threadIdx.y + 2] = reg_0[2];      frag[threadIdx.x][4 * threadIdx.y + 3] = reg_0[3];

        glo_dex_src += 16 * Wpitch_src;

        if (threadIdx.x < 4) {
            reg_1[0] = src[glo_dex_src];                    // forward along W
            reg_1[1] = src[glo_dex_src + depth];            // forward along W
            reg_1[2] = src[glo_dex_src + depth * 2];        // forward along W
            reg_1[3] = src[glo_dex_src + depth * 3];        // forward along W

            // store to shared memory
            frag[16 + threadIdx.x][4 * threadIdx.y] = reg_1[0];             frag[16 + threadIdx.x][4 * threadIdx.y + 1] = reg_1[1];
            frag[16 + threadIdx.x][4 * threadIdx.y + 2] = reg_1[2];         frag[16 + threadIdx.x][4 * threadIdx.y + 3] = reg_1[3];
        }

        if (threadIdx.y < 4) {
            glo_dex_src = (size_t)tidx * Wpitch_src + (tidy * 4 + 64) * depth + i;
            reg_0[0] = src[glo_dex_src];
            
            // store to shared memory
            frag[threadIdx.x][threadIdx.y + 64] = reg_0[0];

            glo_dex_src += 16 * Wpitch_src;

            if (threadIdx.x < 4) {
                reg_1[0] = src[glo_dex_src];
                
                // store to shared memory
                frag[threadIdx.x + 16][threadIdx.y + 64] = reg_1[0];
            }
        }

        __syncthreads();

        // glo_dex_src is as the height of dst
        glo_dex_src = ((size_t)tidx * thread_bound.x + (size_t)tidy) * 4;

        if (tidx < thread_bound.y && tidy < thread_bound.x) {
            for (int ker_iter = 0; ker_iter < ker_size.x * ker_size.y; ++ker_iter)
            {
                *((int*)&reg_1[2]) = ker_iter / ker_size.x;
                *((int*)&reg_1[3]) = ker_iter % ker_size.x;
                int& dx = *((int*)&reg_1[2]);
                int& dy = *((int*)&reg_1[3]);

                glo_dex_dst = glo_dex_src * pitch_dst + ker_iter * depth + i;

                if (dy) {    // not the beginnig of each row on kernel
                    reg_left_shift_float4_4(reg_0);
                    reg_0[3] = frag[threadIdx.x + dx][4 * threadIdx.y + dy + 3];
                }
                else {                            // the beginnig of each row on kernel
                    reg_0[0] = frag[threadIdx.x + dx][4 * threadIdx.y + dy];
                    reg_0[1] = frag[threadIdx.x + dx][4 * threadIdx.y + dy + 1];
                    reg_0[2] = frag[threadIdx.x + dx][4 * threadIdx.y + dy + 2];
                    reg_0[3] = frag[threadIdx.x + dx][4 * threadIdx.y + dy + 3];
                }

                dst[glo_dex_dst] = reg_0[0];            glo_dex_dst += pitch_dst;
                dst[glo_dex_dst] = reg_0[1];            glo_dex_dst += pitch_dst;
                dst[glo_dex_dst] = reg_0[2];            glo_dex_dst += pitch_dst;
                dst[glo_dex_dst] = reg_0[3];
            }
        }

        __syncthreads();
    }    // looping along the depth
}