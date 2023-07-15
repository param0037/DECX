/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "Conv2_fp16_kernels.cuh"


__global__
void decx::conv::GPUK::cu_hConv2_r8_within(const float4* __restrict            src, 
                         const __half* __restrict     kernel,
                         float4* __restrict            dst,
                         const uint            pitch_src, 
                         const uint            pitch_dst,
                         const uint            total_ker_len, 
                         const uint            Wker,
                         const uint2            kernel_shift,
                         const uint2           dst_dims)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ __half src_frag[32][144 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    *((float4*)&reg_0) = src[glo_dex];
    glo_dex += 16 * pitch_src;
    *((float4*)&reg_1) = src[glo_dex];

    hstore_to_shmem_L
    
    if (threadIdx.y < 2) {
        glo_dex = idx * pitch_src + idy + 16;
        *((float4*)&reg_0) = src[glo_dex];
        glo_dex += 16 * pitch_src;
        *((float4*)&reg_1) = src[glo_dex];

        hstore_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    half2 tmp_ker;
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    for (int i = 0; i < total_ker_len; ++i)
    {
        // 这里并不用每次都从共享内存拉全部数据，平移就可以了
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + (i % Wker);
        if (dy == kernel_shift.y) {
            ((__half*)&reg_0)[0] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy];
            ((__half*)&reg_0)[1] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 1];
            ((__half*)&reg_0)[2] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 2];
            ((__half*)&reg_0)[3] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 3];
            ((__half*)&reg_0)[4] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 4];
            ((__half*)&reg_0)[5] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 5];
            ((__half*)&reg_0)[6] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 6];
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }
        else {
            reg_shift_fp16(&reg_0);
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }

        tmp_ker.x = kernel[i];
        tmp_ker.y = tmp_ker.x;
        
        reg_1.x = __hfma2(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = __hfma2(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = __hfma2(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = __hfma2(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    
    if (idx < dst_dims.y && idy < dst_dims.x) {
        dst[glo_dex] = *((float4*)&reg_1);
    }
#endif
}





__global__
void decx::conv::GPUK::cu_hConv2_r16_within(const float4* __restrict          src, 
                                            const __half* __restrict     kernel,
                                            float4* __restrict                dst,
                                            const uint            pitch_src, 
                                            const uint            pitch_dst,
                                            const uint            total_ker_len, 
                                            const uint            Wker,
                                            const uint2            kernel_shift,
                                            const uint2           dst_dims)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ __half src_frag[48][160 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(32)

    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(0)

        glo_dex += 16 * pitch_src;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(16)

        glo_dex += 16 * pitch_src;
        *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    half2 tmp_ker;
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    for (int i = 0; i < total_ker_len; ++i)
    {
        // 这里并不用每次都从共享内存拉全部数据，平移就可以了
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + (i % Wker);
        if (dy == kernel_shift.y) {
            ((__half*)&reg_0)[0] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy];
            ((__half*)&reg_0)[1] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 1];
            ((__half*)&reg_0)[2] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 2];
            ((__half*)&reg_0)[3] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 3];
            ((__half*)&reg_0)[4] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 4];
            ((__half*)&reg_0)[5] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 5];
            ((__half*)&reg_0)[6] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 6];
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }
        else {
            reg_shift_fp16(&reg_0);
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }

        tmp_ker.x = kernel[i];
        tmp_ker.y = tmp_ker.x;

        reg_1.x = __hfma2(reg_0.x, tmp_ker, reg_1.x);
        reg_1.y = __hfma2(reg_0.y, tmp_ker, reg_1.y);
        reg_1.z = __hfma2(reg_0.z, tmp_ker, reg_1.z);
        reg_1.w = __hfma2(reg_0.w, tmp_ker, reg_1.w);
    }

    glo_dex = idx * pitch_dst + idy;

    if (idx < dst_dims.y && idy < dst_dims.x) {
        dst[glo_dex] = *((float4*)&reg_1);
    }
#endif
}



__global__
void decx::conv::GPUK::cu_hConv2_r816_within(const float4* __restrict         src, 
                                             const __half* __restrict     kernel,
                                             float4* __restrict               dst,
                                             const uint            pitch_src, 
                                             const uint            pitch_dst,
                                             const uint            total_ker_len, 
                                             const uint            Wker,
                                             const uint2            kernel_shift,
                                             const uint2           dst_dims)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ __half src_frag[32][160 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    *((float4*)&reg_0) = src[glo_dex];
    glo_dex += 16 * pitch_src;
    *((float4*)&reg_1) = src[glo_dex];

    hstore_to_shmem_L
    
    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        *((float4*)&reg_0) = src[glo_dex];
        glo_dex += 16 * pitch_src;
        *((float4*)&reg_1) = src[glo_dex];

        hstore_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    half2 tmp_ker;
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    for (int i = 0; i < total_ker_len; ++i)
    {
        // 这里并不用每次都从共享内存拉全部数据，平移就可以了
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + (i % Wker);
        if (dy == kernel_shift.y) {
            ((__half*)&reg_0)[0] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy];
            ((__half*)&reg_0)[1] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 1];
            ((__half*)&reg_0)[2] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 2];
            ((__half*)&reg_0)[3] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 3];
            ((__half*)&reg_0)[4] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 4];
            ((__half*)&reg_0)[5] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 5];
            ((__half*)&reg_0)[6] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 6];
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }
        else {
            reg_shift_fp16(&reg_0);
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }

        tmp_ker.x = kernel[i];
        tmp_ker.y = tmp_ker.x;

        reg_1.x = __hfma2(reg_0.x, tmp_ker, reg_1.x);
        reg_1.y = __hfma2(reg_0.y, tmp_ker, reg_1.y);
        reg_1.z = __hfma2(reg_0.z, tmp_ker, reg_1.z);
        reg_1.w = __hfma2(reg_0.w, tmp_ker, reg_1.w);
    }

    glo_dex = idx * pitch_dst + idy;

    if (idx < dst_dims.y && idy < dst_dims.x) {
        dst[glo_dex] = *((float4*)&reg_1);
    }
#endif
}





__global__
void decx::conv::GPUK::cu_hConv2_r168_within(const float4* __restrict       src,
                                             const __half* __restrict     kernel,
                                             float4* __restrict             dst,
                                             const uint            pitch_src,
                                             const uint            pitch_dst,
                                             const uint            total_ker_len,
                                             const uint            Wker,
                                             const uint2            kernel_shift,
                                             const uint2           dst_dims)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ __half src_frag[48][144 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(0)

        glo_dex += 16 * pitch_src;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(16)

        glo_dex += 16 * pitch_src;
    *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(32)

        if (threadIdx.y < 2) {
            glo_dex = idx * pitch_src + idy + 16;
            *((float4*)&reg_0) = src[glo_dex];
            hstore_to_shmem_R3(0)

                glo_dex += 16 * pitch_src;
            *((float4*)&reg_0) = src[glo_dex];
            hstore_to_shmem_R3(16)

                glo_dex += 16 * pitch_src;
            *((float4*)&reg_0) = src[glo_dex];
            hstore_to_shmem_R3(32)
        }

    __syncthreads();

    int dx, dy;
    half2 tmp_ker;
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    for (int i = 0; i < total_ker_len; ++i)
    {
        // 这里并不用每次都从共享内存拉全部数据，平移就可以了
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + (i % Wker);
        if (dy == kernel_shift.y) {
            ((__half*)&reg_0)[0] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy];
            ((__half*)&reg_0)[1] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 1];
            ((__half*)&reg_0)[2] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 2];
            ((__half*)&reg_0)[3] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 3];
            ((__half*)&reg_0)[4] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 4];
            ((__half*)&reg_0)[5] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 5];
            ((__half*)&reg_0)[6] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 6];
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }
        else {
            reg_shift_fp16(&reg_0);
            ((__half*)&reg_0)[7] = src_frag[threadIdx.x + dx][8 * (threadIdx.y) + dy + 7];
        }

        tmp_ker.x = kernel[i];
        tmp_ker.y = tmp_ker.x;

        reg_1.x = __hfma2(reg_0.x, tmp_ker, reg_1.x);
        reg_1.y = __hfma2(reg_0.y, tmp_ker, reg_1.y);
        reg_1.z = __hfma2(reg_0.z, tmp_ker, reg_1.z);
        reg_1.w = __hfma2(reg_0.w, tmp_ker, reg_1.w);
    }

    glo_dex = idx * pitch_dst + idy;
    if (idx < dst_dims.y && idy < dst_dims.x) {
        dst[glo_dex] = *((float4*)&reg_1);
    }
#endif
}