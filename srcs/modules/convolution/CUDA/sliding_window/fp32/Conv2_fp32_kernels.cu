/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "Conv2_fp32_kernels.cuh"

__global__
void cu_sConv2_r8_exact(float4*            src, 
                        float4*            dst,
                        const uint        pitch_src, 
                        const uint        pitch_dst,
                        const uint        total_ker_len, 
                        const uint        Wker)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[32][80 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    glo_dex += 16 * pitch_src;
    reg_1 = src[glo_dex];

    store_to_shmem_L
    
    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        glo_dex += 16 * pitch_src;
        reg_1 = src[glo_dex];

        store_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];

        }
        tmp_ker = ((float*)decx::Const_Mem)[i];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}



__global__
void cu_sConv2_r8_within(float4*            src, 
                         float4*            dst,
                         const uint            pitch_src, 
                         const uint            pitch_dst,
                         const uint            total_ker_len, 
                         const uint            Wker,
                         const int2            kernel_shift)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[32][80 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    glo_dex += 16 * pitch_src;
    reg_1 = src[glo_dex];

    store_to_shmem_L
    
    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        glo_dex += 16 * pitch_src;
        reg_1 = src[glo_dex];

        store_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + (i % Wker);
        if (dy == kernel_shift.y) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)decx::Const_Mem)[i];
        
        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;
}




__global__
void cu_sConv2_r16_exact(float4*               src, 
                         float4*               dst,
                         const uint            pitch_src, 
                         const uint            pitch_dst,
                         const uint            total_ker_len, 
                         const uint            Wker)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[48][96 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(32)

    if (threadIdx.y < 8) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(0)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(16)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)decx::Const_Mem)[i];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}




__global__
void cu_sConv2_r16_within(float4*            src, 
                          float4*            dst,
                          const uint        pitch_src, 
                          const uint        pitch_dst,
                          const uint        total_ker_len, 
                          const uint        Wker,
                          const int2        kernel_shift)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ float src_frag[48][96 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(32)

    if (threadIdx.y < 8) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(0)

            glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(16)

            glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + (i % Wker);
        if (dy == kernel_shift.y) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)decx::Const_Mem)[i];
        
        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}




__global__
void cu_sConv2_r816_exact(float4*                src, 
                          float4*                dst,
                          const uint            pitch_src, 
                          const uint            pitch_dst,
                          const uint            total_ker_len, 
                          const uint            Wker)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[32][96 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    glo_dex += 16 * pitch_src;
    reg_1 = src[glo_dex];

    store_to_shmem_L
    
    if (threadIdx.y < 8) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        glo_dex += 16 * pitch_src;
        reg_1 = src[glo_dex];

        store_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)decx::Const_Mem)[i];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}




__global__
void cu_sConv2_r816_within(float4*                src, 
                           float4*                dst,
                           const uint            pitch_src, 
                           const uint            pitch_dst,
                           const uint            total_ker_len, 
                           const uint            Wker,
                           const int2            kernel_shift)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[32][96 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    glo_dex += 16 * pitch_src;
    reg_1 = src[glo_dex];

    store_to_shmem_L
    
    if (threadIdx.y < 8) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        glo_dex += 16 * pitch_src;
        reg_1 = src[glo_dex];

        store_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + i % Wker;
        if (dy == kernel_shift.y) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)decx::Const_Mem)[i];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}




__global__
void cu_sConv2_r168_exact(float4*            src,
                         float4*            dst,
                         const uint            pitch_src,
                         const uint            pitch_dst,
                         const uint            total_ker_len,
                         const uint            Wker)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[48][80 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(32)

    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(0)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(16)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)decx::Const_Mem)[i];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}




__global__
void cu_sConv2_r168_within(float4*                src, 
                           float4*                dst,
                           const uint            pitch_src, 
                           const uint            pitch_dst,
                           const uint            total_ker_len, 
                           const uint            Wker,
                           const int2            kernel_shift)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[48][80 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(32)

    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(0)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(16)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + i % Wker;
        if (dy == kernel_shift.y) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)decx::Const_Mem)[i];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}




__global__
void cu_sConv2_r8_exact_offset(float4*            src, 
                               float4*            dst,
                               const uint        pitch_src, 
                               const uint        pitch_dst,
                               const uint        total_ker_len, 
                               const uint        Wker,
                               const size_t        offset)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[32][80 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    glo_dex += 16 * pitch_src;
    reg_1 = src[glo_dex];

    store_to_shmem_L
    
    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        glo_dex += 16 * pitch_src;
        reg_1 = src[glo_dex];

        store_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];

        }
        tmp_ker = ((float*)&decx::Const_Mem)[i + offset];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}





__global__
void cu_sConv2_r8_within_offset(float4*                src, 
                                float4*                dst,
                                const uint            pitch_src, 
                                const uint            pitch_dst,
                                const uint            total_ker_len, 
                                const uint            Wker,
                                const int2            kernel_shift,
                                const size_t        offset)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[32][80 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    glo_dex += 16 * pitch_src;
    reg_1 = src[glo_dex];

    store_to_shmem_L
    
    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        glo_dex += 16 * pitch_src;
        reg_1 = src[glo_dex];

        store_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + (i % Wker);
        if (dy == kernel_shift.y) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)decx::Const_Mem)[i + offset];
        
        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }
    
    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;
}




__global__
void cu_sConv2_r16_exact_offset(float4*                src, 
                                float4*                dst,
                                const uint            pitch_src, 
                                const uint            pitch_dst,
                                const uint            total_ker_len, 
                                const uint            Wker,
                                const size_t        offset)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[48][96 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(32)

    if (threadIdx.y < 8) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(0)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(16)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)decx::Const_Mem)[i + offset];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}




__global__
void cu_sConv2_r16_within_offset(float4*        src, 
                                 float4*        dst,
                                 const uint        pitch_src, 
                                 const uint        pitch_dst,
                                 const uint        total_ker_len, 
                                 const uint        Wker,
                                 const int2        kernel_shift,
                                 const size_t    offset)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ float src_frag[48][96 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(32)

    if (threadIdx.y < 8) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(0)

            glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(16)

            glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + (i % Wker);
        if (dy == kernel_shift.y) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)decx::Const_Mem)[i + offset];
        
        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}




__global__
void cu_sConv2_r816_exact_offset(float4*            src, 
                                 float4*            dst,
                                 const uint            pitch_src, 
                                 const uint            pitch_dst,
                                 const uint            total_ker_len, 
                                 const uint            Wker,
                                 const size_t        offset)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[32][96 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    glo_dex += 16 * pitch_src;
    reg_1 = src[glo_dex];

    store_to_shmem_L
    
    if (threadIdx.y < 8) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        glo_dex += 16 * pitch_src;
        reg_1 = src[glo_dex];

        store_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)decx::Const_Mem)[i + offset];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}




__global__
void cu_sConv2_r816_within_offset(float4*                src, 
                                  float4*                dst,
                                  const uint            pitch_src, 
                                  const uint            pitch_dst,
                                  const uint            total_ker_len, 
                                  const uint            Wker,
                                  const int2            kernel_shift,
                                  const size_t            offset)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[32][96 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    glo_dex += 16 * pitch_src;
    reg_1 = src[glo_dex];

    store_to_shmem_L
    
    if (threadIdx.y < 8) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        glo_dex += 16 * pitch_src;
        reg_1 = src[glo_dex];

        store_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + i % Wker;
        if (dy == kernel_shift.y) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)decx::Const_Mem)[i + offset];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}




__global__
void cu_sConv2_r168_exact_offset(float4*            src,
                                 float4*            dst,
                                 const uint            pitch_src,
                                 const uint            pitch_dst,
                                 const uint            total_ker_len,
                                 const uint            Wker,
                                 const size_t        offset)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[48][80 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(32)

    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(0)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(16)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = i / Wker;        dy = i % Wker;
        if (dy == 0) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)decx::Const_Mem)[i + offset];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}




__global__
void cu_sConv2_r168_within_offset(float4*                src, 
                                  float4*                dst,
                                  const uint            pitch_src, 
                                  const uint            pitch_dst,
                                  const uint            total_ker_len, 
                                  const uint            Wker,
                                  const int2            kernel_shift,
                                  const size_t            offset)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ float src_frag[48][80 + sharedmem_offset];

    float4 reg_0, reg_1;

    uint glo_dex = idx * pitch_src + idy;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;
    reg_0 = src[glo_dex];
    store_to_shmem_L3(32)

    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(0)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(16)

        glo_dex += 16 * pitch_src;
        reg_0 = src[glo_dex];
        store_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    float tmp_ker;
    reg_1 = make_float4(init_valuef, init_valuef, init_valuef, init_valuef);
    for (int i = 0; i < total_ker_len; ++i)
    {
        dx = kernel_shift.x + i / Wker;        dy = kernel_shift.y + i % Wker;
        if (dy == kernel_shift.y) {
            reg_0.x = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy];
            reg_0.y = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 1];
            reg_0.z = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 2];
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        else {
            reg_shift_f(&reg_0);
            reg_0.w = src_frag[threadIdx.x + dx][4 * (threadIdx.y) + dy + 3];
        }
        tmp_ker = ((float*)decx::Const_Mem)[i + offset];

        reg_1.x = fmaf(reg_0.x, tmp_ker, reg_1.x);    
        reg_1.y = fmaf(reg_0.y, tmp_ker, reg_1.y);    
        reg_1.z = fmaf(reg_0.z, tmp_ker, reg_1.z);    
        reg_1.w = fmaf(reg_0.w, tmp_ker, reg_1.w);    
    }

    glo_dex = idx * pitch_dst + idy;
    dst[glo_dex] = reg_1;        
}

