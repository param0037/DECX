/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "Conv2_fp16_kernels_accurate.cuh"


__global__
void decx::conv::GPUK::cu_hConv2_r8_within_accu(const float4* __restrict               src,
                                                const __half* kernel,
                                                float4* __restrict               dst,
                                                const uint              pitch_src,
                                                const uint              pitch_dst,
                                                const uint              total_ker_len,
                                                const uint              Wker,
                                                const uint2              kernel_shift,
                                                const uint2             dst_dims)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;

    __shared__ __half src_frag[32][144 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1, reg_2;
    float fval_ker;

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
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    *((uint4*)&reg_2) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    for (int i = 0; i < total_ker_len; ++i)
    {
        // ���ﲢ����ÿ�ζ��ӹ����ڴ���ȫ�����ݣ�ƽ�ƾͿ�����
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

        fval_ker = __half2float(kernel[i]);

        *((float*)&reg_1.x) = fmaf(__half2float(reg_0.x.x), fval_ker, *((float*)&reg_1.x));
        *((float*)&reg_1.y) = fmaf(__half2float(reg_0.x.y), fval_ker, *((float*)&reg_1.y));
        *((float*)&reg_1.z) = fmaf(__half2float(reg_0.y.x), fval_ker, *((float*)&reg_1.z));
        *((float*)&reg_1.w) = fmaf(__half2float(reg_0.y.y), fval_ker, *((float*)&reg_1.w));
        *((float*)&reg_2.x) = fmaf(__half2float(reg_0.z.x), fval_ker, *((float*)&reg_2.x));
        *((float*)&reg_2.y) = fmaf(__half2float(reg_0.z.y), fval_ker, *((float*)&reg_2.y));
        *((float*)&reg_2.z) = fmaf(__half2float(reg_0.w.x), fval_ker, *((float*)&reg_2.z));
        *((float*)&reg_2.w) = fmaf(__half2float(reg_0.w.y), fval_ker, *((float*)&reg_2.w));
    }

    glo_dex = idx * pitch_dst + idy;

    reg_0.x = __floats2half2_rn(*((float*)&reg_1.x), *((float*)&reg_1.y));
    reg_0.y = __floats2half2_rn(*((float*)&reg_1.z), *((float*)&reg_1.w));
    reg_0.z = __floats2half2_rn(*((float*)&reg_2.x), *((float*)&reg_2.y));
    reg_0.w = __floats2half2_rn(*((float*)&reg_2.z), *((float*)&reg_2.w));

    if (idx < dst_dims.y && idy < dst_dims.x) {
        dst[glo_dex] = *((float4*)&reg_0);
    }
#endif
}





__global__
void decx::conv::GPUK::cu_hConv2_r16_within_accu(const float4* __restrict                src, 
                                                 const __half* kernel,
                                                 float4* __restrict                dst,
                                                 const uint             pitch_src, 
                                                 const uint             pitch_dst,
                                                 const uint             total_ker_len, 
                                                 const uint             Wker,
                                                 const uint2             kernel_shift,
                                                 const uint2            dst_dims)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ __half src_frag[48][160 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1, reg_2;
    float fval_ker;

    uint glo_dex = idx * pitch_src + idy;               *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;                          *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;                          *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(32)

    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;           *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(0)

        glo_dex += 16 * pitch_src;                      *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(16)

        glo_dex += 16 * pitch_src;                      *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    *((uint4*)&reg_2) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    for (int i = 0; i < total_ker_len; ++i)
    {
        // ���ﲢ����ÿ�ζ��ӹ����ڴ���ȫ�����ݣ�ƽ�ƾͿ�����
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

        fval_ker = __half2float(kernel[i]);

        *((float*)&reg_1.x) = fmaf(__half2float(reg_0.x.x), fval_ker, *((float*)&reg_1.x));
        *((float*)&reg_1.y) = fmaf(__half2float(reg_0.x.y), fval_ker, *((float*)&reg_1.y));
        *((float*)&reg_1.z) = fmaf(__half2float(reg_0.y.x), fval_ker, *((float*)&reg_1.z));
        *((float*)&reg_1.w) = fmaf(__half2float(reg_0.y.y), fval_ker, *((float*)&reg_1.w));
        *((float*)&reg_2.x) = fmaf(__half2float(reg_0.z.x), fval_ker, *((float*)&reg_2.x));
        *((float*)&reg_2.y) = fmaf(__half2float(reg_0.z.y), fval_ker, *((float*)&reg_2.y));
        *((float*)&reg_2.z) = fmaf(__half2float(reg_0.w.x), fval_ker, *((float*)&reg_2.z));
        *((float*)&reg_2.w) = fmaf(__half2float(reg_0.w.y), fval_ker, *((float*)&reg_2.w));
    }

    glo_dex = idx * pitch_dst + idy;

    reg_0.x = __floats2half2_rn(*((float*)&reg_1.x), *((float*)&reg_1.y));
    reg_0.y = __floats2half2_rn(*((float*)&reg_1.z), *((float*)&reg_1.w));
    reg_0.z = __floats2half2_rn(*((float*)&reg_2.x), *((float*)&reg_2.y));
    reg_0.w = __floats2half2_rn(*((float*)&reg_2.z), *((float*)&reg_2.w));

    if (idx < dst_dims.y && idy < dst_dims.x) {
        dst[glo_dex] = *((float4*)&reg_0);
    }
#endif
}







__global__
void decx::conv::GPUK::cu_hConv2_r816_within_accu(const float4* __restrict          src,
                                                  const __half* kernel,
                                                  float4* __restrict                dst,
                                                  const uint             pitch_src, 
                                                  const uint             pitch_dst,
                                                  const uint             total_ker_len, 
                                                  const uint             Wker,
                                                  const uint2             kernel_shift,
                                                  const uint2            dst_dims)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ __half src_frag[32][160 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1, reg_2;
    float fval_ker;

    uint glo_dex = idx * pitch_src + idy;               *((float4*)&reg_0) = src[glo_dex];
    glo_dex += 16 * pitch_src;                          *((float4*)&reg_1) = src[glo_dex];

    hstore_to_shmem_L
    
    if (threadIdx.y < 4) {
        glo_dex = idx * pitch_src + idy + 16;           *((float4*)&reg_0) = src[glo_dex];
        glo_dex += 16 * pitch_src;                      *((float4*)&reg_1) = src[glo_dex];

        hstore_to_shmem_R
    }

    __syncthreads();

    int dx, dy;
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    *((uint4*)&reg_2) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);

    for (int i = 0; i < total_ker_len; ++i)
    {
        // ���ﲢ����ÿ�ζ��ӹ����ڴ���ȫ�����ݣ�ƽ�ƾͿ�����
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

        fval_ker = __half2float(kernel[i]);

        *((float*)&reg_1.x) = fmaf(__half2float(reg_0.x.x), fval_ker, *((float*)&reg_1.x));
        *((float*)&reg_1.y) = fmaf(__half2float(reg_0.x.y), fval_ker, *((float*)&reg_1.y));
        *((float*)&reg_1.z) = fmaf(__half2float(reg_0.y.x), fval_ker, *((float*)&reg_1.z));
        *((float*)&reg_1.w) = fmaf(__half2float(reg_0.y.y), fval_ker, *((float*)&reg_1.w));
        *((float*)&reg_2.x) = fmaf(__half2float(reg_0.z.x), fval_ker, *((float*)&reg_2.x));
        *((float*)&reg_2.y) = fmaf(__half2float(reg_0.z.y), fval_ker, *((float*)&reg_2.y));
        *((float*)&reg_2.z) = fmaf(__half2float(reg_0.w.x), fval_ker, *((float*)&reg_2.z));
        *((float*)&reg_2.w) = fmaf(__half2float(reg_0.w.y), fval_ker, *((float*)&reg_2.w));
    }

    glo_dex = idx * pitch_dst + idy;

    reg_0.x = __floats2half2_rn(*((float*)&reg_1.x), *((float*)&reg_1.y));
    reg_0.y = __floats2half2_rn(*((float*)&reg_1.z), *((float*)&reg_1.w));
    reg_0.z = __floats2half2_rn(*((float*)&reg_2.x), *((float*)&reg_2.y));
    reg_0.w = __floats2half2_rn(*((float*)&reg_2.z), *((float*)&reg_2.w));

    if (idx < dst_dims.y && idy < dst_dims.x) {
        dst[glo_dex] = *((float4*)&reg_0);
    }
#endif
}




__global__
/**
* The radius of convolutional kernel = 8��ÿ���̴߳���1x4������(one float4)��һ����16x16���̣߳�
* ��һ������Ҫ�Ĺ����ڴ�СΪ(16 * 4 + 8 * 2)*(16 + 8 * 2) ��shmem[32][80]
* So the alignments should be x64 in width, and x16 in height for Ddst
* The dims of Dsrc should be plus 8 * 2 = 16 on all directions(if float4 is consider horizentally, then +4 at width)
*
* ������64 x 16(floats), �⻷��8 x 8(floats)
* constant area: 64 x 16(floats), apron area: 8 x 8(floats)
* �����ά����8����
* */
void decx::conv::GPUK::cu_hConv2_r168_within_accu(const float4* __restrict          src,
                                                  const __half* kernel,
                                                  float4* __restrict                dst,
                                                  const uint              pitch_src,
                                                  const uint              pitch_dst,
                                                  const uint              total_ker_len,
                                                  const uint              Wker,
                                                  const uint2              kernel_shift,
                                                  const uint2             dst_dims)
{
#if __ABOVE_SM_53
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    int idy = threadIdx.y + blockDim.y * blockIdx.y;
    
    __shared__ __half src_frag[48][144 + sharedmem_offset * 2];

    half2_8 reg_0, reg_1, reg_2;
    float fval_ker;

    uint glo_dex = idx * pitch_src + idy;               *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(0)

    glo_dex += 16 * pitch_src;                          *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(16)

    glo_dex += 16 * pitch_src;                          *((float4*)&reg_0) = src[glo_dex];
    hstore_to_shmem_L3(32)

    if (threadIdx.y < 2) {
        glo_dex = idx * pitch_src + idy + 16;           *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(0)

        glo_dex += 16 * pitch_src;                      *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(16)

        glo_dex += 16 * pitch_src;                      *((float4*)&reg_0) = src[glo_dex];
        hstore_to_shmem_R3(32)
    }

    __syncthreads();

    int dx, dy;
    *((uint4*)&reg_1) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);
    *((uint4*)&reg_2) = make_uint4(init_valueUint, init_valueUint, init_valueUint, init_valueUint);

    for (int i = 0; i < total_ker_len; ++i)
    {
        // ���ﲢ����ÿ�ζ��ӹ����ڴ���ȫ�����ݣ�ƽ�ƾͿ�����
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

        fval_ker = __half2float(kernel[i]);

        *((float*)&reg_1.x) = fmaf(__half2float(reg_0.x.x), fval_ker, *((float*)&reg_1.x));
        *((float*)&reg_1.y) = fmaf(__half2float(reg_0.x.y), fval_ker, *((float*)&reg_1.y));
        *((float*)&reg_1.z) = fmaf(__half2float(reg_0.y.x), fval_ker, *((float*)&reg_1.z));
        *((float*)&reg_1.w) = fmaf(__half2float(reg_0.y.y), fval_ker, *((float*)&reg_1.w));
        *((float*)&reg_2.x) = fmaf(__half2float(reg_0.z.x), fval_ker, *((float*)&reg_2.x));
        *((float*)&reg_2.y) = fmaf(__half2float(reg_0.z.y), fval_ker, *((float*)&reg_2.y));
        *((float*)&reg_2.z) = fmaf(__half2float(reg_0.w.x), fval_ker, *((float*)&reg_2.z));
        *((float*)&reg_2.w) = fmaf(__half2float(reg_0.w.y), fval_ker, *((float*)&reg_2.w));
    }

    glo_dex = idx * pitch_dst + idy;

    reg_0.x = __floats2half2_rn(*((float*)&reg_1.x), *((float*)&reg_1.y));
    reg_0.y = __floats2half2_rn(*((float*)&reg_1.z), *((float*)&reg_1.w));
    reg_0.z = __floats2half2_rn(*((float*)&reg_2.x), *((float*)&reg_2.y));
    reg_0.w = __floats2half2_rn(*((float*)&reg_2.z), *((float*)&reg_2.w));
    if (idx < dst_dims.y && idy < dst_dims.x) {
        dst[glo_dex] = *((float4*)&reg_0);
    }
#endif
}