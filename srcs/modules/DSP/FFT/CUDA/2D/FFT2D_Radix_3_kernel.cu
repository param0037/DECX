/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "FFT2D_Radix_3_kernel.cuh"



__global__
void decx::signal::GPUK::cu_FFT2D_R3_R2C_first(const float* src, 
                            float2* dst, 
                            const uint B_ops_num,
                            const uint pitchsrc,        // in float
                            const uint pitchdst,
                            const uint procH)        // in float2
{
    uint dex_w = threadIdx.y + blockDim.y * blockIdx.y;
    uint dex_h = threadIdx.x + blockDim.x * blockIdx.x;

    size_t dex_src = dex_h * pitchsrc + dex_w;
    size_t dex_base_dst = dex_h * pitchdst;

    float recv[3];
    de::CPf res[3] = { de::CPf(0, 0), de::CPf(0, 0), de::CPf(0, 0) };

    if (dex_h < procH && dex_w < B_ops_num) {
        recv[0] = src[dex_src];
        recv[1] = src[dex_src + B_ops_num];
        recv[2] = src[dex_src + B_ops_num * 2];

        res[0].real = __fadd_rn(__fadd_rn(recv[0], recv[1]), recv[2]);

        res[1].real = fmaf(-0.5f, __fadd_rn(recv[1], recv[2]), recv[0]);
        res[1].image = __fmul_rn(__fsub_rn(recv[1], recv[2]), 0.8660254f);

        res[2].real = res[1].real;
        res[2].image = __fmul_rn(__fsub_rn(recv[2], recv[1]), 0.8660254f);

        dst[dex_base_dst + dex_w * 3] = *((float2*)&res[0]);
        dst[dex_base_dst + dex_w * 3 + 1] = *((float2*)&res[1]);
        dst[dex_base_dst + dex_w * 3 + 2] = *((float2*)&res[2]);
    }
}




__global__
void decx::signal::GPUK::cu_FFT2D_R3_C2C_first(const float2* src, 
                            float2* dst, 
                            const uint B_ops_num,
                            const uint pitch,       // in float2
                            const uint procH)
{
    using namespace decx::signal::cuda::dev;
    uint dex_w = threadIdx.y + blockDim.y * blockIdx.y;
    uint dex_h = threadIdx.x + blockDim.x * blockIdx.x;

    size_t dex_base_dst = dex_h * pitch;

    de::CPf recv[3];
    de::CPf res[3] = { de::CPf(0, 0), de::CPf(0, 0), de::CPf(0, 0) };

    if (dex_h < procH && dex_w < B_ops_num) {
        *((float2*)&recv[0]) = src[dex_base_dst + dex_w];
        *((float2*)&recv[1]) = src[dex_base_dst + dex_w + B_ops_num];
        *((float2*)&recv[2]) = src[dex_base_dst + dex_w + B_ops_num * 2];
        
        res[0].real = __fadd_rn(__fadd_rn(recv[0].real, recv[1].real), recv[2].real);
        res[0].image = __fadd_rn(__fadd_rn(recv[0].image, recv[1].image), recv[2].image);

        res[1] = _complex_fma(recv[1], de::CPf(-0.5, 0.8660254f), recv[0]);
        res[1] = _complex_fma(recv[2], de::CPf(-0.5, -0.8660254f), res[1]);

        res[2] = _complex_fma(recv[1], de::CPf(-0.5, -0.8660254f), recv[0]);
        res[2] = _complex_fma(recv[2], de::CPf(-0.5, 0.8660254f), res[2]);

        dst[dex_base_dst + dex_w * 3] = *((float2*)&res[0]);
        dst[dex_base_dst + dex_w * 3 + 1] = *((float2*)&res[1]);
        dst[dex_base_dst + dex_w * 3 + 2] = *((float2*)&res[2]);
    }
}


__global__
void decx::signal::GPUK::cu_IFFT2D_R3_C2R_once(const float2* src, 
                            float* dst, 
                            const uint B_ops_num,
                            const uint pitch,       // in float2
                            const uint procH)
{
    using namespace decx::signal::cuda::dev;
    uint dex_w = threadIdx.y + blockDim.y * blockIdx.y;
    uint dex_h = threadIdx.x + blockDim.x * blockIdx.x;

    size_t dex_base_dst = dex_h * pitch;

    de::CPf recv[3], tmp;
    float res[3];
    const float signal_len = (float)B_ops_num * 3;

    if (dex_h < procH && dex_w < B_ops_num) {
        *((float2*)&recv[0]) = src[dex_base_dst + dex_w];
        *((float2*)&recv[1]) = src[dex_base_dst + dex_w + B_ops_num];
        *((float2*)&recv[2]) = src[dex_base_dst + dex_w + B_ops_num * 2];

        res[0] = __fadd_rn(__fadd_rn(recv[0].real, recv[1].real), recv[2].real);

        tmp = _complex_fma(recv[1], de::CPf(-0.5, 0.8660254f), recv[0]);
        res[1] = _complex_fma_preserve_R(recv[2], de::CPf(-0.5, -0.8660254f), tmp);

        tmp = _complex_fma(recv[1], de::CPf(-0.5, -0.8660254f), recv[0]);
        res[2] = _complex_fma_preserve_R(recv[2], de::CPf(-0.5, 0.8660254f), tmp);

#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            res[i] = __fdividef(res[i], signal_len);
        }

        dst[dex_base_dst * 2 + dex_w * 3] = res[0];
        dst[dex_base_dst * 2 + dex_w * 3 + 1] = res[1];
        dst[dex_base_dst * 2 + dex_w * 3 + 2] = res[2];
    }
}


__global__
void decx::signal::GPUK::cu_IFFT2D_R3_C2C_first(const float2* src, 
                            float2* dst, 
                            const uint B_ops_num,
                            const uint pitch,       // in float2
                            const uint procH)
{
    using namespace decx::signal::cuda::dev;
    uint dex_w = threadIdx.y + blockDim.y * blockIdx.y;
    uint dex_h = threadIdx.x + blockDim.x * blockIdx.x;

    size_t dex_base_dst = dex_h * pitch;

    de::CPf recv[3];
    de::CPf res[3] = { de::CPf(0, 0), de::CPf(0, 0), de::CPf(0, 0) };

    const float signal_len = (float)B_ops_num * 3;

    if (dex_h < procH && dex_w < B_ops_num) {
        *((float2*)&recv[0]) = src[dex_base_dst + dex_w];
        *((float2*)&recv[1]) = src[dex_base_dst + dex_w + B_ops_num];
        *((float2*)&recv[2]) = src[dex_base_dst + dex_w + B_ops_num * 2];
        
        // conj and * /= signal_len
        recv[0].real = __fdividef(recv[0].real, signal_len);        recv[0].image = __fdividef(recv[0].image, -signal_len);
        recv[1].real = __fdividef(recv[1].real, signal_len);        recv[1].image = __fdividef(recv[1].image, -signal_len);
        recv[2].real = __fdividef(recv[2].real, signal_len);        recv[2].image = __fdividef(recv[2].image, -signal_len);

        res[0].real = __fadd_rn(__fadd_rn(recv[0].real, recv[1].real), recv[2].real);
        res[0].image = __fadd_rn(__fadd_rn(recv[0].image, recv[1].image), recv[2].image);

        res[1] = _complex_fma(recv[1], de::CPf(-0.5, 0.8660254f), recv[0]);
        res[1] = _complex_fma(recv[2], de::CPf(-0.5, -0.8660254f), res[1]);

        res[2] = _complex_fma(recv[1], de::CPf(-0.5, -0.8660254f), recv[0]);
        res[2] = _complex_fma(recv[2], de::CPf(-0.5, 0.8660254f), res[2]);

        dst[dex_base_dst + dex_w * 3] = *((float2*)&res[0]);
        dst[dex_base_dst + dex_w * 3 + 1] = *((float2*)&res[1]);
        dst[dex_base_dst + dex_w * 3 + 2] = *((float2*)&res[2]);
    }
}



__global__
void decx::signal::GPUK::cu_FFT2D_R3_C2C(const float2* src, 
                     float2* dst, 
                     const uint B_ops_num, 
                     const uint warp_proc_len,
                     const uint pitch,              // in de::CPf
                     const uint procH)
{
    using namespace decx::signal::cuda::dev;
    uint dex_w = threadIdx.y + blockDim.y * blockIdx.y;
    uint dex_h = threadIdx.x + blockDim.x * blockIdx.x;
    uint dex_store_w;

    size_t dex_base = dex_h * pitch;

    uint warp_loc_id, num_of_Bcalc_in_warp = warp_proc_len / 3;

    de::CPf recv[3], tmp;
    de::CPf W, res[3] = { de::CPf(0, 0), de::CPf(0, 0), de::CPf(0, 0) };

    if (dex_h < procH && dex_w < B_ops_num) {
        *((float2*)&recv[0]) = src[dex_base + dex_w];
        *((float2*)&recv[1]) = src[dex_base + dex_w + B_ops_num];
        *((float2*)&recv[2]) = src[dex_base + dex_w + B_ops_num * 2];

        warp_loc_id = dex_w % num_of_Bcalc_in_warp;
        W.dev_construct_with_phase(Two_Pi * (float)warp_loc_id / (float)warp_proc_len);
        recv[1] = _complex_mul(recv[1], W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 2) / (float)warp_proc_len);
        recv[2] = _complex_mul(recv[2], W);

        res[0].real = recv[0].real + recv[1].real + recv[2].real;
        res[0].image = recv[0].image + recv[1].image + recv[2].image;

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(-0.5, 0.8660254f), tmp);
        res[1] = _complex_fma(recv[2], de::CPf(-0.5, -0.8660254f), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(-0.5, -0.8660254f), tmp);
        res[2] = _complex_fma(recv[2], de::CPf(-0.5, 0.8660254f), tmp);

        dex_store_w = (dex_w / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id;

        dst[dex_base + dex_store_w] = *((float2*)&res[0]);
        dst[dex_base + dex_store_w + num_of_Bcalc_in_warp] = *((float2*)&res[1]);
        dst[dex_base + dex_store_w + num_of_Bcalc_in_warp * 2] = *((float2*)&res[2]);
    }
}


__global__
void decx::signal::GPUK::cu_IFFT2D_R3_C2R_last(const float2* src, 
                           float* dst, 
                           const uint B_ops_num, 
                           const uint warp_proc_len,
                           const uint pitch,              // in de::CPf
                           const uint procH)
{
    using namespace decx::signal::cuda::dev;
    uint dex_w = threadIdx.y + blockDim.y * blockIdx.y;
    uint dex_h = threadIdx.x + blockDim.x * blockIdx.x;
    uint dex_store_w;

    size_t dex_base = dex_h * pitch;

    uint warp_loc_id, num_of_Bcalc_in_warp = warp_proc_len / 3;

    de::CPf recv[3], tmp;
    de::CPf W;
    float res[3];
    const float signal_len = (float)B_ops_num * 3;

    if (dex_h < procH && dex_w < B_ops_num) {
        *((float2*)&recv[0]) = src[dex_base + dex_w];
        *((float2*)&recv[1]) = src[dex_base + dex_w + B_ops_num];
        *((float2*)&recv[2]) = src[dex_base + dex_w + B_ops_num * 2];

        warp_loc_id = dex_w % num_of_Bcalc_in_warp;
        W.dev_construct_with_phase(Two_Pi * (float)warp_loc_id / (float)warp_proc_len);
        recv[1] = _complex_mul(recv[1], W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 2) / (float)warp_proc_len);
        recv[2] = _complex_mul(recv[2], W);

        res[0] = recv[0].real + recv[1].real + recv[2].real;

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(-0.5, 0.8660254f), tmp);
        res[1] = _complex_fma_preserve_R(recv[2], de::CPf(-0.5, -0.8660254f), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(-0.5, -0.8660254f), tmp);
        res[2] = _complex_fma_preserve_R(recv[2], de::CPf(-0.5, 0.8660254f), tmp);

        dex_store_w = (dex_w / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id;

#pragma unroll 3
        for (int i = 0; i < 3; ++i) {
            res[i] = __fdividef(res[i], signal_len);
        }

        dst[dex_base * 2 + dex_store_w] = res[0];
        dst[dex_base * 2 + dex_store_w + num_of_Bcalc_in_warp] = res[1];
        dst[dex_base * 2 + dex_store_w + num_of_Bcalc_in_warp * 2] = res[2];
    }
}


__global__
/*
* @param B_ops_num : in Vec4
* @param warp_proc_len : element
* @param pitch : in float4
*/
void decx::signal::GPUK::cu_FFT2D_R3_C2C_vec4(const float4* src, 
                          float4* dst, 
                          const uint B_ops_num, 
                          const uint warp_proc_len,
                          const uint pitch,
                          const uint procH)
{
    using namespace decx::signal::cuda::dev;
    uint dex_w = threadIdx.y + blockDim.y * blockIdx.y;
    uint dex_h = threadIdx.x + blockDim.x * blockIdx.x;
    uint dex_store_w;

    size_t dex_base = dex_h * pitch;

    uint warp_loc_id, num_of_Bcalc_in_warp = warp_proc_len / 3 / 4;

    de::CPf W/*, tmp[4]*/;
    float4 recv[6], res;

    if (dex_h < procH && dex_w < B_ops_num) {
        recv[0] = src[dex_base + dex_w * 2];
        recv[1] = src[dex_base + dex_w * 2 + 1];
        recv[2] = src[dex_base + (dex_w + B_ops_num) * 2];
        recv[3] = src[dex_base + (dex_w + B_ops_num) * 2 + 1];
        recv[4] = src[dex_base + (dex_w + B_ops_num * 2) * 2];
        recv[5] = src[dex_base + (dex_w + B_ops_num * 2) * 2 + 1];

        warp_loc_id = dex_w % (size_t)num_of_Bcalc_in_warp;
        dex_store_w = (dex_w / (size_t)num_of_Bcalc_in_warp) * (size_t)warp_proc_len / 4 + warp_loc_id;

        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4) / (float)warp_proc_len);
        *((de::CPf*)&recv[2].x) = _complex_mul(*((de::CPf*)&recv[2].x), W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 1) / (float)warp_proc_len);
        *((de::CPf*)&recv[2].z) = _complex_mul(*((de::CPf*)&recv[2].z), W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 2) / (float)warp_proc_len);
        *((de::CPf*)&recv[3].x) = _complex_mul(*((de::CPf*)&recv[3].x), W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 3) / (float)warp_proc_len);
        *((de::CPf*)&recv[3].z) = _complex_mul(*((de::CPf*)&recv[3].z), W);

        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4) * 2 / (float)warp_proc_len);
        *((de::CPf*)&recv[4].x) = _complex_mul(*((de::CPf*)&recv[4].x), W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 1) * 2 / (float)warp_proc_len);
        *((de::CPf*)&recv[4].z) = _complex_mul(*((de::CPf*)&recv[4].z), W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 2) * 2 / (float)warp_proc_len);
        *((de::CPf*)&recv[5].x) = _complex_mul(*((de::CPf*)&recv[5].x), W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 3) * 2 / (float)warp_proc_len);
        *((de::CPf*)&recv[5].z) = _complex_mul(*((de::CPf*)&recv[5].z), W);

        // output 1
        res.x = recv[0].x + recv[2].x + recv[4].x;
        res.y = recv[0].y + recv[2].y + recv[4].y;
        res.z = recv[0].z + recv[2].z + recv[4].z;
        res.w = recv[0].w + recv[2].w + recv[4].w;
        dst[dex_base + dex_store_w * 2] = res;

        res.x = recv[1].x + recv[3].x + recv[5].x;
        res.y = recv[1].y + recv[3].y + recv[5].y;
        res.z = recv[1].z + recv[3].z + recv[5].z;
        res.w = recv[1].w + recv[3].w + recv[5].w;
        dst[dex_base + dex_store_w * 2 + 1] = res;

        // output 2
        dex_store_w += num_of_Bcalc_in_warp;
        res.x = recv[0].x;          res.y = recv[0].y;
        res.z = recv[0].z;          res.w = recv[0].w;
        *((de::CPf*)&res.x) = _complex_fma(*((de::CPf*)&recv[2].x), de::CPf(-0.5, 0.8660254f), *((de::CPf*)&res.x));
        *((de::CPf*)&res.x) = _complex_fma(*((de::CPf*)&recv[4].x), de::CPf(-0.5, -0.8660254f), *((de::CPf*)&res.x));
        *((de::CPf*)&res.z) = _complex_fma(*((de::CPf*)&recv[2].z), de::CPf(-0.5, 0.8660254f), *((de::CPf*)&res.z));
        *((de::CPf*)&res.z) = _complex_fma(*((de::CPf*)&recv[4].z), de::CPf(-0.5, -0.8660254f), *((de::CPf*)&res.z));
        dst[dex_base + dex_store_w * 2] = res;

        res.x = recv[1].x;          res.y = recv[1].y;
        res.z = recv[1].z;          res.w = recv[1].w;
        *((de::CPf*)&res.x) = _complex_fma(*((de::CPf*)&recv[3].x), de::CPf(-0.5, 0.8660254f), *((de::CPf*)&res.x));
        *((de::CPf*)&res.x) = _complex_fma(*((de::CPf*)&recv[5].x), de::CPf(-0.5, -0.8660254f), *((de::CPf*)&res.x));
        *((de::CPf*)&res.z) = _complex_fma(*((de::CPf*)&recv[3].z), de::CPf(-0.5, 0.8660254f), *((de::CPf*)&res.z));
        *((de::CPf*)&res.z) = _complex_fma(*((de::CPf*)&recv[5].z), de::CPf(-0.5, -0.8660254f), *((de::CPf*)&res.z));
        dst[dex_base + dex_store_w * 2 + 1] = res;

        // output 3
        dex_store_w += num_of_Bcalc_in_warp;
        res.x = recv[0].x;          res.y = recv[0].y;
        res.z = recv[0].z;          res.w = recv[0].w;
        *((de::CPf*)&res.x) = _complex_fma(*((de::CPf*)&recv[2].x), de::CPf(-0.5, -0.8660254f), *((de::CPf*)&res.x));
        *((de::CPf*)&res.x) = _complex_fma(*((de::CPf*)&recv[4].x), de::CPf(-0.5, 0.8660254f), *((de::CPf*)&res.x));
        *((de::CPf*)&res.z) = _complex_fma(*((de::CPf*)&recv[2].z), de::CPf(-0.5, -0.8660254f), *((de::CPf*)&res.z));
        *((de::CPf*)&res.z) = _complex_fma(*((de::CPf*)&recv[4].z), de::CPf(-0.5, 0.8660254f), *((de::CPf*)&res.z));
        dst[dex_base + dex_store_w * 2] = res;

        res.x = recv[1].x;          res.y = recv[1].y;
        res.z = recv[1].z;          res.w = recv[1].w;
        *((de::CPf*)&res.x) = _complex_fma(*((de::CPf*)&recv[3].x), de::CPf(-0.5, -0.8660254f), *((de::CPf*)&res.x));
        *((de::CPf*)&res.x) = _complex_fma(*((de::CPf*)&recv[5].x), de::CPf(-0.5, 0.8660254f), *((de::CPf*)&res.x));
        *((de::CPf*)&res.z) = _complex_fma(*((de::CPf*)&recv[3].z), de::CPf(-0.5, -0.8660254f), *((de::CPf*)&res.z));
        *((de::CPf*)&res.z) = _complex_fma(*((de::CPf*)&recv[5].z), de::CPf(-0.5, 0.8660254f), *((de::CPf*)&res.z));
        dst[dex_base + dex_store_w * 2 + 1] = res;
    }
}



__global__
/*
* @param B_ops_num : in Vec4
* @param warp_proc_len : element
* @param pitch : in float4
*/
void decx::signal::GPUK::cu_IFFT2D_R3_C2R_vec4_last(const float4* src, 
                                float4* dst, 
                                const uint B_ops_num, 
                                const uint warp_proc_len,
                                const uint pitch,
                                const uint procH)
{
    using namespace decx::signal::cuda::dev;
#define _fdividef_float4_ \
    res.x = __fdividef(res.x, signal_len);  \
    res.y = __fdividef(res.y, signal_len);  \
    res.z = __fdividef(res.z, signal_len);  \
    res.w = __fdividef(res.w, signal_len);  \

    uint dex_w = threadIdx.y + blockDim.y * blockIdx.y;
    uint dex_h = threadIdx.x + blockDim.x * blockIdx.x;
    uint dex_store_w;

    size_t dex_base = dex_h * pitch;
    uint warp_loc_id, num_of_Bcalc_in_warp = warp_proc_len / 3 / 4;

    const float signal_len = (float)B_ops_num * 3 * 4;

    de::CPf W, tmp;
    float4 recv[6], res;

    if (dex_h < procH && dex_w < B_ops_num) {
        recv[0] = src[dex_base + dex_w * 2];
        recv[1] = src[dex_base + dex_w * 2 + 1];
        recv[2] = src[dex_base + (dex_w + B_ops_num) * 2];
        recv[3] = src[dex_base + (dex_w + B_ops_num) * 2 + 1];
        recv[4] = src[dex_base + (dex_w + B_ops_num * 2) * 2];
        recv[5] = src[dex_base + (dex_w + B_ops_num * 2) * 2 + 1];

        warp_loc_id = dex_w % (size_t)num_of_Bcalc_in_warp;
        dex_store_w = (dex_w / (size_t)num_of_Bcalc_in_warp) * (size_t)warp_proc_len / 4 + warp_loc_id;

        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4) / (float)warp_proc_len);
        *((de::CPf*)&recv[2].x) = _complex_mul(*((de::CPf*)&recv[2].x), W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 1) / (float)warp_proc_len);
        *((de::CPf*)&recv[2].z) = _complex_mul(*((de::CPf*)&recv[2].z), W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 2) / (float)warp_proc_len);
        *((de::CPf*)&recv[3].x) = _complex_mul(*((de::CPf*)&recv[3].x), W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 3) / (float)warp_proc_len);
        *((de::CPf*)&recv[3].z) = _complex_mul(*((de::CPf*)&recv[3].z), W);

        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4) * 2 / (float)warp_proc_len);
        *((de::CPf*)&recv[4].x) = _complex_mul(*((de::CPf*)&recv[4].x), W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 1) * 2 / (float)warp_proc_len);
        *((de::CPf*)&recv[4].z) = _complex_mul(*((de::CPf*)&recv[4].z), W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 2) * 2 / (float)warp_proc_len);
        *((de::CPf*)&recv[5].x) = _complex_mul(*((de::CPf*)&recv[5].x), W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 3) * 2 / (float)warp_proc_len);
        *((de::CPf*)&recv[5].z) = _complex_mul(*((de::CPf*)&recv[5].z), W);

        // output 1
        res.x = recv[0].x + recv[2].x + recv[4].x;
        res.y = recv[0].z + recv[2].z + recv[4].z;
        res.z = recv[1].x + recv[3].x + recv[5].x;
        res.w = recv[1].z + recv[3].z + recv[5].z;

        _fdividef_float4_;
        dst[dex_base + dex_store_w] = res;      // store

        // output 2
        dex_store_w += num_of_Bcalc_in_warp;
        tmp.real = recv[0].x;          tmp.image = recv[0].y;
        tmp = _complex_fma(*((de::CPf*)&recv[2].x), de::CPf(-0.5, 0.8660254f), tmp);
        res.x = _complex_fma_preserve_R(*((de::CPf*)&recv[4].x), de::CPf(-0.5, -0.8660254f), tmp);
        tmp.real = recv[0].z;          tmp.image = recv[0].w;
        tmp = _complex_fma(*((de::CPf*)&recv[2].z), de::CPf(-0.5, 0.8660254f), tmp);
        res.y = _complex_fma_preserve_R(*((de::CPf*)&recv[4].z), de::CPf(-0.5, -0.8660254f), tmp);
        tmp.real = recv[1].x;          tmp.image = recv[1].y;
        tmp = _complex_fma(*((de::CPf*)&recv[3].x), de::CPf(-0.5, 0.8660254f), tmp);
        res.z = _complex_fma_preserve_R(*((de::CPf*)&recv[5].x), de::CPf(-0.5, -0.8660254f), tmp);
        tmp.real = recv[1].z;          tmp.image = recv[1].w;
        tmp = _complex_fma(*((de::CPf*)&recv[3].z), de::CPf(-0.5, 0.8660254f), tmp);
        res.w = _complex_fma_preserve_R(*((de::CPf*)&recv[5].z), de::CPf(-0.5, -0.8660254f), tmp);

        _fdividef_float4_;
        dst[dex_base + dex_store_w] = res;          // store

        // output 3
        dex_store_w += num_of_Bcalc_in_warp;
        tmp.real = recv[0].x;          tmp.image = recv[0].y;
        tmp = _complex_fma(*((de::CPf*)&recv[2].x), de::CPf(-0.5, -0.8660254f), tmp);
        res.x = _complex_fma_preserve_R(*((de::CPf*)&recv[4].x), de::CPf(-0.5, 0.8660254f), tmp);
        tmp.real = recv[0].z;          tmp.image = recv[0].w;
        tmp = _complex_fma(*((de::CPf*)&recv[2].z), de::CPf(-0.5, -0.8660254f), tmp);
        res.y = _complex_fma_preserve_R(*((de::CPf*)&recv[4].z), de::CPf(-0.5, 0.8660254f), tmp);
        tmp.real = recv[1].x;          tmp.image = recv[1].y;
        tmp = _complex_fma(*((de::CPf*)&recv[3].x), de::CPf(-0.5, -0.8660254f), tmp);
        res.z = _complex_fma_preserve_R(*((de::CPf*)&recv[5].x), de::CPf(-0.5, 0.8660254f), tmp);
        tmp.real = recv[1].z;          tmp.image = recv[1].w;
        tmp = _complex_fma(*((de::CPf*)&recv[3].z), de::CPf(-0.5, -0.8660254f), tmp);
        res.w = _complex_fma_preserve_R(*((de::CPf*)&recv[5].z), de::CPf(-0.5, 0.8660254f), tmp);

        _fdividef_float4_;
        dst[dex_base + dex_store_w] = res;      // store
    }
#undef _fdividef_float4_
}