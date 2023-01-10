/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "FFT2D_Radix_4_kernel.cuh"


__global__
void decx::signal::GPUK::cu_FFT2D_R4_R2C_first(const float* src,
                            float4* dst, 
                            const uint B_ops_num,
                            const uint pitchsrc,        // in float
                            const uint pitchdst,
                            const uint procH)        // in float4
{
    uint dex_w = threadIdx.y + blockDim.y * blockIdx.y;
    uint dex_h = threadIdx.x + blockDim.x * blockIdx.x;

    size_t dex_src = dex_h * pitchsrc + dex_w;
    size_t dex_base_dst = dex_h * pitchdst;

    float recv[4];
    de::CPf res[4] = { de::CPf(0, 0), de::CPf(0, 0),
                       de::CPf(0, 0), de::CPf(0, 0) };

    if (dex_h < procH && dex_w < B_ops_num) {
        recv[0] = src[dex_src];
        recv[1] = src[dex_src + B_ops_num];
        recv[2] = src[dex_src + B_ops_num * 2];
        recv[3] = src[dex_src + B_ops_num * 3];

        res[0].real = __fadd_rn(__fadd_rn(recv[0], recv[1]), __fadd_rn(recv[2], recv[3]));
        res[0].image = 0;

        res[1].real = __fsub_rn(recv[0], recv[2]);
        res[1].image = __fsub_rn(recv[1], recv[3]);

        res[2].real = __fadd_rn(__fsub_rn(recv[0], recv[1]), __fsub_rn(recv[2], recv[3]));
        res[2].image = 0;

        res[3].real = __fsub_rn(recv[0], recv[2]);
        res[3].image = __fsub_rn(recv[3], recv[1]);

        dst[dex_base_dst + dex_w * 2] = *((float4*)&res[0]);
        dst[dex_base_dst + dex_w * 2 + 1] = *((float4*)&res[2]);
    }
}



__global__
void decx::signal::GPUK::cu_IFFT2D_R4_C2C_first(const float2* src, 
                            float4* dst, 
                            const uint B_ops_num,
                            const uint pitch,       // in float4
                            const uint procH)
{
    uint dex_w = threadIdx.y + blockDim.y * blockIdx.y;
    uint dex_h = threadIdx.x + blockDim.x * blockIdx.x;

    size_t dex_base_dst = dex_h * pitch;

    de::CPf recv[4], tmp;
    de::CPf res[4] = { de::CPf(0, 0), de::CPf(0, 0),
                       de::CPf(0, 0), de::CPf(0, 0) };

    const float signal_len = (float)B_ops_num * 4;

    if (dex_h < procH && dex_w < B_ops_num) {
        *((float2*)&recv[0]) = src[dex_base_dst * 2 + dex_w];
        *((float2*)&recv[1]) = src[dex_base_dst * 2 + dex_w + B_ops_num];
        *((float2*)&recv[2]) = src[dex_base_dst * 2 + dex_w + B_ops_num * 2];
        *((float2*)&recv[3]) = src[dex_base_dst * 2 + dex_w + B_ops_num * 3];
        
        // conj and * /= signal_len
        recv[0].real = __fdividef(recv[0].real, signal_len);     recv[0].image = __fdividef(recv[0].image, -signal_len);
        recv[1].real = __fdividef(recv[1].real, signal_len);     recv[1].image = __fdividef(recv[1].image, -signal_len);
        recv[2].real = __fdividef(recv[2].real, signal_len);     recv[2].image = __fdividef(recv[2].image, -signal_len);
        recv[3].real = __fdividef(recv[3].real, signal_len);     recv[3].image = __fdividef(recv[3].image, -signal_len);

        res[0].real = recv[0].real + recv[1].real + recv[2].real + recv[3].real;
        res[0].image = recv[0].image + recv[1].image + recv[2].image + recv[3].image;

        tmp = recv[0];
        tmp.real -= recv[1].image;
        tmp.image += recv[1].real;
        tmp.real -= recv[2].real;
        tmp.image -= recv[2].image;
        res[1].real = tmp.real + recv[3].image;
        res[1].image = tmp.image - recv[3].real;

        tmp = recv[0];
        tmp.real -= recv[1].real;
        tmp.image -= recv[1].image;
        tmp.real += recv[2].real;
        tmp.image += recv[2].image;
        res[2].real = tmp.real - recv[3].real;
        res[2].image = tmp.image - recv[3].image;

        tmp = recv[0];
        tmp.real += recv[1].image;
        tmp.image -= recv[1].real;
        tmp.real -= recv[2].real;
        tmp.image -= recv[2].image;
        res[3].real = tmp.real - recv[3].image;
        res[3].image = tmp.image + recv[3].real;

        dst[dex_base_dst + dex_w * 2] = *((float4*)&res[0]);
        dst[dex_base_dst + dex_w * 2 + 1] = *((float4*)&res[2]);
    }
}



__global__
void decx::signal::GPUK::cu_IFFT2D_R4_C2R_once(const float2* src, 
                            float4* dst, 
                            const uint B_ops_num,
                            const uint pitch,       // in float4
                            const uint procH)
{
    uint dex_w = threadIdx.y + blockDim.y * blockIdx.y;
    uint dex_h = threadIdx.x + blockDim.x * blockIdx.x;

    size_t dex_base_dst = dex_h * pitch;

    de::CPf recv[4], tmp;
    float4 res;

    const float signal_len = (float)B_ops_num * 4;

    if (dex_h < procH && dex_w < B_ops_num) {
        *((float2*)&recv[0]) = src[dex_base_dst * 2 + dex_w];
        *((float2*)&recv[1]) = src[dex_base_dst * 2 + dex_w + B_ops_num];
        *((float2*)&recv[2]) = src[dex_base_dst * 2 + dex_w + B_ops_num * 2];
        *((float2*)&recv[3]) = src[dex_base_dst * 2 + dex_w + B_ops_num * 3];
        
        res.x = recv[0].real + recv[1].real + recv[2].real + recv[3].real;

        tmp = recv[0];
        tmp.real -= recv[1].image;
        tmp.image += recv[1].real;
        tmp.real -= recv[2].real;
        tmp.image -= recv[2].image;
        res.y = tmp.real + recv[3].image;

        tmp = recv[0];
        tmp.real -= recv[1].real;
        tmp.image -= recv[1].image;
        tmp.real += recv[2].real;
        tmp.image += recv[2].image;
        res.z = tmp.real - recv[3].real;

        tmp = recv[0];
        tmp.real += recv[1].image;
        tmp.image -= recv[1].real;
        tmp.real -= recv[2].real;
        tmp.image -= recv[2].image;
        res.w = tmp.real - recv[3].image;

        res.x = __fdividef(res.x, signal_len);
        res.y = __fdividef(res.y, signal_len);
        res.z = __fdividef(res.z, signal_len);
        res.w = __fdividef(res.w, signal_len);

        dst[dex_base_dst + dex_w] = res;
    }
}



__global__
void decx::signal::GPUK::cu_FFT2D_R4_C2C_first(const float2* src, 
                            float4* dst, 
                            const uint B_ops_num,
                            const uint pitch,       // in float4
                            const uint procH)
{
    uint dex_w = threadIdx.y + blockDim.y * blockIdx.y;
    uint dex_h = threadIdx.x + blockDim.x * blockIdx.x;

    size_t dex_base_dst = dex_h * pitch;

    de::CPf recv[4], tmp;
    de::CPf res[4] = { de::CPf(0, 0), de::CPf(0, 0),
                       de::CPf(0, 0), de::CPf(0, 0) };

    const float signal_len = (float)B_ops_num * 4;

    if (dex_h < procH && dex_w < B_ops_num) {
        *((float2*)&recv[0]) = src[dex_base_dst * 2 + dex_w];
        *((float2*)&recv[1]) = src[dex_base_dst * 2 + dex_w + B_ops_num];
        *((float2*)&recv[2]) = src[dex_base_dst * 2 + dex_w + B_ops_num * 2];
        *((float2*)&recv[3]) = src[dex_base_dst * 2 + dex_w + B_ops_num * 3];
        
        res[0].real = recv[0].real + recv[1].real + recv[2].real + recv[3].real;
        res[0].image = recv[0].image + recv[1].image + recv[2].image + recv[3].image;

        tmp = recv[0];
        tmp.real -= recv[1].image;
        tmp.image += recv[1].real;
        tmp.real -= recv[2].real;
        tmp.image -= recv[2].image;
        res[1].real = tmp.real + recv[3].image;
        res[1].image = tmp.image - recv[3].real;

        tmp = recv[0];
        tmp.real -= recv[1].real;
        tmp.image -= recv[1].image;
        tmp.real += recv[2].real;
        tmp.image += recv[2].image;
        res[2].real = tmp.real - recv[3].real;
        res[2].image = tmp.image - recv[3].image;

        tmp = recv[0];
        tmp.real += recv[1].image;
        tmp.image -= recv[1].real;
        tmp.real -= recv[2].real;
        tmp.image -= recv[2].image;
        res[3].real = tmp.real - recv[3].image;
        res[3].image = tmp.image + recv[3].real;

        dst[dex_base_dst + dex_w * 2] = *((float4*)&res[0]);
        dst[dex_base_dst + dex_w * 2 + 1] = *((float4*)&res[2]);
    }
}



__global__
void decx::signal::GPUK::cu_FFT2D_R4_R2C_first_vec4(const float4* src, 
                                float4* dst, 
                                const uint B_ops_num,
                                const uint pitch,
                                const uint procH)
{
    uint dex_w = threadIdx.y + blockDim.y * blockIdx.y;
    uint dex_h = threadIdx.x + blockDim.x * blockIdx.x;

    size_t dex_base = dex_h * pitch;

    float4 recv[4];
    de::CPf res[4];

    if (dex_h < procH && dex_w < B_ops_num) {
        recv[0] = src[dex_base + dex_w];
        recv[1] = src[dex_base + dex_w + B_ops_num];
        recv[2] = src[dex_base + dex_w + B_ops_num * 2];
        recv[3] = src[dex_base + dex_w + B_ops_num * 3];
        /*if (dex_w == 0 && dex_h == 0) {
            printf("%f\n", recv[0].x);
            printf("%f\n", recv[0].y);
            printf("%f\n", recv[0].z);
            printf("%f\n", recv[0].w);
        }*/

        res[0].real = __fadd_rn(__fadd_rn(recv[0].x, recv[1].x), __fadd_rn(recv[2].x, recv[3].x));
        res[0].image = 0;
        res[1].real = __fsub_rn(recv[0].x, recv[2].x);
        res[1].image = __fsub_rn(recv[1].x, recv[3].x);
        res[2].real = __fadd_rn(__fsub_rn(recv[0].x, recv[1].x), __fsub_rn(recv[2].x, recv[3].x));
        res[2].image = 0;
        res[3].real = __fsub_rn(recv[0].x, recv[2].x);
        res[3].image = __fsub_rn(recv[3].x, recv[1].x);
        dst[dex_base + dex_w * 8] = *((float4*)&res[0]);
        dst[dex_base + dex_w * 8 + 1] = *((float4*)&res[2]);

        res[0].real = __fadd_rn(__fadd_rn(recv[0].y, recv[1].y), __fadd_rn(recv[2].y, recv[3].y));
        res[0].image = 0;
        res[1].real = __fsub_rn(recv[0].y, recv[2].y);
        res[1].image = __fsub_rn(recv[1].y, recv[3].y);
        res[2].real = __fadd_rn(__fsub_rn(recv[0].y, recv[1].y), __fsub_rn(recv[2].y, recv[3].y));
        res[2].image = 0;
        res[3].real = __fsub_rn(recv[0].y, recv[2].y);
        res[3].image = __fsub_rn(recv[3].y, recv[1].y);
        dst[dex_base + dex_w * 8 + 2] = *((float4*)&res[0]);
        dst[dex_base + dex_w * 8 + 3] = *((float4*)&res[2]);

        res[0].real = __fadd_rn(__fadd_rn(recv[0].z, recv[1].z), __fadd_rn(recv[2].z, recv[3].z));
        res[0].image = 0;
        res[1].real = __fsub_rn(recv[0].z, recv[2].z);
        res[1].image = __fsub_rn(recv[1].z, recv[3].z);
        res[2].real = __fadd_rn(__fsub_rn(recv[0].z, recv[1].z), __fsub_rn(recv[2].z, recv[3].z));
        res[2].image = 0;
        res[3].real = __fsub_rn(recv[0].z, recv[2].z);
        res[3].image = __fsub_rn(recv[3].z, recv[1].z);
        dst[dex_base + dex_w * 8 + 4] = *((float4*)&res[0]);
        dst[dex_base + dex_w * 8 + 5] = *((float4*)&res[2]);

        res[0].real = __fadd_rn(__fadd_rn(recv[0].w, recv[1].w), __fadd_rn(recv[2].w, recv[3].w));
        res[0].image = 0;
        res[1].real = __fsub_rn(recv[0].w, recv[2].w);
        res[1].image = __fsub_rn(recv[1].w, recv[3].w);
        res[2].real = __fadd_rn(__fsub_rn(recv[0].w, recv[1].w), __fsub_rn(recv[2].w, recv[3].w));
        res[2].image = 0;
        res[3].real = __fsub_rn(recv[0].w, recv[2].w);
        res[3].image = __fsub_rn(recv[3].w, recv[1].w);
        dst[dex_base + dex_w * 8 + 6] = *((float4*)&res[0]);
        dst[dex_base + dex_w * 8 + 7] = *((float4*)&res[2]);
    }
}



__global__
void decx::signal::GPUK::cu_IFFT2D_R4_C2C_first_vec4(const float4* src, 
                                float4* dst, const 
                                uint B_ops_num,
                                const uint pitch,           // src's, dst's is tow times bigger than it
                                const uint procH)
{
    uint dex_w = threadIdx.y + blockDim.y * blockIdx.y;
    uint dex_h = threadIdx.x + blockDim.x * blockIdx.x;

    size_t dex_base_src = dex_h * pitch;

    de::CPf recv[16];
    float4 res[2];

    const float signal_len = (float)B_ops_num * 4 * 4;

    if (dex_h < procH && dex_w < B_ops_num) {
        *((float4*)&recv[0]) = src[dex_base_src + dex_w * 2];
        *((float4*)&recv[2]) = src[dex_base_src + dex_w * 2 + 1];
        *((float4*)&recv[4]) = src[dex_base_src + (dex_w + B_ops_num) * 2];
        *((float4*)&recv[6]) = src[dex_base_src + (dex_w + B_ops_num) * 2 + 1];
        *((float4*)&recv[8]) = src[dex_base_src + (dex_w + B_ops_num * 2) * 2];
        *((float4*)&recv[10]) = src[dex_base_src + (dex_w + B_ops_num * 2) * 2 + 1];
        *((float4*)&recv[12]) = src[dex_base_src + (dex_w + B_ops_num * 3) * 2];
        *((float4*)&recv[14]) = src[dex_base_src + (dex_w + B_ops_num * 3) * 2 + 1];

#pragma unroll 16
        for (int i = 0; i < 16; ++i) {
            recv[i].real = __fdividef(recv[i].real, signal_len);    
            recv[i].image = __fdividef(recv[i].image, -signal_len);
        }
#pragma unroll 4
        for (int k = 0; k < 4; ++k) {
            res[0].x = __fadd_rn(__fadd_rn(recv[k].real, recv[k + 4].real), __fadd_rn(recv[k + 8].real, recv[k + 12].real));
            res[0].y = __fadd_rn(__fadd_rn(recv[k].image, recv[k + 4].image), __fadd_rn(recv[k + 8].image, recv[k + 12].image));

            res[0].z = __fsub_rn(__fsub_rn(recv[k].real, recv[k + 4].image), __fsub_rn(recv[k + 8].real, recv[k + 12].image));
            res[0].w = __fsub_rn(__fadd_rn(recv[k].image, recv[k + 4].real), __fadd_rn(recv[k + 8].image, recv[k + 12].real));

            res[1].x = __fadd_rn(__fsub_rn(recv[k].real, recv[k + 4].real), __fsub_rn(recv[k + 8].real, recv[k + 12].real));
            res[1].y = __fadd_rn(__fsub_rn(recv[k].image, recv[k + 4].image), __fsub_rn(recv[k + 8].image, recv[k + 12].image));

            res[1].z = __fsub_rn(__fadd_rn(recv[k].real, recv[k + 4].image), __fadd_rn(recv[k + 8].real, recv[k + 12].image));
            res[1].w = __fsub_rn(__fsub_rn(recv[k].image, recv[k + 4].real), __fsub_rn(recv[k + 8].image, recv[k + 12].real));

            dst[dex_base_src + dex_w * 8 + 2 * k] = res[0];
            dst[dex_base_src + dex_w * 8 + 2 * k + 1] = res[1];
        }
    }
}



__global__
void decx::signal::GPUK::cu_FFT2D_R4_C2C_first_vec4(const float4* src, 
                                float4* dst, const 
                                uint B_ops_num,
                                const uint pitch,           // src's, dst's is tow times bigger than it
                                const uint procH)
{
    uint dex_w = threadIdx.y + blockDim.y * blockIdx.y;
    uint dex_h = threadIdx.x + blockDim.x * blockIdx.x;

    size_t dex_base_src = dex_h * pitch;

    de::CPf recv[16];
    float4 res[2];

    const float signal_len = (float)B_ops_num * 4 * 4;

    if (dex_h < procH && dex_w < B_ops_num) {
        *((float4*)&recv[0]) = src[dex_base_src + dex_w * 2];
        *((float4*)&recv[2]) = src[dex_base_src + dex_w * 2 + 1];
        *((float4*)&recv[4]) = src[dex_base_src + (dex_w + B_ops_num) * 2];
        *((float4*)&recv[6]) = src[dex_base_src + (dex_w + B_ops_num) * 2 + 1];
        *((float4*)&recv[8]) = src[dex_base_src + (dex_w + B_ops_num * 2) * 2];
        *((float4*)&recv[10]) = src[dex_base_src + (dex_w + B_ops_num * 2) * 2 + 1];
        *((float4*)&recv[12]) = src[dex_base_src + (dex_w + B_ops_num * 3) * 2];
        *((float4*)&recv[14]) = src[dex_base_src + (dex_w + B_ops_num * 3) * 2 + 1];

#pragma unroll 4
        for (int k = 0; k < 4; ++k) {
            res[0].x = __fadd_rn(__fadd_rn(recv[k].real, recv[k + 4].real), __fadd_rn(recv[k + 8].real, recv[k + 12].real));
            res[0].y = __fadd_rn(__fadd_rn(recv[k].image, recv[k + 4].image), __fadd_rn(recv[k + 8].image, recv[k + 12].image));

            res[0].z = __fsub_rn(__fsub_rn(recv[k].real, recv[k + 4].image), __fsub_rn(recv[k + 8].real, recv[k + 12].image));
            res[0].w = __fsub_rn(__fadd_rn(recv[k].image, recv[k + 4].real), __fadd_rn(recv[k + 8].image, recv[k + 12].real));

            res[1].x = __fadd_rn(__fsub_rn(recv[k].real, recv[k + 4].real), __fsub_rn(recv[k + 8].real, recv[k + 12].real));
            res[1].y = __fadd_rn(__fsub_rn(recv[k].image, recv[k + 4].image), __fsub_rn(recv[k + 8].image, recv[k + 12].image));

            res[1].z = __fsub_rn(__fadd_rn(recv[k].real, recv[k + 4].image), __fadd_rn(recv[k + 8].real, recv[k + 12].image));
            res[1].w = __fsub_rn(__fsub_rn(recv[k].image, recv[k + 4].real), __fsub_rn(recv[k + 8].image, recv[k + 12].real));

            dst[dex_base_src + dex_w * 8 + 2 * k] = res[0];
            dst[dex_base_src + dex_w * 8 + 2 * k + 1] = res[1];
        }
    }
}




__global__
void decx::signal::GPUK::cu_FFT2D_R4_C2C(const float2* src, 
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

    uint warp_loc_id, num_of_Bcalc_in_warp = warp_proc_len / 4;

    de::CPf recv[4], tmp;
    de::CPf W, res[4] = { de::CPf(0, 0), de::CPf(0, 0), de::CPf(0, 0) };

    if (dex_h < procH && dex_w < B_ops_num) {
        *((float2*)&recv[0]) = src[dex_base + dex_w];
        *((float2*)&recv[1]) = src[dex_base + dex_w + B_ops_num];
        *((float2*)&recv[2]) = src[dex_base + dex_w + B_ops_num * 2];
        *((float2*)&recv[3]) = src[dex_base + dex_w + B_ops_num * 3];

        warp_loc_id = dex_w % num_of_Bcalc_in_warp;
        W.dev_construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)warp_loc_id, (float)warp_proc_len)));
        recv[1] = _complex_mul(recv[1], W);
        W.dev_construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)(warp_loc_id * 2), (float)warp_proc_len)));
        recv[2] = _complex_mul(recv[2], W);
        W.dev_construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)(warp_loc_id * 3), (float)warp_proc_len)));
        recv[3] = _complex_mul(recv[3], W);

        res[0].real = recv[0].real + recv[1].real + recv[2].real + recv[3].real;
        res[0].image = recv[0].image + recv[1].image + recv[2].image + recv[3].image;

        tmp = recv[0];
        tmp.real -= recv[1].image;
        tmp.image += recv[1].real;
        tmp.real -= recv[2].real;
        tmp.image -= recv[2].image;
        res[1].real = tmp.real + recv[3].image;
        res[1].image = tmp.image - recv[3].real;

        tmp = recv[0];
        tmp.real -= recv[1].real;
        tmp.image -= recv[1].image;
        tmp.real += recv[2].real;
        tmp.image += recv[2].image;
        res[2].real = tmp.real - recv[3].real;
        res[2].image = tmp.image - recv[3].image;

        tmp = recv[0];
        tmp.real += recv[1].image;
        tmp.image -= recv[1].real;
        tmp.real -= recv[2].real;
        tmp.image -= recv[2].image;
        res[3].real = tmp.real - recv[3].image;
        res[3].image = tmp.image + recv[3].real;

        dex_store_w = (dex_w / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id;

        dst[dex_base + dex_store_w] = *((float2*)&res[0]);
        dst[dex_base + dex_store_w + num_of_Bcalc_in_warp] = *((float2*)&res[1]);
        dst[dex_base + dex_store_w + num_of_Bcalc_in_warp * 2] = *((float2*)&res[2]);
        dst[dex_base + dex_store_w + num_of_Bcalc_in_warp * 3] = *((float2*)&res[3]);
    }
}



__global__
void decx::signal::GPUK::cu_IFFT2D_R4_C2R_last(const float2* src, 
                     float* dst, 
                     const uint B_ops_num, 
                     const uint warp_proc_len,
                     const uint pitch,              // in float2
                     const uint procH)
{
    using namespace decx::signal::cuda::dev;
    uint dex_w = threadIdx.y + blockDim.y * blockIdx.y;
    uint dex_h = threadIdx.x + blockDim.x * blockIdx.x;
    uint dex_store_w;

    size_t dex_base = dex_h * pitch;

    uint warp_loc_id, num_of_Bcalc_in_warp = warp_proc_len / 4;

    de::CPf recv[4], tmp;
    de::CPf W;
    float res[4];
    const float signal_len = (float)B_ops_num * 4;

    if (dex_h < procH && dex_w < B_ops_num) {
        *((float2*)&recv[0]) = src[dex_base + dex_w];
        *((float2*)&recv[1]) = src[dex_base + dex_w + B_ops_num];
        *((float2*)&recv[2]) = src[dex_base + dex_w + B_ops_num * 2];
        *((float2*)&recv[3]) = src[dex_base + dex_w + B_ops_num * 3];

        warp_loc_id = dex_w % num_of_Bcalc_in_warp;
        W.dev_construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)warp_loc_id, (float)warp_proc_len)));
        recv[1] = _complex_mul(recv[1], W);
        W.dev_construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)(warp_loc_id * 2), (float)warp_proc_len)));
        recv[2] = _complex_mul(recv[2], W);
        W.dev_construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)(warp_loc_id * 3), (float)warp_proc_len)));
        recv[3] = _complex_mul(recv[3], W);

        res[0] = recv[0].real + recv[1].real + recv[2].real + recv[3].real;

        /*tmp = recv[0];
        tmp.real -= recv[1].image;
        tmp.image += recv[1].real;
        tmp.real -= recv[2].real;
        tmp.image -= recv[2].image;
        res[1] = tmp.real + recv[3].image;

        tmp = recv[0];
        tmp.real -= recv[1].real;
        tmp.image -= recv[1].image;
        tmp.real += recv[2].real;
        tmp.image += recv[2].image;
        res[2] = tmp.real - recv[3].real;

        tmp = recv[0];
        tmp.real += recv[1].image;
        tmp.image -= recv[1].real;
        tmp.real -= recv[2].real;
        tmp.image -= recv[2].image;
        res[3] = tmp.real - recv[3].image;*/

        res[1] = __fsub_rn(__fadd_rn(recv[0].real, recv[3].image), __fadd_rn(recv[1].image, recv[2].real));

        res[2] = __fadd_rn(__fsub_rn(recv[0].real, recv[1].real), __fsub_rn(recv[2].real, recv[3].real));

        res[3] = __fsub_rn(__fadd_rn(recv[0].real, recv[1].image), __fadd_rn(recv[2].real, recv[3].image));

        dex_store_w = (dex_w / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id;

        res[0] = __fdividef(res[0], signal_len);
        res[1] = __fdividef(res[1], signal_len);
        res[2] = __fdividef(res[2], signal_len);
        res[3] = __fdividef(res[3], signal_len);

        dst[dex_base * 2 + dex_store_w] = res[0];
        dst[dex_base * 2 + dex_store_w + num_of_Bcalc_in_warp] = res[1];
        dst[dex_base * 2 + dex_store_w + num_of_Bcalc_in_warp * 2] = res[2];
        dst[dex_base * 2 + dex_store_w + num_of_Bcalc_in_warp * 3] = res[3];
    }
}



__global__
/*
* @param B_ops_num : in Vec4
* @param warp_proc_len : element
* @param pitch : in float4
*/
void decx::signal::GPUK::cu_FFT2D_R4_C2C_vec4(const float4* src, 
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

    uint warp_loc_id, num_of_Bcalc_in_warp = warp_proc_len / 4 / 4;

    de::CPf W, tmp[4];
    float4 recv[8], res;

    if (dex_h < procH && dex_w < B_ops_num) {
        recv[0] = src[dex_base + dex_w * 2];
        recv[1] = src[dex_base + dex_w * 2 + 1];
        recv[2] = src[dex_base + (dex_w + B_ops_num) * 2];
        recv[3] = src[dex_base + (dex_w + B_ops_num) * 2 + 1];
        recv[4] = src[dex_base + (dex_w + B_ops_num * 2) * 2];
        recv[5] = src[dex_base + (dex_w + B_ops_num * 2) * 2 + 1];
        recv[6] = src[dex_base + (dex_w + B_ops_num * 3) * 2];
        recv[7] = src[dex_base + (dex_w + B_ops_num * 3) * 2 + 1];

        warp_loc_id = dex_w % (size_t)num_of_Bcalc_in_warp;
        dex_store_w = (dex_w / (size_t)num_of_Bcalc_in_warp) * (size_t)warp_proc_len / 4 + warp_loc_id;

        W.dev_construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)(warp_loc_id * 4), (float)warp_proc_len)));
        *((de::CPf*)&recv[2].x) = _complex_mul(*((de::CPf*)&recv[2].x), W);
        W.dev_construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)(warp_loc_id * 4 + 1), (float)warp_proc_len)));
        *((de::CPf*)&recv[2].z) = _complex_mul(*((de::CPf*)&recv[2].z), W);
        W.dev_construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)(warp_loc_id * 4 + 2), (float)warp_proc_len)));
        *((de::CPf*)&recv[3].x) = _complex_mul(*((de::CPf*)&recv[3].x), W);
        W.dev_construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)(warp_loc_id * 4 + 3), (float)warp_proc_len)));
        *((de::CPf*)&recv[3].z) = _complex_mul(*((de::CPf*)&recv[3].z), W);

        W.dev_construct_with_phase(__fmul_rn(Four_Pi, __fdividef((float)(warp_loc_id * 4), (float)warp_proc_len)));
        *((de::CPf*)&recv[4].x) = _complex_mul(*((de::CPf*)&recv[4].x), W);
        W.dev_construct_with_phase(__fmul_rn(Four_Pi, __fdividef((float)(warp_loc_id * 4 + 1), (float)warp_proc_len)));
        *((de::CPf*)&recv[4].z) = _complex_mul(*((de::CPf*)&recv[4].z), W);
        W.dev_construct_with_phase(__fmul_rn(Four_Pi, __fdividef((float)(warp_loc_id * 4 + 2), (float)warp_proc_len)));
        *((de::CPf*)&recv[5].x) = _complex_mul(*((de::CPf*)&recv[5].x), W);
        W.dev_construct_with_phase(__fmul_rn(Four_Pi, __fdividef((float)(warp_loc_id * 4 + 3), (float)warp_proc_len)));
        *((de::CPf*)&recv[5].z) = _complex_mul(*((de::CPf*)&recv[5].z), W);

        W.dev_construct_with_phase(__fmul_rn(Six_Pi, __fdividef((float)(warp_loc_id * 4), (float)warp_proc_len)));
        *((de::CPf*)&recv[6].x) = _complex_mul(*((de::CPf*)&recv[6].x), W);
        W.dev_construct_with_phase(__fmul_rn(Six_Pi, __fdividef((float)(warp_loc_id * 4 + 1), (float)warp_proc_len)));
        *((de::CPf*)&recv[6].z) = _complex_mul(*((de::CPf*)&recv[6].z), W);
        W.dev_construct_with_phase(__fmul_rn(Six_Pi, __fdividef((float)(warp_loc_id * 4 + 2), (float)warp_proc_len)));
        *((de::CPf*)&recv[7].x) = _complex_mul(*((de::CPf*)&recv[7].x), W);
        W.dev_construct_with_phase(__fmul_rn(Six_Pi, __fdividef((float)(warp_loc_id * 4 + 3), (float)warp_proc_len)));
        *((de::CPf*)&recv[7].z) = _complex_mul(*((de::CPf*)&recv[7].z), W);

        // output 1
        res.x = recv[0].x + recv[2].x + recv[4].x + recv[6].x;
        res.y = recv[0].y + recv[2].y + recv[4].y + recv[6].y;
        res.z = recv[0].z + recv[2].z + recv[4].z + recv[6].z;
        res.w = recv[0].w + recv[2].w + recv[4].w + recv[6].w;
        dst[dex_base + dex_store_w * 2] = res;

        res.x = recv[1].x + recv[3].x + recv[5].x + recv[7].x;
        res.y = recv[1].y + recv[3].y + recv[5].y + recv[7].y;
        res.z = recv[1].z + recv[3].z + recv[5].z + recv[7].z;
        res.w = recv[1].w + recv[3].w + recv[5].w + recv[7].w;
        dst[dex_base + dex_store_w * 2 + 1] = res;

        // output 2
        dex_store_w += num_of_Bcalc_in_warp;
        res.x = recv[0].x;          res.y = recv[0].y;
        res.z = recv[0].z;          res.w = recv[0].w;
        res.x -= recv[2].y;         res.z -= recv[2].w;
        res.y += recv[2].x;         res.w += recv[2].z;
        res.x -= recv[4].x;         res.z -= recv[4].z;
        res.y -= recv[4].y;         res.w -= recv[4].w;
        res.x += recv[6].y;         res.z += recv[6].w;
        res.y -= recv[6].x;         res.w -= recv[6].z;
        dst[dex_base + dex_store_w * 2] = res;

        res.x = recv[1].x;          res.y = recv[1].y;
        res.z = recv[1].z;          res.w = recv[1].w;
        res.x -= recv[3].y;         res.z -= recv[3].w;
        res.y += recv[3].x;         res.w += recv[3].z;
        res.x -= recv[5].x;         res.z -= recv[5].z;
        res.y -= recv[5].y;         res.w -= recv[5].w;
        res.x += recv[7].y;         res.z += recv[7].w;
        res.y -= recv[7].x;         res.w -= recv[7].z;
        dst[dex_base + dex_store_w * 2 + 1] = res;

        // output 3
        dex_store_w += num_of_Bcalc_in_warp;
        res.x = recv[0].x;          res.y = recv[0].y;
        res.z = recv[0].z;          res.w = recv[0].w;
        res.x -= recv[2].x;         res.z -= recv[2].z;
        res.y -= recv[2].y;         res.w -= recv[2].w;
        res.x += recv[4].x;         res.z += recv[4].z;
        res.y += recv[4].y;         res.w += recv[4].w;
        res.x -= recv[6].x;         res.z -= recv[6].z;
        res.y -= recv[6].y;         res.w -= recv[6].w;
        dst[dex_base + dex_store_w * 2] = res;

        res.x = recv[1].x;          res.y = recv[1].y;
        res.z = recv[1].z;          res.w = recv[1].w;
        res.x -= recv[3].x;         res.z -= recv[3].z;
        res.y -= recv[3].y;         res.w -= recv[3].w;
        res.x += recv[5].x;         res.z += recv[5].z;
        res.y += recv[5].y;         res.w += recv[5].w;
        res.x -= recv[7].x;         res.z -= recv[7].z;
        res.y -= recv[7].y;         res.w -= recv[7].w;
        dst[dex_base + dex_store_w * 2 + 1] = res;

        // output 4
        dex_store_w += num_of_Bcalc_in_warp;
        res.x = recv[0].x;          res.y = recv[0].y;
        res.z = recv[0].z;          res.w = recv[0].w;
        res.x += recv[2].y;         res.z += recv[2].w;
        res.y -= recv[2].x;         res.w -= recv[2].z;
        res.x -= recv[4].x;         res.z -= recv[4].z;
        res.y -= recv[4].y;         res.w -= recv[4].w;
        res.x -= recv[6].y;         res.z -= recv[6].w;
        res.y += recv[6].x;         res.w += recv[6].z;
        dst[dex_base + dex_store_w * 2] = res;

        res.x = recv[1].x;          res.y = recv[1].y;
        res.z = recv[1].z;          res.w = recv[1].w;
        res.x += recv[3].y;         res.z += recv[3].w;
        res.y -= recv[3].x;         res.w -= recv[3].z;
        res.x -= recv[5].x;         res.z -= recv[5].z;
        res.y -= recv[5].y;         res.w -= recv[5].w;
        res.x -= recv[7].y;         res.z -= recv[7].w;
        res.y += recv[7].x;         res.w += recv[7].z;
        dst[dex_base + dex_store_w * 2 + 1] = res;
    }
}



__global__
/*
* @param B_ops_num : in Vec4
* @param warp_proc_len : element
* @param pitch : in float4
*/
void decx::signal::GPUK::cu_IFFT2D_R4_C2R_vec4_last(const float4* src, 
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

    uint warp_loc_id, num_of_Bcalc_in_warp = warp_proc_len / 4 / 4;

    de::CPf W, tmp[4];
    float4 recv[8], res;
    const float signal_len = (float)B_ops_num * 4 * 4;

    if (dex_h < procH && dex_w < B_ops_num) {
        recv[0] = src[dex_base + dex_w * 2];
        recv[1] = src[dex_base + dex_w * 2 + 1];
        recv[2] = src[dex_base + (dex_w + B_ops_num) * 2];
        recv[3] = src[dex_base + (dex_w + B_ops_num) * 2 + 1];
        recv[4] = src[dex_base + (dex_w + B_ops_num * 2) * 2];
        recv[5] = src[dex_base + (dex_w + B_ops_num * 2) * 2 + 1];
        recv[6] = src[dex_base + (dex_w + B_ops_num * 3) * 2];
        recv[7] = src[dex_base + (dex_w + B_ops_num * 3) * 2 + 1];

        warp_loc_id = dex_w % (size_t)num_of_Bcalc_in_warp;
        dex_store_w = (dex_w / (size_t)num_of_Bcalc_in_warp) * (size_t)warp_proc_len / 4 + warp_loc_id;

        W.dev_construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)(warp_loc_id * 4), (float)warp_proc_len)));
        *((de::CPf*)&recv[2].x) = _complex_mul(*((de::CPf*)&recv[2].x), W);
        W.dev_construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)(warp_loc_id * 4 + 1), (float)warp_proc_len)));
        *((de::CPf*)&recv[2].z) = _complex_mul(*((de::CPf*)&recv[2].z), W);
        W.dev_construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)(warp_loc_id * 4 + 2), (float)warp_proc_len)));
        *((de::CPf*)&recv[3].x) = _complex_mul(*((de::CPf*)&recv[3].x), W);
        W.dev_construct_with_phase(__fmul_rn(Two_Pi, __fdividef((float)(warp_loc_id * 4 + 3), (float)warp_proc_len)));
        *((de::CPf*)&recv[3].z) = _complex_mul(*((de::CPf*)&recv[3].z), W);

        W.dev_construct_with_phase(__fmul_rn(Four_Pi, __fdividef((float)(warp_loc_id * 4), (float)warp_proc_len)));
        *((de::CPf*)&recv[4].x) = _complex_mul(*((de::CPf*)&recv[4].x), W);
        W.dev_construct_with_phase(__fmul_rn(Four_Pi, __fdividef((float)(warp_loc_id * 4 + 1), (float)warp_proc_len)));
        *((de::CPf*)&recv[4].z) = _complex_mul(*((de::CPf*)&recv[4].z), W);
        W.dev_construct_with_phase(__fmul_rn(Four_Pi, __fdividef((float)(warp_loc_id * 4 + 2), (float)warp_proc_len)));
        *((de::CPf*)&recv[5].x) = _complex_mul(*((de::CPf*)&recv[5].x), W);
        W.dev_construct_with_phase(__fmul_rn(Four_Pi, __fdividef((float)(warp_loc_id * 4 + 3), (float)warp_proc_len)));
        *((de::CPf*)&recv[5].z) = _complex_mul(*((de::CPf*)&recv[5].z), W);

        W.dev_construct_with_phase(__fmul_rn(Six_Pi, __fdividef((float)(warp_loc_id * 4), (float)warp_proc_len)));
        *((de::CPf*)&recv[6].x) = _complex_mul(*((de::CPf*)&recv[6].x), W);
        W.dev_construct_with_phase(__fmul_rn(Six_Pi, __fdividef((float)(warp_loc_id * 4 + 1), (float)warp_proc_len)));
        *((de::CPf*)&recv[6].z) = _complex_mul(*((de::CPf*)&recv[6].z), W);
        W.dev_construct_with_phase(__fmul_rn(Six_Pi, __fdividef((float)(warp_loc_id * 4 + 2), (float)warp_proc_len)));
        *((de::CPf*)&recv[7].x) = _complex_mul(*((de::CPf*)&recv[7].x), W);
        W.dev_construct_with_phase(__fmul_rn(Six_Pi, __fdividef((float)(warp_loc_id * 4 + 3), (float)warp_proc_len)));
        *((de::CPf*)&recv[7].z) = _complex_mul(*((de::CPf*)&recv[7].z), W);

        // output 1
        res.x = recv[0].x + recv[2].x + recv[4].x + recv[6].x;
        res.y = recv[0].z + recv[2].z + recv[4].z + recv[6].z;
        res.z = recv[1].x + recv[3].x + recv[5].x + recv[7].x;
        res.w = recv[1].z + recv[3].z + recv[5].z + recv[7].z;
        res.x = __fdividef(res.x, signal_len);      res.y = __fdividef(res.y, signal_len);
        res.z = __fdividef(res.z, signal_len);      res.w = __fdividef(res.w, signal_len);
        dst[dex_base + dex_store_w] = res;

        // output 2
        dex_store_w += num_of_Bcalc_in_warp;
        res.x = recv[0].x;          res.y = recv[0].z;   
        res.z = recv[1].x;          res.w = recv[1].z;
        res.x -= recv[2].y;         res.y -= recv[2].w;
        res.x -= recv[4].x;         res.y -= recv[4].z;
        res.x += recv[6].y;         res.y += recv[6].w;
        res.z -= recv[3].y;         res.w -= recv[3].w;
        res.z -= recv[5].x;         res.w -= recv[5].z;
        res.z += recv[7].y;         res.w += recv[7].w;
        res.x = __fdividef(res.x, signal_len);      res.y = __fdividef(res.y, signal_len);
        res.z = __fdividef(res.z, signal_len);      res.w = __fdividef(res.w, signal_len);
        dst[dex_base + dex_store_w] = res;

        // output 3
        dex_store_w += num_of_Bcalc_in_warp;
        res.x = recv[0].x;          res.z = recv[1].x;
        res.y = recv[0].z;          res.w = recv[1].z;
        res.x -= recv[2].x;         res.y -= recv[2].z;
        res.x += recv[4].x;         res.y += recv[4].z;
        res.x -= recv[6].x;         res.y -= recv[6].z;
        res.z -= recv[3].x;         res.w -= recv[3].z;
        res.z += recv[5].x;         res.w += recv[5].z;
        res.z -= recv[7].x;         res.w -= recv[7].z;
        res.x = __fdividef(res.x, signal_len);      res.y = __fdividef(res.y, signal_len);
        res.z = __fdividef(res.z, signal_len);      res.w = __fdividef(res.w, signal_len);
        dst[dex_base + dex_store_w] = res;

        // output 4
        dex_store_w += num_of_Bcalc_in_warp;
        res.x = recv[0].x;          res.z = recv[1].x;
        res.y = recv[0].z;          res.w = recv[1].z;
        res.x += recv[2].y;         res.y += recv[2].w;
        res.x -= recv[4].x;         res.y -= recv[4].z;
        res.x -= recv[6].y;         res.y -= recv[6].w;
        res.z += recv[3].y;         res.w += recv[3].w;
        res.z -= recv[5].x;         res.w -= recv[5].z;
        res.z -= recv[7].y;         res.w -= recv[7].w;
        res.x = __fdividef(res.x, signal_len);      res.y = __fdividef(res.y, signal_len);
        res.z = __fdividef(res.z, signal_len);      res.w = __fdividef(res.w, signal_len);
        dst[dex_base + dex_store_w] = res;
    }
}