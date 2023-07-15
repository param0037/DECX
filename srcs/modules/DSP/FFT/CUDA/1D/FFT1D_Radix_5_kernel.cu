/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "FFT1D_Radix_5_kernel.cuh"


__global__
void decx::signal::GPUK::cu_FFT1D_R5_R2C_first(const float* __restrict src, float2* __restrict dst, const size_t B_ops_num)
{
    size_t dex = threadIdx.x + blockDim.x * blockIdx.x;

    float recv[5];
    de::CPf res[5];

    if (dex < B_ops_num) {
        recv[0] = src[dex];
        recv[1] = src[dex + B_ops_num];
        recv[2] = src[dex + B_ops_num * 2];
        recv[3] = src[dex + B_ops_num * 3];
        recv[4] = src[dex + B_ops_num * 4];

        res[0].real = recv[0] + recv[1] + recv[2] + recv[3] + recv[4];
        res[0].image = 0;

        res[1].real = recv[0] + recv[1] * 0.309017;
        res[1].image = recv[1] * 0.9510565;
        res[1].real += -0.809017 * recv[2];
        res[1].image += 0.5877853 * recv[2];
        res[1].real += -0.809017 * recv[3];
        res[1].image += -0.5877853 * recv[3];
        res[1].real += 0.309017 * recv[4];
        res[1].image += -0.9510565 * recv[4];

        res[2].real = recv[0] + recv[1] * -0.809017;
        res[2].image = recv[1] * 0.5877853;
        res[2].real += 0.309017 * recv[2];
        res[2].image += -0.9510565 * recv[2];
        res[2].real += 0.309017 * recv[3];
        res[2].image += 0.9510565 * recv[3];
        res[2].real += -0.809017 * recv[4];
        res[2].image += -0.5877853 * recv[4];

        res[3].real = recv[0] + recv[1] * -0.809017;
        res[3].image = recv[1] * -0.5877853;
        res[3].real += 0.309017 * recv[2];
        res[3].image += 0.9510565 * recv[2];
        res[3].real += 0.309017 * recv[3];
        res[3].image += -0.9510565 * recv[3];
        res[3].real += -0.809017 * recv[4];
        res[3].image += 0.5877853 * recv[4];

        res[4].real = recv[0] + recv[1] * 0.309017;
        res[4].image = recv[1] * -0.9510565;
        res[4].real += -0.809017 * recv[2];
        res[4].image += -0.5877853 * recv[2];
        res[4].real += -0.809017 * recv[3];
        res[4].image += 0.5877853 * recv[3];
        res[4].real += 0.309017 * recv[4];
        res[4].image += 0.9510565 * recv[4];

        dst[dex * 5] = *((float2*)&res[0]);
        dst[dex * 5 + 1] = *((float2*)&res[1]);
        dst[dex * 5 + 2] = *((float2*)&res[2]);
        dst[dex * 5 + 3] = *((float2*)&res[3]);
        dst[dex * 5 + 4] = *((float2*)&res[4]);
    }
}



__global__
void decx::signal::GPUK::cu_IFFT1D_R5_C2C_first(const float2* __restrict src, float2* __restrict dst, const size_t B_ops_num)
{
    using namespace decx::signal::cuda::dev;
    size_t dex = threadIdx.x + blockDim.x * blockIdx.x;

    de::CPf recv[5], tmp;
    de::CPf res[5];

    const float signal_len = (float)B_ops_num * 5;

    if (dex < B_ops_num) {
        *((float2*)&recv[0]) = src[dex];
        *((float2*)&recv[1]) = src[dex + B_ops_num];
        *((float2*)&recv[2]) = src[dex + B_ops_num * 2];
        *((float2*)&recv[3]) = src[dex + B_ops_num * 3];
        *((float2*)&recv[4]) = src[dex + B_ops_num * 4];

#pragma unroll 5
        for (int i = 0; i < 5; ++i) {
            recv[i].real = __fdividef(recv[i].real, signal_len);
            recv[i].image = __fdividef(recv[i].image, -signal_len);
        }

        res[0].real = recv[0].real + recv[1].real + recv[2].real + recv[3].real + recv[4].real;
        res[0].image = recv[0].image + recv[1].image + recv[2].image + recv[3].image + recv[4].image;

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(0.309017, 0.9510565), tmp);
        tmp = _complex_fma(recv[2], de::CPf(-0.809017, 0.5877853), tmp);
        tmp = _complex_fma(recv[3], de::CPf(-0.809017, -0.5877853), tmp);
        res[1] = _complex_fma(recv[4], de::CPf(0.309017, -0.9510565), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(-0.809017, 0.5877853), tmp);
        tmp = _complex_fma(recv[2], de::CPf(0.309017, -0.9510565), tmp);
        tmp = _complex_fma(recv[3], de::CPf(0.309017, 0.9510565), tmp);
        res[2] = _complex_fma(recv[4], de::CPf(-0.809017, -0.5877853), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(-0.809017, -0.5877853), tmp);
        tmp = _complex_fma(recv[2], de::CPf(0.309017, 0.9510565), tmp);
        tmp = _complex_fma(recv[3], de::CPf(0.309017, -0.9510565), tmp);
        res[3] = _complex_fma(recv[4], de::CPf(-0.809017, 0.5877853), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(0.309017, -0.9510565), tmp);
        tmp = _complex_fma(recv[2], de::CPf(-0.809017, -0.5877853), tmp);
        tmp = _complex_fma(recv[3], de::CPf(-0.809017, 0.5877853), tmp);
        res[4] = _complex_fma(recv[4], de::CPf(0.309017, 0.9510565), tmp);

        dst[dex * 5] = *((float2*)&res[0]);
        dst[dex * 5 + 1] = *((float2*)&res[1]);
        dst[dex * 5 + 2] = *((float2*)&res[2]);
        dst[dex * 5 + 3] = *((float2*)&res[3]);
        dst[dex * 5 + 4] = *((float2*)&res[4]);
    }
}


__global__
void decx::signal::GPUK::cu_FFT1D_R5_C2C_first(const float2* __restrict src, float2* __restrict dst, const size_t B_ops_num)
{
    using namespace decx::signal::cuda::dev;
    size_t dex = threadIdx.x + blockDim.x * blockIdx.x;

    de::CPf recv[5], tmp;
    de::CPf res[5];

    const float signal_len = (float)B_ops_num * 5;

    if (dex < B_ops_num) {
        *((float2*)&recv[0]) = src[dex];
        *((float2*)&recv[1]) = src[dex + B_ops_num];
        *((float2*)&recv[2]) = src[dex + B_ops_num * 2];
        *((float2*)&recv[3]) = src[dex + B_ops_num * 3];
        *((float2*)&recv[4]) = src[dex + B_ops_num * 4];

        res[0].real = recv[0].real + recv[1].real + recv[2].real + recv[3].real + recv[4].real;
        res[0].image = recv[0].image + recv[1].image + recv[2].image + recv[3].image + recv[4].image;

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(0.309017, 0.9510565), tmp);
        tmp = _complex_fma(recv[2], de::CPf(-0.809017, 0.5877853), tmp);
        tmp = _complex_fma(recv[3], de::CPf(-0.809017, -0.5877853), tmp);
        res[1] = _complex_fma(recv[4], de::CPf(0.309017, -0.9510565), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(-0.809017, 0.5877853), tmp);
        tmp = _complex_fma(recv[2], de::CPf(0.309017, -0.9510565), tmp);
        tmp = _complex_fma(recv[3], de::CPf(0.309017, 0.9510565), tmp);
        res[2] = _complex_fma(recv[4], de::CPf(-0.809017, -0.5877853), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(-0.809017, -0.5877853), tmp);
        tmp = _complex_fma(recv[2], de::CPf(0.309017, 0.9510565), tmp);
        tmp = _complex_fma(recv[3], de::CPf(0.309017, -0.9510565), tmp);
        res[3] = _complex_fma(recv[4], de::CPf(-0.809017, 0.5877853), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(0.309017, -0.9510565), tmp);
        tmp = _complex_fma(recv[2], de::CPf(-0.809017, -0.5877853), tmp);
        tmp = _complex_fma(recv[3], de::CPf(-0.809017, 0.5877853), tmp);
        res[4] = _complex_fma(recv[4], de::CPf(0.309017, 0.9510565), tmp);

        dst[dex * 5] = *((float2*)&res[0]);
        dst[dex * 5 + 1] = *((float2*)&res[1]);
        dst[dex * 5 + 2] = *((float2*)&res[2]);
        dst[dex * 5 + 3] = *((float2*)&res[3]);
        dst[dex * 5 + 4] = *((float2*)&res[4]);
    }
}



__global__
void decx::signal::GPUK::cu_IFFT1D_R5_C2R_once(const float2* __restrict src, float* __restrict dst, const size_t B_ops_num)
{
    using namespace decx::signal::cuda::dev;
    size_t dex = threadIdx.x + blockDim.x * blockIdx.x;

    de::CPf recv[5], tmp;
    float res[5];

    const float signal_len = (float)B_ops_num * 5;

    if (dex < B_ops_num) {
        *((float2*)&recv[0]) = src[dex];
        *((float2*)&recv[1]) = src[dex + B_ops_num];
        *((float2*)&recv[2]) = src[dex + B_ops_num * 2];
        *((float2*)&recv[3]) = src[dex + B_ops_num * 3];
        *((float2*)&recv[4]) = src[dex + B_ops_num * 4];

#pragma unroll 5
        for (int i = 0; i < 5; ++i) {
            recv[i].real = __fdividef(recv[i].real, signal_len);
            recv[i].image = __fdividef(recv[i].image, -signal_len);
        }

        res[0] = recv[0].real + recv[1].real + recv[2].real + recv[3].real + recv[4].real;

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(0.309017, 0.9510565), tmp);
        tmp = _complex_fma(recv[2], de::CPf(-0.809017, 0.5877853), tmp);
        tmp = _complex_fma(recv[3], de::CPf(-0.809017, -0.5877853), tmp);
        res[1] = _complex_fma_preserve_R(recv[4], de::CPf(0.309017, -0.9510565), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(-0.809017, 0.5877853), tmp);
        tmp = _complex_fma(recv[2], de::CPf(0.309017, -0.9510565), tmp);
        tmp = _complex_fma(recv[3], de::CPf(0.309017, 0.9510565), tmp);
        res[2] = _complex_fma_preserve_R(recv[4], de::CPf(-0.809017, -0.5877853), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(-0.809017, -0.5877853), tmp);
        tmp = _complex_fma(recv[2], de::CPf(0.309017, 0.9510565), tmp);
        tmp = _complex_fma(recv[3], de::CPf(0.309017, -0.9510565), tmp);
        res[3] = _complex_fma_preserve_R(recv[4], de::CPf(-0.809017, 0.5877853), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(0.309017, -0.9510565), tmp);
        tmp = _complex_fma(recv[2], de::CPf(-0.809017, -0.5877853), tmp);
        tmp = _complex_fma(recv[3], de::CPf(-0.809017, 0.5877853), tmp);
        res[4] = _complex_fma_preserve_R(recv[4], de::CPf(0.309017, 0.9510565), tmp);

        dst[dex * 5] = res[0];
        dst[dex * 5 + 1] = res[1];
        dst[dex * 5 + 2] = res[2];
        dst[dex * 5 + 3] = res[3];
        dst[dex * 5 + 4] = res[4];
    }
}



__global__
void decx::signal::GPUK::cu_FFT1D_R5_C2C(const float2* src, float2* dst, const size_t B_ops_num, const uint warp_proc_len)
{
    using namespace decx::signal::cuda::dev;
    size_t dex = threadIdx.x + blockDim.x * blockIdx.x;
    size_t dex_store;

    uint warp_loc_id, num_of_Bcalc_in_warp = warp_proc_len / 5;

    de::CPf recv[5], tmp;
    de::CPf W, res[5];

    if (dex < B_ops_num) {
        *((float2*)&recv[0]) = src[dex];
        *((float2*)&recv[1]) = src[dex + B_ops_num];
        *((float2*)&recv[2]) = src[dex + B_ops_num * 2];
        *((float2*)&recv[3]) = src[dex + B_ops_num * 3];
        *((float2*)&recv[4]) = src[dex + B_ops_num * 4];

        warp_loc_id = dex % num_of_Bcalc_in_warp;
        W.dev_construct_with_phase(Two_Pi * (float)warp_loc_id / (float)warp_proc_len);
        recv[1] = _complex_mul(recv[1], W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 2) / (float)warp_proc_len);
        recv[2] = _complex_mul(recv[2], W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 3) / (float)warp_proc_len);
        recv[3] = _complex_mul(recv[3], W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4) / (float)warp_proc_len);
        recv[4] = _complex_mul(recv[4], W);

        res[0].real = recv[0].real + recv[1].real + recv[2].real + recv[3].real + recv[4].real;
        res[0].image = recv[0].image + recv[1].image + recv[2].image + recv[3].image + recv[4].image;

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(0.309017, 0.9510565), tmp);
        tmp = _complex_fma(recv[2], de::CPf(-0.809017, 0.5877853), tmp);
        tmp = _complex_fma(recv[3], de::CPf(-0.809017, -0.5877853), tmp);
        res[1] = _complex_fma(recv[4], de::CPf(0.309017, -0.9510565), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(-0.809017, 0.5877853), tmp);
        tmp = _complex_fma(recv[2], de::CPf(0.309017, -0.9510565), tmp);
        tmp = _complex_fma(recv[3], de::CPf(0.309017, 0.9510565), tmp);
        res[2] = _complex_fma(recv[4], de::CPf(-0.809017, -0.5877853), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(-0.809017, -0.5877853), tmp);
        tmp = _complex_fma(recv[2], de::CPf(0.309017, 0.9510565), tmp);
        tmp = _complex_fma(recv[3], de::CPf(0.309017, -0.9510565), tmp);
        res[3] = _complex_fma(recv[4], de::CPf(-0.809017, 0.5877853), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(0.309017, -0.9510565), tmp);
        tmp = _complex_fma(recv[2], de::CPf(-0.809017, -0.5877853), tmp);
        tmp = _complex_fma(recv[3], de::CPf(-0.809017, 0.5877853), tmp);
        res[4] = _complex_fma(recv[4], de::CPf(0.309017, 0.9510565), tmp);

        dex_store = (dex / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id;

        dst[dex_store] = *((float2*)&res[0]);
        dst[dex_store + num_of_Bcalc_in_warp] = *((float2*)&res[1]);
        dst[dex_store + num_of_Bcalc_in_warp * 2] = *((float2*)&res[2]);
        dst[dex_store + num_of_Bcalc_in_warp * 3] = *((float2*)&res[3]);
        dst[dex_store + num_of_Bcalc_in_warp * 4] = *((float2*)&res[4]);
    }
}



__global__
void decx::signal::GPUK::cu_IFFT1D_R5_C2R_last(const float2* src, float* dst, const size_t B_ops_num, const uint warp_proc_len)
{
    using namespace decx::signal::cuda::dev;
    size_t dex = threadIdx.x + blockDim.x * blockIdx.x;
    size_t dex_store;

    uint warp_loc_id, num_of_Bcalc_in_warp = warp_proc_len / 5;

    de::CPf recv[5], tmp;
    de::CPf W;
    float res[5];

    if (dex < B_ops_num) {
        *((float2*)&recv[0]) = src[dex];
        *((float2*)&recv[1]) = src[dex + B_ops_num];
        *((float2*)&recv[2]) = src[dex + B_ops_num * 2];
        *((float2*)&recv[3]) = src[dex + B_ops_num * 3];
        *((float2*)&recv[4]) = src[dex + B_ops_num * 4];

        warp_loc_id = dex % num_of_Bcalc_in_warp;
        W.dev_construct_with_phase(Two_Pi * (float)warp_loc_id / (float)warp_proc_len);
        recv[1] = _complex_mul(recv[1], W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 2) / (float)warp_proc_len);
        recv[2] = _complex_mul(recv[2], W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 3) / (float)warp_proc_len);
        recv[3] = _complex_mul(recv[3], W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4) / (float)warp_proc_len);
        recv[4] = _complex_mul(recv[4], W);

        res[0] = recv[0].real + recv[1].real + recv[2].real + recv[3].real + recv[4].real;

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(0.309017, 0.9510565), tmp);
        tmp = _complex_fma(recv[2], de::CPf(-0.809017, 0.5877853), tmp);
        tmp = _complex_fma(recv[3], de::CPf(-0.809017, -0.5877853), tmp);
        res[1] = _complex_fma_preserve_R(recv[4], de::CPf(0.309017, -0.9510565), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(-0.809017, 0.5877853), tmp);
        tmp = _complex_fma(recv[2], de::CPf(0.309017, -0.9510565), tmp);
        tmp = _complex_fma(recv[3], de::CPf(0.309017, 0.9510565), tmp);
        res[2] = _complex_fma_preserve_R(recv[4], de::CPf(-0.809017, -0.5877853), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(-0.809017, -0.5877853), tmp);
        tmp = _complex_fma(recv[2], de::CPf(0.309017, 0.9510565), tmp);
        tmp = _complex_fma(recv[3], de::CPf(0.309017, -0.9510565), tmp);
        res[3] = _complex_fma_preserve_R(recv[4], de::CPf(-0.809017, 0.5877853), tmp);

        tmp = recv[0];
        tmp = _complex_fma(recv[1], de::CPf(0.309017, -0.9510565), tmp);
        tmp = _complex_fma(recv[2], de::CPf(-0.809017, -0.5877853), tmp);
        tmp = _complex_fma(recv[3], de::CPf(-0.809017, 0.5877853), tmp);
        res[4] = _complex_fma_preserve_R(recv[4], de::CPf(0.309017, 0.9510565), tmp);

        dex_store = (dex / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id;

        dst[dex_store] = res[0];
        dst[dex_store + num_of_Bcalc_in_warp] = res[1];
        dst[dex_store + num_of_Bcalc_in_warp * 2] = res[2];
        dst[dex_store + num_of_Bcalc_in_warp * 3] = res[3];
        dst[dex_store + num_of_Bcalc_in_warp * 4] = res[4];
    }
}


__global__
/*
* @param B_ops_num : in Vec4
* @param warp_proc_len : element
*/
void decx::signal::GPUK::cu_FFT1D_R5_C2C_vec4(const float4* src, float4* dst, const size_t B_ops_num, const uint warp_proc_len)
{
    using namespace decx::signal::cuda::dev;
    size_t dex = threadIdx.x + blockDim.x * blockIdx.x;
    size_t dex_store;

    uint warp_loc_id, num_of_Bcalc_in_warp = warp_proc_len / 5 / 4;

    de::CPf W, tmp[4];
    float4 res, recv[5][2];

    if (dex < B_ops_num) {
        recv[0][0] = src[dex * 2];                             recv[0][1] = src[dex * 2 + 1];
        recv[1][0] = src[(dex + B_ops_num) * 2];               recv[1][1] = src[(dex + B_ops_num) * 2 + 1];
        recv[2][0] = src[(dex + B_ops_num * 2) * 2];           recv[2][1] = src[(dex + B_ops_num * 2) * 2 + 1];
        recv[3][0] = src[(dex + B_ops_num * 3) * 2];           recv[3][1] = src[(dex + B_ops_num * 3) * 2 + 1];
        recv[4][0] = src[(dex + B_ops_num * 4) * 2];           recv[4][1] = src[(dex + B_ops_num * 4) * 2 + 1];

        warp_loc_id = dex % (size_t)num_of_Bcalc_in_warp;
        dex_store = (dex / (size_t)num_of_Bcalc_in_warp) * (size_t)warp_proc_len / 4 + warp_loc_id;

        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4) / (float)warp_proc_len);
        *((de::CPf*)&recv[1][0].x) = _complex_mul(*((de::CPf*)&recv[1][0].x), W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 1) / (float)warp_proc_len);
        *((de::CPf*)&recv[1][0].z) = _complex_mul(*((de::CPf*)&recv[1][0].z), W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 2) / (float)warp_proc_len);
        *((de::CPf*)&recv[1][1].x) = _complex_mul(*((de::CPf*)&recv[1][1].x), W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 3) / (float)warp_proc_len);
        *((de::CPf*)&recv[1][1].z) = _complex_mul(*((de::CPf*)&recv[1][1].z), W);

        W.dev_construct_with_phase(Four_Pi * (float)(warp_loc_id * 4) / (float)warp_proc_len);
        *((de::CPf*)&recv[2][0].x) = _complex_mul(*((de::CPf*)&recv[2][0].x), W);
        W.dev_construct_with_phase(Four_Pi * (float)(warp_loc_id * 4 + 1) / (float)warp_proc_len);
        *((de::CPf*)&recv[2][0].z) = _complex_mul(*((de::CPf*)&recv[2][0].z), W);
        W.dev_construct_with_phase(Four_Pi * (float)(warp_loc_id * 4 + 2) / (float)warp_proc_len);
        *((de::CPf*)&recv[2][1].x) = _complex_mul(*((de::CPf*)&recv[2][1].x), W);
        W.dev_construct_with_phase(Four_Pi * (float)(warp_loc_id * 4 + 3) / (float)warp_proc_len);
        *((de::CPf*)&recv[2][1].z) = _complex_mul(*((de::CPf*)&recv[2][1].z), W);

        W.dev_construct_with_phase(Six_Pi * (float)(warp_loc_id * 4) / (float)warp_proc_len);
        *((de::CPf*)&recv[3][0].x) = _complex_mul(*((de::CPf*)&recv[3][0].x), W);
        W.dev_construct_with_phase(Six_Pi * (float)(warp_loc_id * 4 + 1) / (float)warp_proc_len);
        *((de::CPf*)&recv[3][0].z) = _complex_mul(*((de::CPf*)&recv[3][0].z), W);
        W.dev_construct_with_phase(Six_Pi * (float)(warp_loc_id * 4 + 2) / (float)warp_proc_len);
        *((de::CPf*)&recv[3][1].x) = _complex_mul(*((de::CPf*)&recv[3][1].x), W);
        W.dev_construct_with_phase(Six_Pi * (float)(warp_loc_id * 4 + 3) / (float)warp_proc_len);
        *((de::CPf*)&recv[3][1].z) = _complex_mul(*((de::CPf*)&recv[3][1].z), W);

        W.dev_construct_with_phase(Eight_Pi * (float)(warp_loc_id * 4) / (float)warp_proc_len);
        *((de::CPf*)&recv[4][0].x) = _complex_mul(*((de::CPf*)&recv[4][0].x), W);
        W.dev_construct_with_phase(Eight_Pi * (float)(warp_loc_id * 4 + 1) / (float)warp_proc_len);
        *((de::CPf*)&recv[4][0].z) = _complex_mul(*((de::CPf*)&recv[4][0].z), W);
        W.dev_construct_with_phase(Eight_Pi * (float)(warp_loc_id * 4 + 2) / (float)warp_proc_len);
        *((de::CPf*)&recv[4][1].x) = _complex_mul(*((de::CPf*)&recv[4][1].x), W);
        W.dev_construct_with_phase(Eight_Pi * (float)(warp_loc_id * 4 + 3) / (float)warp_proc_len);
        *((de::CPf*)&recv[4][1].z) = _complex_mul(*((de::CPf*)&recv[4][1].z), W);

        res = _complex_sum2_4(recv[0][0], recv[1][0], recv[2][0], recv[3][0]);
        res = _complex_add2(recv[4][0], res);
        dst[dex_store * 2] = res;
        res = _complex_sum2_4(recv[0][1], recv[1][1], recv[2][1], recv[3][1]);
        res = _complex_add2(recv[4][1], res);
        dst[dex_store * 2 + 1] = res;

        dex_store += num_of_Bcalc_in_warp;
        res = recv[0][0];
        res = _complex_2fma1(recv[1][0], de::CPf(0.309017, 0.9510565), res);
        res = _complex_2fma1(recv[2][0], de::CPf(-0.809017, 0.5877853), res);
        res = _complex_2fma1(recv[3][0], de::CPf(-0.809017, -0.5877853), res);
        res = _complex_2fma1(recv[4][0], de::CPf(0.309017, -0.9510565), res);
        dst[dex_store * 2] = res;
        res = recv[0][1];
        res = _complex_2fma1(recv[1][1], de::CPf(0.309017, 0.9510565), res);
        res = _complex_2fma1(recv[2][1], de::CPf(-0.809017, 0.5877853), res);
        res = _complex_2fma1(recv[3][1], de::CPf(-0.809017, -0.5877853), res);
        res = _complex_2fma1(recv[4][1], de::CPf(0.309017, -0.9510565), res);
        dst[dex_store * 2 + 1] = res;

        dex_store += num_of_Bcalc_in_warp;
        res = *((float4*)&recv[0][0]);
        res = _complex_2fma1(recv[1][0], de::CPf(-0.809017, 0.5877853), res);
        res = _complex_2fma1(recv[2][0], de::CPf(0.309017, -0.9510565), res);
        res = _complex_2fma1(recv[3][0], de::CPf(0.309017, 0.9510565), res);
        res = _complex_2fma1(recv[4][0], de::CPf(-0.809017, -0.5877853), res);
        dst[dex_store * 2] = res;
        res = recv[0][1];
        res = _complex_2fma1(recv[1][1], de::CPf(-0.809017, 0.5877853), res);
        res = _complex_2fma1(recv[2][1], de::CPf(0.309017, -0.9510565), res);
        res = _complex_2fma1(recv[3][1], de::CPf(0.309017, 0.9510565), res);
        res = _complex_2fma1(recv[4][1], de::CPf(-0.809017, -0.5877853), res);
        dst[dex_store * 2 + 1] = res;

        dex_store += num_of_Bcalc_in_warp;
        res = *((float4*)&recv[0][0]);
        res = _complex_2fma1(recv[1][0], de::CPf(-0.809017, -0.5877853), res);
        res = _complex_2fma1(recv[2][0], de::CPf(0.309017, 0.9510565), res);
        res = _complex_2fma1(recv[3][0], de::CPf(0.309017, -0.9510565), res);
        res = _complex_2fma1(recv[4][0], de::CPf(-0.809017, 0.5877853), res);
        dst[dex_store * 2] = res;
        res = recv[0][1];
        res = _complex_2fma1(recv[1][1], de::CPf(-0.809017, -0.5877853), res);
        res = _complex_2fma1(recv[2][1], de::CPf(0.309017, 0.9510565), res);
        res = _complex_2fma1(recv[3][1], de::CPf(0.309017, -0.9510565), res);
        res = _complex_2fma1(recv[4][1], de::CPf(-0.809017, 0.5877853), res);
        dst[dex_store * 2 + 1] = res;

        dex_store += num_of_Bcalc_in_warp;
        res = *((float4*)&recv[0][0]);
        res = _complex_2fma1(recv[1][0], de::CPf(0.309017, -0.9510565), res);
        res = _complex_2fma1(recv[2][0], de::CPf(-0.809017, -0.5877853), res);
        res = _complex_2fma1(recv[3][0], de::CPf(-0.809017, 0.5877853), res);
        res = _complex_2fma1(recv[4][0], de::CPf(0.309017, 0.9510565), res);
        dst[dex_store * 2] = res;
        res = recv[0][1];
        res = _complex_2fma1(recv[1][1], de::CPf(0.309017, -0.9510565), res);
        res = _complex_2fma1(recv[2][1], de::CPf(-0.809017, -0.5877853), res);
        res = _complex_2fma1(recv[3][1], de::CPf(-0.809017, 0.5877853), res);
        res = _complex_2fma1(recv[4][1], de::CPf(0.309017, 0.9510565), res);
        dst[dex_store * 2 + 1] = res;
    }
}



__global__
/**
* Somehow it won't be executed, no matter how hard I try
* @param B_ops_num : in Vec4
* @param warp_proc_len : element
*/
void decx::signal::GPUK::cu_IFFT1D_R5_C2R_vec4_last(const float4* src, float4* dst, const size_t B_ops_num, const uint warp_proc_len)
{
    using namespace decx::signal::cuda::dev;
    size_t dex = threadIdx.x + blockDim.x * blockIdx.x;
    size_t dex_store;

    uint warp_loc_id, num_of_Bcalc_in_warp = warp_proc_len / 5 / 4;

    de::CPf W;
    float4 res, recv[5][2], tmp;

    if (dex < B_ops_num) {
        recv[0][0] = src[dex * 2];                             recv[0][1] = src[dex * 2 + 1];
        recv[1][0] = src[(dex + B_ops_num) * 2];               recv[1][1] = src[(dex + B_ops_num) * 2 + 1];
        recv[2][0] = src[(dex + B_ops_num * 2) * 2];           recv[2][1] = src[(dex + B_ops_num * 2) * 2 + 1];
        recv[3][0] = src[(dex + B_ops_num * 3) * 2];           recv[3][1] = src[(dex + B_ops_num * 3) * 2 + 1];
        recv[4][0] = src[(dex + B_ops_num * 4) * 2];           recv[4][1] = src[(dex + B_ops_num * 4) * 2 + 1];

        warp_loc_id = dex % (size_t)num_of_Bcalc_in_warp;
        dex_store = (dex / (size_t)num_of_Bcalc_in_warp) * (size_t)warp_proc_len / 4 + warp_loc_id;

        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4) / (float)warp_proc_len);
        *((de::CPf*)&recv[1][0].x) = _complex_mul(*((de::CPf*)&recv[1][0].x), W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 1) / (float)warp_proc_len);
        *((de::CPf*)&recv[1][0].z) = _complex_mul(*((de::CPf*)&recv[1][0].z), W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 2) / (float)warp_proc_len);
        *((de::CPf*)&recv[1][1].x) = _complex_mul(*((de::CPf*)&recv[1][1].x), W);
        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 3) / (float)warp_proc_len);
        *((de::CPf*)&recv[1][1].z) = _complex_mul(*((de::CPf*)&recv[1][1].z), W);

        W.dev_construct_with_phase(Four_Pi * (float)(warp_loc_id * 4) / (float)warp_proc_len);
        *((de::CPf*)&recv[2][0].x) = _complex_mul(*((de::CPf*)&recv[2][0].x), W);
        W.dev_construct_with_phase(Four_Pi * (float)(warp_loc_id * 4 + 1) / (float)warp_proc_len);
        *((de::CPf*)&recv[2][0].z) = _complex_mul(*((de::CPf*)&recv[2][0].z), W);
        W.dev_construct_with_phase(Four_Pi * (float)(warp_loc_id * 4 + 2) / (float)warp_proc_len);
        *((de::CPf*)&recv[2][1].x) = _complex_mul(*((de::CPf*)&recv[2][1].x), W);
        W.dev_construct_with_phase(Four_Pi * (float)(warp_loc_id * 4 + 3) / (float)warp_proc_len);
        *((de::CPf*)&recv[2][1].z) = _complex_mul(*((de::CPf*)&recv[2][1].z), W);

        W.dev_construct_with_phase(Six_Pi * (float)(warp_loc_id * 4) / (float)warp_proc_len);
        *((de::CPf*)&recv[3][0].x) = _complex_mul(*((de::CPf*)&recv[3][0].x), W);
        W.dev_construct_with_phase(Six_Pi * (float)(warp_loc_id * 4 + 1) / (float)warp_proc_len);
        *((de::CPf*)&recv[3][0].z) = _complex_mul(*((de::CPf*)&recv[3][0].z), W);
        W.dev_construct_with_phase(Six_Pi * (float)(warp_loc_id * 4 + 2) / (float)warp_proc_len);
        *((de::CPf*)&recv[3][1].x) = _complex_mul(*((de::CPf*)&recv[3][1].x), W);
        W.dev_construct_with_phase(Six_Pi * (float)(warp_loc_id * 4 + 3) / (float)warp_proc_len);
        *((de::CPf*)&recv[3][1].z) = _complex_mul(*((de::CPf*)&recv[3][1].z), W);

        W.dev_construct_with_phase(Eight_Pi * (float)(warp_loc_id * 4) / (float)warp_proc_len);
        *((de::CPf*)&recv[4][0].x) = _complex_mul(*((de::CPf*)&recv[4][0].x), W);
        W.dev_construct_with_phase(Eight_Pi * (float)(warp_loc_id * 4 + 1) / (float)warp_proc_len);
        *((de::CPf*)&recv[4][0].z) = _complex_mul(*((de::CPf*)&recv[4][0].z), W);
        W.dev_construct_with_phase(Eight_Pi * (float)(warp_loc_id * 4 + 2) / (float)warp_proc_len);
        *((de::CPf*)&recv[4][1].x) = _complex_mul(*((de::CPf*)&recv[4][1].x), W);
        W.dev_construct_with_phase(Eight_Pi * (float)(warp_loc_id * 4 + 3) / (float)warp_proc_len);
        *((de::CPf*)&recv[4][1].z) = _complex_mul(*((de::CPf*)&recv[4][1].z), W);

        res.x = __fadd_rn(__fadd_rn(recv[0][0].x, recv[1][0].x), __fadd_rn(recv[2][0].x, recv[3][0].x));
        res.x = __fadd_rn(recv[4][0].x, res.x);
        res.y = __fadd_rn(__fadd_rn(recv[0][0].z, recv[1][0].z), __fadd_rn(recv[2][0].z, recv[3][0].z));
        res.y = __fadd_rn(recv[4][0].z, res.y);
        res.z = __fadd_rn(__fadd_rn(recv[0][1].x, recv[1][1].x), __fadd_rn(recv[2][1].x, recv[3][1].x));
        res.z = __fadd_rn(recv[4][1].x, res.z);
        res.w = __fadd_rn(__fadd_rn(recv[0][1].z, recv[1][1].z), __fadd_rn(recv[2][1].z, recv[3][1].z));
        res.w = __fadd_rn(recv[4][1].z, res.w);
        dst[dex_store] = res;

        dex_store += num_of_Bcalc_in_warp;
        tmp = recv[0][0];
        tmp = _complex_2fma1(recv[1][0], de::CPf(0.309017, 0.9510565), tmp);
        tmp = _complex_2fma1(recv[2][0], de::CPf(-0.809017, 0.5877853), tmp);
        tmp = _complex_2fma1(recv[3][0], de::CPf(-0.809017, -0.5877853), tmp);
        res.x = recv[4][0].x * 0.309017 + recv[4][0].y * 0.9510565 + tmp.x;
        res.y = recv[4][0].z * 0.309017 + recv[4][0].w * 0.9510565 + tmp.z;

        tmp = recv[0][1];
        tmp = _complex_2fma1(recv[1][1], de::CPf(0.309017, 0.9510565), tmp);
        tmp = _complex_2fma1(recv[2][1], de::CPf(-0.809017, 0.5877853), tmp);
        tmp = _complex_2fma1(recv[3][1], de::CPf(-0.809017, -0.5877853), tmp);
        res.z = recv[4][1].x * 0.309017 + recv[4][1].y * 0.9510565 + tmp.x;
        res.w = recv[4][1].z * 0.309017 + recv[4][1].w * 0.9510565 + tmp.z;
        dst[dex_store] = res;

        dex_store += num_of_Bcalc_in_warp;
        tmp = recv[0][0];
        tmp = _complex_2fma1(recv[1][0], de::CPf(-0.809017, 0.5877853), tmp);
        tmp = _complex_2fma1(recv[2][0], de::CPf(0.309017, -0.9510565), tmp);
        tmp = _complex_2fma1(recv[3][0], de::CPf(0.309017, 0.9510565), tmp);
        res.x = recv[4][0].x * -0.809017 + recv[4][0].y * 0.5877853 + tmp.x;
        res.y = recv[4][0].z * -0.809017 + recv[4][0].w * 0.5877853 + tmp.z;

        tmp = recv[0][1];
        tmp = _complex_2fma1(recv[1][1], de::CPf(-0.809017, 0.5877853), tmp);
        tmp = _complex_2fma1(recv[2][1], de::CPf(0.309017, -0.9510565), tmp);
        tmp = _complex_2fma1(recv[3][1], de::CPf(0.309017, 0.9510565), tmp);
        res.z = recv[4][1].x * -0.809017 + recv[4][1].y * 0.5877853 + tmp.x;
        res.w = recv[4][1].z * -0.809017 + recv[4][1].w * 0.5877853 + tmp.z;
        dst[dex_store] = res;

        dex_store += num_of_Bcalc_in_warp;
        tmp = recv[0][0];
        tmp = _complex_2fma1(recv[1][0], de::CPf(-0.809017, -0.5877853), tmp);
        tmp = _complex_2fma1(recv[2][0], de::CPf(0.309017, 0.9510565), tmp);
        tmp = _complex_2fma1(recv[3][0], de::CPf(0.309017, -0.9510565), tmp);
        res.x = recv[4][0].x * -0.809017 - recv[4][0].y * 0.5877853 + tmp.x;
        res.y = recv[4][0].z * -0.809017 - recv[4][0].w * 0.5877853 + tmp.z;

        tmp = recv[0][1];
        tmp = _complex_2fma1(recv[1][1], de::CPf(-0.809017, -0.5877853), tmp);
        tmp = _complex_2fma1(recv[2][1], de::CPf(0.309017, 0.9510565), tmp);
        tmp = _complex_2fma1(recv[3][1], de::CPf(0.309017, -0.9510565), tmp);
        res.z = recv[4][1].x * -0.809017 - recv[4][1].y * 0.5877853 + tmp.x;
        res.w = recv[4][1].z * -0.809017 - recv[4][1].w * 0.5877853 + tmp.z;
        dst[dex_store] = res;

        dex_store += num_of_Bcalc_in_warp;
        tmp = recv[0][0];
        tmp = _complex_2fma1(recv[1][0], de::CPf(0.309017, -0.9510565), tmp);
        tmp = _complex_2fma1(recv[2][0], de::CPf(-0.809017, -0.5877853), tmp);
        tmp = _complex_2fma1(recv[3][0], de::CPf(-0.809017, 0.5877853), tmp);
        res.x = recv[4][0].x * 0.309017 - recv[4][0].y * 0.9510565 + tmp.x;
        res.y = recv[4][0].z * 0.309017 - recv[4][0].w * 0.9510565 + tmp.z;
        
        tmp = recv[0][1];
        tmp = _complex_2fma1(recv[1][1], de::CPf(0.309017, -0.9510565), tmp);
        tmp = _complex_2fma1(recv[2][1], de::CPf(-0.809017, -0.5877853), tmp);
        tmp = _complex_2fma1(recv[3][1], de::CPf(-0.809017, 0.5877853), tmp);
        res.z = recv[4][1].x * 0.309017 - recv[4][1].y * 0.9510565 + tmp.x;
        res.w = recv[4][1].z * 0.309017 - recv[4][1].w * 0.9510565 + tmp.z;
        dst[dex_store] = res;
    }
}