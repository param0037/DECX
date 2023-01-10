/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "FFT1D_Radix_2_kernel.cuh"


__global__
void decx::signal::GPUK::cu_FFT1D_R2_R2C_first(const float* src, float4* dst, const size_t B_ops_num)
{
    size_t dex = threadIdx.x + blockDim.x * blockIdx.x;

    float2 recv;
    de::CPf res[2] = { de::CPf(0, 0), de::CPf(0, 0) };

    if (dex < B_ops_num) {
        recv.x = src[dex];
        recv.y = src[dex + B_ops_num];

        res[0].real = __fadd_rn(recv.x, recv.y);
        res[1].real = __fsub_rn(recv.x, recv.y);

        dst[dex] = *((float4*)&res[0]);
    }
}


__global__
void decx::signal::GPUK::cu_IFFT1D_R2_C2C_first(const float2* src, float4* dst, const size_t B_ops_num)
{
    size_t dex = threadIdx.x + blockDim.x * blockIdx.x;

    float4 recv;
    de::CPf res[2];
    const float signal_length = (float)B_ops_num * 2;

    if (dex < B_ops_num) {
        *((float2*)&recv.x) = src[dex];
        *((float2*)&recv.z) = src[dex + B_ops_num];

        recv.x = __fdividef(recv.x, signal_length);
        recv.y = __fdividef(recv.y, -signal_length);
        recv.z = __fdividef(recv.z, signal_length);
        recv.w = __fdividef(recv.w, -signal_length);

        res[0].real = __fadd_rn(recv.x, recv.z);
        res[0].image = __fadd_rn(recv.y, recv.w);
        res[1].real = __fsub_rn(recv.x, recv.z);
        res[1].image = __fsub_rn(recv.y, recv.w);

        dst[dex] = *((float4*)&res[0]);
    }
}


__global__
void decx::signal::GPUK::cu_FFT1D_R2_C2C_first(const float2* src, float4* dst, const size_t B_ops_num)
{
    size_t dex = threadIdx.x + blockDim.x * blockIdx.x;

    float4 recv;
    de::CPf res[2];
    const float signal_length = (float)B_ops_num * 2;

    if (dex < B_ops_num) {
        *((float2*)&recv.x) = src[dex];
        *((float2*)&recv.z) = src[dex + B_ops_num];

        res[0].real = __fadd_rn(recv.x, recv.z);
        res[0].image = __fadd_rn(recv.y, recv.w);
        res[1].real = __fsub_rn(recv.x, recv.z);
        res[1].image = __fsub_rn(recv.y, recv.w);

        dst[dex] = *((float4*)&res[0]);
    }
}



__global__
void decx::signal::GPUK::cu_IFFT1D_R2_C2R_once(const float2* src, float2* dst, const size_t B_ops_num)
{
    size_t dex = threadIdx.x + blockDim.x * blockIdx.x;

    float4 recv;
    float2 res;
    const float signal_length = (float)B_ops_num * 2;

    if (dex < B_ops_num) {
        *((float2*)&recv.x) = src[dex];
        *((float2*)&recv.z) = src[dex + B_ops_num];

        recv.x = __fdividef(recv.x, signal_length);
        recv.y = __fdividef(recv.y, -signal_length);
        recv.z = __fdividef(recv.z, signal_length);
        recv.w = __fdividef(recv.w, -signal_length);

        res.x = __fadd_rn(recv.x, recv.z);
        res.y = __fsub_rn(recv.x, recv.z);

        dst[dex] = res;
    }
}



__global__
void decx::signal::GPUK::cu_FFT1D_R2_C2C(const float2* src, float2* dst, const size_t B_ops_num, const uint warp_proc_len)
{
    size_t dex = threadIdx.x + blockDim.x * blockIdx.x;
    size_t dex_store;

    uint warp_loc_id, num_of_Bcalc_in_warp = warp_proc_len / 2;

    de::CPf recv[2], tmp;
    de::CPf W, res[2] = { de::CPf(0, 0), de::CPf(0, 0) };

    if (dex < B_ops_num) {
        *((float2*)&recv[0]) = src[dex];
        *((float2*)&recv[1]) = src[dex + B_ops_num];

        warp_loc_id = dex % num_of_Bcalc_in_warp;
        W.dev_construct_with_phase(Two_Pi * (float)warp_loc_id / (float)warp_proc_len);

        tmp.real = __fsub_rn(__fmul_rn(recv[1].real, W.real), __fmul_rn(recv[1].image, W.image));
        tmp.image = __fadd_rn(__fmul_rn(recv[1].real, W.image), __fmul_rn(recv[1].image, W.real));

        res[0].real = __fadd_rn(recv[0].real, tmp.real);
        res[0].image = __fadd_rn(recv[0].image, tmp.image);
        res[1].real = __fsub_rn(recv[0].real, tmp.real);
        res[1].image = __fsub_rn(recv[0].image, tmp.image);

        dex_store = (dex / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id;

        dst[dex_store] = *((float2*)&res[0]);
        dst[dex_store + num_of_Bcalc_in_warp] = *((float2*)&res[1]);
    }
}



__global__
void decx::signal::GPUK::cu_IFFT1D_R2_C2R_last(const float2* src, float* dst, const size_t B_ops_num, const uint warp_proc_len)
{
    size_t dex = threadIdx.x + blockDim.x * blockIdx.x;
    size_t dex_store;

    uint warp_loc_id, num_of_Bcalc_in_warp = warp_proc_len / 2;
    const float signal_len = (float)B_ops_num * 2;

    de::CPf recv[2], tmp;
    de::CPf W;
    float res[2];

    if (dex < B_ops_num) {
        *((float2*)&recv[0]) = src[dex];
        *((float2*)&recv[1]) = src[dex + B_ops_num];

        warp_loc_id = dex % num_of_Bcalc_in_warp;
        W.dev_construct_with_phase(Two_Pi * (float)warp_loc_id / (float)warp_proc_len);

        tmp.real = __fsub_rn(__fmul_rn(recv[1].real, W.real), __fmul_rn(recv[1].image, W.image));
        tmp.image = __fadd_rn(__fmul_rn(recv[1].real, W.image), __fmul_rn(recv[1].image, W.real));

        res[0] = __fdividef(__fadd_rn(recv[0].real, tmp.real), signal_len);
        res[1] = __fdividef(__fsub_rn(recv[0].real, tmp.real), signal_len);

        dex_store = (dex / num_of_Bcalc_in_warp) * warp_proc_len + warp_loc_id;

        dst[dex_store] = res[0];
        dst[dex_store + num_of_Bcalc_in_warp] = res[1];
    }
}


__global__
/*
* @param B_ops_num : in Vec4
* @param warp_proc_len : element
*/
void decx::signal::GPUK::cu_FFT1D_R2_C2C_vec4(const float4* src, float4* dst, const size_t B_ops_num, const uint warp_proc_len)
{
    size_t dex = threadIdx.x + blockDim.x * blockIdx.x;
    size_t dex_store;

    uint warp_loc_id, num_of_Bcalc_in_warp = warp_proc_len / 2 / 4;

    de::CPf W, tmp[4];
    float4 recv[4], res;

    if (dex < B_ops_num) {
        recv[0] = src[dex * 2];
        recv[1] = src[dex * 2 + 1];
        recv[2] = src[(dex + B_ops_num) * 2];
        recv[3] = src[(dex + B_ops_num) * 2 + 1];

        warp_loc_id = dex % (size_t)num_of_Bcalc_in_warp;
        dex_store = (dex / (size_t)num_of_Bcalc_in_warp) * (size_t)warp_proc_len / 4 + warp_loc_id;

        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4) / (float)warp_proc_len);
        tmp[0].real = __fsub_rn(__fmul_rn(recv[2].x, W.real), __fmul_rn(recv[2].y, W.image));
        tmp[0].image = __fadd_rn(__fmul_rn(recv[2].x, W.image), __fmul_rn(recv[2].y, W.real));

        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 1) / (float)warp_proc_len);
        tmp[1].real = __fsub_rn(__fmul_rn(recv[2].z, W.real), __fmul_rn(recv[2].w, W.image));
        tmp[1].image = __fadd_rn(__fmul_rn(recv[2].z, W.image), __fmul_rn(recv[2].w, W.real));

        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 2) / (float)warp_proc_len);
        tmp[2].real = __fsub_rn(__fmul_rn(recv[3].x, W.real), __fmul_rn(recv[3].y, W.image));
        tmp[2].image = __fadd_rn(__fmul_rn(recv[3].x, W.image), __fmul_rn(recv[3].y, W.real));

        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 3) / (float)warp_proc_len);
        tmp[3].real = __fsub_rn(__fmul_rn(recv[3].z, W.real), __fmul_rn(recv[3].w, W.image));
        tmp[3].image = __fadd_rn(__fmul_rn(recv[3].z, W.image), __fmul_rn(recv[3].w, W.real));

        res.x = __fadd_rn(recv[0].x, tmp[0].real);
        res.y = __fadd_rn(recv[0].y, tmp[0].image);
        res.z = __fadd_rn(recv[0].z, tmp[1].real);
        res.w = __fadd_rn(recv[0].w, tmp[1].image);
        dst[dex_store * 2] = res;

        res.x = __fadd_rn(recv[1].x, tmp[2].real);
        res.y = __fadd_rn(recv[1].y, tmp[2].image);
        res.z = __fadd_rn(recv[1].z, tmp[3].real);
        res.w = __fadd_rn(recv[1].w, tmp[3].image);
        dst[dex_store * 2 + 1] = res;

        dex_store += num_of_Bcalc_in_warp;

        res.x = __fsub_rn(recv[0].x, tmp[0].real);
        res.y = __fsub_rn(recv[0].y, tmp[0].image);
        res.z = __fsub_rn(recv[0].z, tmp[1].real);
        res.w = __fsub_rn(recv[0].w, tmp[1].image);
        dst[dex_store * 2] = res;

        res.x = __fsub_rn(recv[1].x, tmp[2].real);
        res.y = __fsub_rn(recv[1].y, tmp[2].image);
        res.z = __fsub_rn(recv[1].z, tmp[3].real);
        res.w = __fsub_rn(recv[1].w, tmp[3].image);
        dst[dex_store * 2 + 1] = res;
    }
}



__global__
/*
* @param B_ops_num : in Vec4
* @param warp_proc_len : element
*/
void decx::signal::GPUK::cu_IFFT1D_R2_C2R_vec4_last(const float4* src, float4* dst, const size_t B_ops_num, const uint warp_proc_len)
{
    using namespace decx::signal::cuda::dev;
    size_t dex = threadIdx.x + blockDim.x * blockIdx.x;
    size_t dex_store;

    uint warp_loc_id, num_of_Bcalc_in_warp = warp_proc_len / 2 / 4;

    de::CPf W, tmp[4];
    float4 recv[4], res;

    if (dex < B_ops_num) {
        recv[0] = src[dex * 2];
        recv[1] = src[dex * 2 + 1];
        recv[2] = src[(dex + B_ops_num) * 2];
        recv[3] = src[(dex + B_ops_num) * 2 + 1];

        warp_loc_id = dex % (size_t)num_of_Bcalc_in_warp;
        dex_store = (dex / (size_t)num_of_Bcalc_in_warp) * (size_t)warp_proc_len / 4 + warp_loc_id;

        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4) / (float)warp_proc_len);
        *((de::CPf*)&recv[2].x) = _complex_mul(*((de::CPf*)&recv[2].x), W);

        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 1) / (float)warp_proc_len);
        *((de::CPf*)&recv[2].z) = _complex_mul(*((de::CPf*)&recv[2].z), W);

        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 2) / (float)warp_proc_len);
        *((de::CPf*)&recv[3].x) = _complex_mul(*((de::CPf*)&recv[3].x), W);

        W.dev_construct_with_phase(Two_Pi * (float)(warp_loc_id * 4 + 3) / (float)warp_proc_len);
        *((de::CPf*)&recv[3].z) = _complex_mul(*((de::CPf*)&recv[3].z), W);

        res.x = __fadd_rn(recv[0].x, recv[2].x);
        res.y = __fadd_rn(recv[0].z, recv[2].z);
        res.z = __fadd_rn(recv[1].x, recv[3].x);
        res.w = __fadd_rn(recv[1].z, recv[3].z);

        dst[dex_store] = res;

        dex_store += num_of_Bcalc_in_warp;

        res.x = __fsub_rn(recv[0].x, recv[2].x);
        res.y = __fsub_rn(recv[0].z, recv[2].z);
        res.z = __fsub_rn(recv[1].x, recv[3].x);
        res.w = __fsub_rn(recv[1].z, recv[3].z);

        dst[dex_store] = res;
    }
}