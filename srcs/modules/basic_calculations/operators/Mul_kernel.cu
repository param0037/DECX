/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "Mul_kernel.cuh"



__global__
/**
* int* x2, add together
* @param len : have considered vec4
*/
void decx::calc::GPUK::mul_m_ivec4(float4* A, float4* B, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    int4 tmpA, tmpB, tmpdst;

    if (tid < len) {
        tmpA = *((int4*)&A[tid]);
        tmpB = *((int4*)&B[tid]);

        tmpdst.x = tmpA.x * tmpB.x;
        tmpdst.y = tmpA.y * tmpB.y;
        tmpdst.z = tmpA.z * tmpB.z;
        tmpdst.w = tmpA.w * tmpB.w;

        dst[tid] = *((float4*)&tmpdst);
    }
}


__global__
/**
* int* x2, add together
* @param eq_pitch : have considered vec4
* @param bounds.x : The width, in float4
* @param bounds.y : The height, in float
*/
void decx::calc::GPUK::mul_m_ivec4_2D(float4* A, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    int4 tmpA, tmpB, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        *((float4*)&tmpA) = A[dex];
        *((float4*)&tmpB) = B[dex];

        tmpdst.x = tmpA.x * tmpB.x;
        tmpdst.y = tmpA.y * tmpB.y;
        tmpdst.z = tmpA.z * tmpB.z;
        tmpdst.w = tmpA.w * tmpB.w;

        dst[dex] = *((float4*)&tmpdst);
    }
}


__global__
/**
* int* x2, add together
* @param len : have considered vec4
*/
void decx::calc::GPUK::mul_m_fvec4(float4* A, float4* B, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    float4 tmpA, tmpB, tmpdst;

    if (tid < len) {
        tmpA = A[tid];
        tmpB = B[tid];

        tmpdst.x = __fmul_rn(tmpA.x, tmpB.x);
        tmpdst.y = __fmul_rn(tmpA.y, tmpB.y);
        tmpdst.z = __fmul_rn(tmpA.z, tmpB.z);
        tmpdst.w = __fmul_rn(tmpA.w, tmpB.w);

        dst[tid] = tmpdst;
    }
}



__global__
/**
* int* x2, add together
* @param eq_pitch : have considered vec4
* @param bounds.x : The width, in float4
* @param bounds.y : The height, in float
*/
void decx::calc::GPUK::mul_m_fvec4_2D(float4* A, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    float4 tmpA, tmpB, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        tmpA = A[dex];
        tmpB = B[dex];

        tmpdst.x = __fmul_rn(tmpA.x, tmpB.x);
        tmpdst.y = __fmul_rn(tmpA.y, tmpB.y);
        tmpdst.z = __fmul_rn(tmpA.z, tmpB.z);
        tmpdst.w = __fmul_rn(tmpA.w, tmpB.w);

        dst[dex] = tmpdst;
    }
}



__global__
void decx::calc::GPUK::mul_m_hvec8(float4* A, float4* B, float4* dst, const size_t len)
{
#if __ABOVE_SM_53
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    half2_8 tmpA, tmpB, tmpdst;

    if (tid < len) {
        tmpA = *((half2_8*)&A[tid]);
        tmpB = *((half2_8*)&B[tid]);

        tmpdst.x = __hmul2(tmpA.x, tmpB.x);
        tmpdst.y = __hmul2(tmpA.y, tmpB.y);
        tmpdst.z = __hmul2(tmpA.z, tmpB.z);
        tmpdst.w = __hmul2(tmpA.w, tmpB.w);

        dst[tid] = *((float4*)&tmpdst);
    }
#endif
}



__global__
/**
* int* x2, add together
* @param eq_pitch : have considered vec4
* @param bounds.x : The width, in float4
* @param bounds.y : The height, in float
*/
void decx::calc::GPUK::mul_m_hvec8_2D(float4* A, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
#if __ABOVE_SM_53
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    half2_8 tmpA, tmpB, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        *((float4*)&tmpA) = A[dex];
        *((float4*)&tmpB) = B[dex];

        tmpdst.x = __hmul2(tmpA.x, tmpB.x);
        tmpdst.y = __hmul2(tmpA.y, tmpB.y);
        tmpdst.z = __hmul2(tmpA.z, tmpB.z);
        tmpdst.w = __hmul2(tmpA.w, tmpB.w);

        dst[dex] = *((float4*)&tmpdst);
    }
#endif
}



__global__
/**
* int* x2, add together
* @param len : have considered vec4
*/
void decx::calc::GPUK::mul_m_dvec2(float4* A, float4* B, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    double2 tmpA, tmpB, tmpdst;

    if (tid < len) {
        tmpA = *((double2*)&A[tid]);
        tmpB = *((double2*)&B[tid]);

        tmpdst.x = __dmul_rn(tmpA.x, tmpB.x);
        tmpdst.y = __dmul_rn(tmpA.y, tmpB.y);

        dst[tid] = *((float4*)&tmpdst);
    }
}



__global__
/**
* int* x2, add together
* @param eq_pitch : have considered vec4
* @param bounds.x : The width, in float4
* @param bounds.y : The height, in float
*/
void decx::calc::GPUK::mul_m_dvec2_2D(float4* A, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    double2 tmpA, tmpB, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        *((float4*)&tmpA) = A[dex];
        *((float4*)&tmpB) = B[dex];

        tmpdst.x = __dmul_rn(tmpA.x, tmpB.x);
        tmpdst.y = __dmul_rn(tmpA.y, tmpB.y);

        dst[dex] = *((float4*)&tmpdst);
    }
}



// ----------------------------- C --------------------------------------


__global__
void decx::calc::GPUK::mul_c_ivec4(float4* src, int __x, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    int4 tmpsrc, tmpdst;

    if (tid < len) {
        tmpsrc = *((int4*)&src[tid]);

        tmpdst.x = tmpsrc.x * __x;
        tmpdst.y = tmpsrc.y * __x;
        tmpdst.z = tmpsrc.z * __x;
        tmpdst.w = tmpsrc.w * __x;

        dst[tid] = *((float4*)&tmpdst);
    }
}


__global__
/**
* int* x2, add together
* @param eq_pitch : have considered vec4
* @param bounds.x : The width, in float4
* @param bounds.y : The height, in float
*/
void decx::calc::GPUK::mul_c_ivec4_2D(float4* src, int __x, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    int4 tmpsrc, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        *((float4*)&tmpsrc) = src[dex];

        tmpdst.x = tmpsrc.x * __x;
        tmpdst.y = tmpsrc.y * __x;
        tmpdst.z = tmpsrc.z * __x;
        tmpdst.w = tmpsrc.w * __x;

        dst[dex] = *((float4*)&tmpdst);
    }
}


__global__
void decx::calc::GPUK::mul_c_fvec4(float4* src, float __x, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    float4 tmpsrc, tmpdst;

    if (tid < len) {
        tmpsrc = src[tid];

        tmpdst.x = __fmul_rn(tmpsrc.x, __x);
        tmpdst.y = __fmul_rn(tmpsrc.y, __x);
        tmpdst.z = __fmul_rn(tmpsrc.z, __x);
        tmpdst.w = __fmul_rn(tmpsrc.w, __x);

        dst[tid] = tmpdst;
    }
}



__global__
/**
* int* x2, add together
* @param eq_pitch : have considered vec4
* @param bounds.x : The width, in float4
* @param bounds.y : The height, in float
*/
void decx::calc::GPUK::mul_c_fvec4_2D(float4* src, float __x, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    float4 tmpsrc, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        tmpsrc = src[dex];

        tmpdst.x = __fmul_rn(tmpsrc.x, __x);
        tmpdst.y = __fmul_rn(tmpsrc.y, __x);
        tmpdst.z = __fmul_rn(tmpsrc.z, __x);
        tmpdst.w = __fmul_rn(tmpsrc.w, __x);

        dst[dex] = tmpdst;
    }
}


__global__
void decx::calc::GPUK::mul_c_hvec8(float4* src, half2 __x, float4* dst, const size_t len)
{
#if __ABOVE_SM_53
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    half2_8 tmpsrc, tmpdst;

    if (tid < len) {
        tmpsrc = *((half2_8*)&src[tid]);

        tmpdst.x = __hmul2(tmpsrc.x, __x);
        tmpdst.y = __hmul2(tmpsrc.y, __x);
        tmpdst.z = __hmul2(tmpsrc.z, __x);
        tmpdst.w = __hmul2(tmpsrc.w, __x);

        dst[tid] = *((float4*)&tmpdst);
    }
#endif
}



__global__
/**
* int* x2, add together
* @param eq_pitch : have considered vec4
* @param bounds.x : The width, in float4
* @param bounds.y : The height, in float
*/
void decx::calc::GPUK::mul_c_hvec8_2D(float4* src, half2 __x, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
#if __ABOVE_SM_53
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    half2_8 tmpsrc, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        *((float4*)&tmpsrc) = src[dex];

        tmpdst.x = __hmul2(tmpsrc.x, __x);
        tmpdst.y = __hmul2(tmpsrc.y, __x);
        tmpdst.z = __hmul2(tmpsrc.z, __x);
        tmpdst.w = __hmul2(tmpsrc.w, __x);

        dst[dex] = *((float4*)&tmpdst);
    }
#endif
}



__global__
void decx::calc::GPUK::mul_c_dvec2(float4* src, double __x, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    double2 tmpsrc, tmpdst;

    if (tid < len) {
        tmpsrc = *((double2*)&src[tid]);

        tmpdst.x = __dmul_rn(tmpsrc.x, __x);
        tmpdst.y = __dmul_rn(tmpsrc.y, __x);

        dst[tid] = *((float4*)&tmpdst);
    }
}



__global__
/**
* int* x2, add together
* @param eq_pitch : have considered vec4
* @param bounds.x : The width, in float4
* @param bounds.y : The height, in float
*/
void decx::calc::GPUK::mul_c_dvec2_2D(float4* src, double __x, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    double2 tmpsrc, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        *((float4*)&tmpsrc) = src[dex];

        tmpdst.x = __dmul_rn(tmpsrc.x, __x);
        tmpdst.y = __dmul_rn(tmpsrc.y, __x);

        dst[dex] = *((float4*)&tmpdst);
    }
}