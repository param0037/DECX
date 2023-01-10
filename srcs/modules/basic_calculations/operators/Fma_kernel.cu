/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "Fma_kernel.cuh"


__global__
/**
* int* x2, add together
* @param len : have considered vec4
*/
void decx::calc::GPUK::fma_m_ivec4(float4* A, float4* B, float4* C, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    int4 tmpA, tmpB, tmpC, tmpdst;

    if (tid < len) {
        tmpA = *((int4*)&A[tid]);
        tmpB = *((int4*)&B[tid]);
        tmpC = *((int4*)&C[tid]);

        tmpdst.x = tmpA.x * tmpB.x + tmpC.x;
        tmpdst.y = tmpA.y * tmpB.y + tmpC.y;
        tmpdst.z = tmpA.z * tmpB.z + tmpC.z;
        tmpdst.w = tmpA.w * tmpB.w + tmpC.w;

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
void decx::calc::GPUK::fma_m_ivec4_2D(float4* A, float4* B, float4* C, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    int4 tmpA, tmpB, tmpC, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        *((float4*)&tmpA) = A[dex];
        *((float4*)&tmpB) = B[dex];
        *((float4*)&tmpC) = C[dex];

        tmpdst.x = tmpA.x * tmpB.x + tmpC.x;
        tmpdst.y = tmpA.y * tmpB.y + tmpC.y;
        tmpdst.z = tmpA.z * tmpB.z + tmpC.z;
        tmpdst.w = tmpA.w * tmpB.w + tmpC.w;

        dst[dex] = *((float4*)&tmpdst);
    }
}



__global__
/**
* int* x2, add together
* @param len : have considered vec4
*/
void decx::calc::GPUK::fma_m_fvec4(float4* A, float4* B, float4* C, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    float4 tmpA, tmpB, tmpC, tmpdst;

    if (tid < len) {
        tmpA = A[tid];
        tmpB = B[tid];
        tmpC = C[tid];

        tmpdst.x = fmaf(tmpA.x, tmpB.x, tmpC.x);
        tmpdst.y = fmaf(tmpA.y, tmpB.y, tmpC.y);
        tmpdst.z = fmaf(tmpA.z, tmpB.z, tmpC.z);
        tmpdst.w = fmaf(tmpA.w, tmpB.w, tmpC.w);

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
void decx::calc::GPUK::fma_m_fvec4_2D(float4* A, float4* B, float4* C, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    float4 tmpA, tmpB, tmpC, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        tmpA = A[dex];
        tmpB = B[dex];
        tmpC = C[dex];

        tmpdst.x = fmaf(tmpA.x, tmpB.x, tmpC.x);
        tmpdst.y = fmaf(tmpA.y, tmpB.y, tmpC.y);
        tmpdst.z = fmaf(tmpA.z, tmpB.z, tmpC.z);
        tmpdst.w = fmaf(tmpA.w, tmpB.w, tmpC.w);

        dst[dex] = tmpdst;
    }
}


__global__
void decx::calc::GPUK::fma_m_hvec8(float4* A, float4* B, float4* C, float4* dst, const size_t len)
{
#if __ABOVE_SM_53
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    half2_8 tmpA, tmpB, tmpC, tmpdst;

    if (tid < len) {
        tmpA = *((half2_8*)&A[tid]);
        tmpB = *((half2_8*)&B[tid]);
        tmpC = *((half2_8*)&C[tid]);

        tmpdst.x = __hfma2(tmpA.x, tmpB.x, tmpC.x);
        tmpdst.y = __hfma2(tmpA.y, tmpB.y, tmpC.y);
        tmpdst.z = __hfma2(tmpA.z, tmpB.z, tmpC.z);
        tmpdst.w = __hfma2(tmpA.w, tmpB.w, tmpC.w);

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
void decx::calc::GPUK::fma_m_hvec8_2D(float4* A, float4* B, float4* C, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
#if __ABOVE_SM_53
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    half2_8 tmpA, tmpB, tmpC, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        *((float4*)&tmpA) = A[dex];
        *((float4*)&tmpB) = B[dex];
        *((float4*)&tmpC) = C[dex];

        tmpdst.x = __hfma2(tmpA.x, tmpB.x, tmpC.x);
        tmpdst.y = __hfma2(tmpA.y, tmpB.y, tmpC.y);
        tmpdst.z = __hfma2(tmpA.z, tmpB.z, tmpC.z);
        tmpdst.w = __hfma2(tmpA.w, tmpB.w, tmpC.w);

        dst[dex] = *((float4*)&tmpdst);
    }
#endif
}



__global__
/**
* int* x2, add together
* @param len : have considered vec4
*/
void decx::calc::GPUK::fma_m_dvec2(float4* A, float4* B, float4* C, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    double2 tmpA, tmpB, tmpC, tmpdst;

    if (tid < len) {
        tmpA = *((double2*)&A[tid]);
        tmpB = *((double2*)&B[tid]);
        tmpC = *((double2*)&C[tid]);

        tmpdst.x = fma(tmpA.x, tmpB.x, tmpC.x);
        tmpdst.y = fma(tmpA.y, tmpB.y, tmpC.y);

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
void decx::calc::GPUK::fma_m_dvec2_2D(float4* A, float4* B, float4* C, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    double2 tmpA, tmpB, tmpC, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        tmpA = *((double2*)&A[dex]);
        tmpB = *((double2*)&B[dex]);
        tmpC = *((double2*)&C[dex]);

        tmpdst.x = fma(tmpA.x, tmpB.x, tmpC.x);
        tmpdst.y = fma(tmpA.y, tmpB.y, tmpC.y);

        dst[dex] = *((float4*)&tmpdst);
    }
}



// ----------------------------- C --------------------------------------


__global__
void decx::calc::GPUK::fma_c_ivec4(float4* A, int __x, float4* B, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    int4 tmpA, tmpB, tmpdst;

    if (tid < len) {
        tmpA = *((int4*)&A[tid]);
        tmpB = *((int4*)&B[tid]);

        tmpdst.x = tmpA.x * __x + tmpB.x;
        tmpdst.y = tmpA.y * __x + tmpB.y;
        tmpdst.z = tmpA.z * __x + tmpB.z;
        tmpdst.w = tmpA.w * __x + tmpB.w;

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
void decx::calc::GPUK::fma_c_ivec4_2D(float4* A, int __x, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    int4 tmpA, tmpB, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        *((float4*)&tmpA) = A[dex];
        *((float4*)&tmpB) = B[dex];

        tmpdst.x = tmpA.x * __x + tmpB.x;
        tmpdst.y = tmpA.y * __x + tmpB.y;
        tmpdst.z = tmpA.z * __x + tmpB.z;
        tmpdst.w = tmpA.w * __x + tmpB.w;

        dst[dex] = *((float4*)&tmpdst);
    }
}



__global__
void decx::calc::GPUK::fma_c_fvec4(float4* A, float __x, float4* B, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    float4 tmpA, tmpB, tmpdst;

    if (tid < len) {
        tmpA = A[tid];
        tmpB = B[tid];

        tmpdst.x = fmaf(tmpA.x, __x, tmpB.x);
        tmpdst.y = fmaf(tmpA.y, __x, tmpB.y);
        tmpdst.z = fmaf(tmpA.z, __x, tmpB.z);
        tmpdst.w = fmaf(tmpA.w, __x, tmpB.w);

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
void decx::calc::GPUK::fma_c_fvec4_2D(float4* A, float __x, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    float4 tmpA, tmpB, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        tmpA = A[dex];
        tmpB = B[dex];

        tmpdst.x = fmaf(tmpA.x, __x, tmpB.x);
        tmpdst.y = fmaf(tmpA.y, __x, tmpB.y);
        tmpdst.z = fmaf(tmpA.z, __x, tmpB.z);
        tmpdst.w = fmaf(tmpA.w, __x, tmpB.w);

        dst[dex] = tmpdst;
    }
}



__global__
void decx::calc::GPUK::fma_c_hvec8(float4* A, half2 __x, float4* B, float4* dst, const size_t len)
{
#if __ABOVE_SM_53
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    half2_8 tmpA, tmpB, tmpdst;

    if (tid < len) {
        tmpA = *((half2_8*)&A[tid]);
        tmpB = *((half2_8*)&B[tid]);

        tmpdst.x = __hfma2(tmpA.x, __x, tmpB.x);
        tmpdst.y = __hfma2(tmpA.y, __x, tmpB.y);
        tmpdst.z = __hfma2(tmpA.z, __x, tmpB.z);
        tmpdst.w = __hfma2(tmpA.w, __x, tmpB.w);

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
void decx::calc::GPUK::fma_c_hvec8_2D(float4* A, half2 __x, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
#if __ABOVE_SM_53
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    half2_8 tmpA, tmpB, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        *((float4*)&tmpA) = A[dex];
        *((float4*)&tmpB) = B[dex];

        tmpdst.x = __hfma2(tmpA.x, __x, tmpB.x);
        tmpdst.y = __hfma2(tmpA.y, __x, tmpB.y);
        tmpdst.z = __hfma2(tmpA.z, __x, tmpB.z);
        tmpdst.w = __hfma2(tmpA.w, __x, tmpB.w);

        dst[dex] = *((float4*)&tmpdst);
    }
#endif
}



__global__
void decx::calc::GPUK::fma_c_dvec2(float4* A, double __x, float4* B, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    double2 tmpA, tmpB, tmpdst;

    if (tid < len) {
        tmpA = *((double2*)&A[tid]);
        tmpB = *((double2*)&B[tid]);

        tmpdst.x = fma(tmpA.x, __x, tmpB.x);
        tmpdst.y = fma(tmpA.y, __x, tmpB.y);

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
void decx::calc::GPUK::fma_c_dvec2_2D(float4* A, double __x, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds)
{
    uint tidx = threadIdx.x + blockIdx.x * blockDim.x;
    uint tidy = threadIdx.y + blockIdx.y * blockDim.y;

    size_t dex = (size_t)tidx * eq_pitch + (size_t)tidy;
    double2 tmpA, tmpB, tmpdst;

    if (tidx < bounds.y && tidy < bounds.x) {
        *((float4*)&tmpA) = A[dex];
        *((float4*)&tmpB) = B[dex];

        tmpdst.x = fma(tmpA.x, __x, tmpB.x);
        tmpdst.y = fma(tmpA.y, __x, tmpB.y);

        dst[dex] = *((float4*)&tmpdst);
    }
}