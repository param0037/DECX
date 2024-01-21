/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _CP_ADD_KERNEL_CUH_
#define _CP_ADD_KERNEL_CUH_

#include "../../core/basic.h"


__device__ __inline__
float4 _complexf_4_add(float4 _A, float4 _B)
{
    float4 res;
    res.x = __fadd_rn(_A.x, _B.x);
    res.y = __fadd_rn(_A.y, _B.y);
    res.z = __fadd_rn(_A.z, _B.z);
    res.w = __fadd_rn(_A.w, _B.w);
}


__global__
/**
* int* x2, add together
* @param len : have considered vec4
*/
void cpf_add_m_vec4(float4* A, float4* B, float4* dst, const size_t len)
{
    size_t tid = threadIdx.x + blockDim.x * blockIdx.x;
    float4 tmpA, tmpB, tmpdst;

    if (tid < len) {
        tmpA = A[tid];
        tmpB = B[tid];

        tmpdst = _complexf_4_add(tmpA, tmpB);

        dst[tid] = tmpdst;
    }
}


#endif