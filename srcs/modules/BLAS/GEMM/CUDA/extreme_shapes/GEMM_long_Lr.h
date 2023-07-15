/**
*    ---------------------------------------------------------------------
*    Author : Wayne anderson
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright © Wayne,
*    2021.04.16
*/

#ifndef _GEMM_LONG_LR_H_
#define _GEMM_LONG_LR_H_

#include "../../../classes/GPU_Matrix.h"
#include "GEMM_long_linear_region.cuh"

using decx::_GPU_Matrix;

namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH GEMM_Long_Lr(de::GPU_Matrix<float>& A, de::GPU_Matrix<float>& B, de::GPU_Matrix<float>& dst);
    }
}


de::DH de::cuda::GEMM_Long_Lr(de::GPU_Matrix<float>& A, de::GPU_Matrix<float>& B, de::GPU_Matrix<float>& dst)
{
    de::DH handle;

    _GPU_Matrix<float>* _A = dynamic_cast<_GPU_Matrix<float>*>(&A);
    _GPU_Matrix<float>* _B = dynamic_cast<_GPU_Matrix<float>*>(&B);
    _GPU_Matrix<float>* _dst = dynamic_cast<_GPU_Matrix<float>*>(&dst);

    uint float4_pitch_A = _A->pitch / 4;
    uint float4_pitch_B = _B->pitch / 4;

    dim3 grid(_dst->height / 16, float4_pitch_B / 16);
    dim3 block(256);

    cu_GEMM_LongLrWB_fp32<<<grid, block>>>(
        reinterpret_cast<float4*>(_A->Mat.ptr),
        reinterpret_cast<float4*>(_B->Mat.ptr),
        reinterpret_cast<float4*>(_dst->Mat.ptr),
        float4_pitch_A,
        float4_pitch_B,
        _A->pitch / 64);

    return handle;
}


#endif