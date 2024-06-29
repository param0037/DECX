/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/


#ifndef _GEMM_FP32_KERNEL_CALLERS_H_
#define _GEMM_FP32_KERNEL_CALLERS_H_

#include "GEMM_fp32.cuh"
#include "../fp16/GEMM_fp16_accurate.cuh"
#include "../../../GEMM_utils.h"
#include "../../../../../classes/GPU_Matrix.h"
#include "../../../../../core/cudaStream_management/cudaStream_queue.h"
#include "../../../../../core/cudaStream_management/cudaEvent_queue.h"


namespace decx
{
    /**
    * @param DA, DB, Ddst are the device memories with dimensions that already fit in the kernel demands
    * --x128 aligned on both hA and wB and x16 aligned on K (wA = hB)
    * @param pitch_A(x16 aligned), pitch_B(x128 aligned) are the widths of DA and DB (true widths) (in float)
    * @param hA is the height of DA (x128 aligned)
    */
    static void sGEMM_part(float* DA, float* DB, float* Ddst,
        const int pitch_A, const int pitch_B, const int hA, decx::cuda_stream* S);


    /**
    * @param DA, DB, Ddst are the device memories with dimensions that already fit in the kernel demands
    * --x128 aligned on both hA and wB and x16 aligned on K (wA = hB)
    * @param pitch_A(x16 aligned), pitch_B(x128 aligned) are the widths of DA and DB (true widths) (in float)
    * @param hA is the height of DA (x128 aligned)
    */
    static void sGEMM_part_ABC(float* DA, float* DB, float* DC, float* Ddst,
        const int pitch_A, const int pitch_B, const int hA, decx::cuda_stream* S);



    static void dev_sGEMM_part(_GPU_Matrix* _A, _GPU_Matrix* _B, _GPU_Matrix* _dst,
        decx::cuda_stream* S);



    static void dev_sGEMM_part_ABC(_GPU_Matrix* _A, _GPU_Matrix* _B, _GPU_Matrix* _C, 
        _GPU_Matrix* _dst, decx::cuda_stream* S);
}



static
void decx::sGEMM_part(float* DA, float* DB, float* Ddst,
    const int pitch_A, const int pitch_B, const int hA, decx::cuda_stream* S)
{
    int threads = GEMM_BlockDim * GEMM_BlockDim;
    dim3 grid(hA / (GEMM_BlockDim * 8), pitch_B / (GEMM_BlockDim * 8));

    const uint __iter = pitch_A / GEMM_BlockDim;

    decx::gemm::GPUK::cu_GEMM_fp32_spec << <grid, threads, 0, S->get_raw_stream_ref() >> > (
        reinterpret_cast<float4*>(DA),
        reinterpret_cast<float4*>(DB),
        reinterpret_cast<float4*>(Ddst), pitch_A / 4, pitch_B / 4, __iter);
}




static
void decx::sGEMM_part_ABC(float* DA, float* DB, float* DC, float* Ddst,
    const int pitch_A, const int pitch_B, const int hA, decx::cuda_stream* S)
{
    int threads = GEMM_BlockDim * GEMM_BlockDim;
    dim3 grid(hA / (GEMM_BlockDim * 8), pitch_B / (GEMM_BlockDim * 8));

    const uint __iter = pitch_A / GEMM_BlockDim;

    decx::gemm::GPUK::cu_GEMM_fp32_ABC_spec << <grid, threads, 0, S->get_raw_stream_ref() >> > (
        reinterpret_cast<float4*>(DA),
        reinterpret_cast<float4*>(DB),
        reinterpret_cast<float4*>(DC),
        reinterpret_cast<float4*>(Ddst), pitch_A / 4, pitch_B / 4, __iter);
}


static void decx::dev_sGEMM_part(_GPU_Matrix* _A, _GPU_Matrix* _B, _GPU_Matrix* _dst,
    decx::cuda_stream* S)
{
    const uint block = 256;
    const dim3 grid(decx::utils::ceil<uint>(_A->Height(), 16 * 8), decx::utils::ceil<uint>(_B->Pitch(), 16 * 8));

    if ((_B->Pitch() % 128) || (_A->Height() % 128)) {        // dstdims CAN NOT be divided into integers
        if (_B->Height() % 16) {
            decx::gemm::GPUK::cu_GEMM_fp32_anyWH_anyL << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_dst->Mat.ptr), _A->Pitch() / 4, _B->Pitch() / 4, _A->Height(), _B->Height(),
                decx::utils::ceil<uint>(_A->Pitch(), 16));
        }
        else {
            decx::gemm::GPUK::cu_GEMM_fp32_anyWH_specL << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_dst->Mat.ptr), _A->Pitch() / 4, _B->Pitch() / 4, _A->Height(), _A->Pitch() / 16);
        }
    }
    else {
        if (_B->Height() % 16) {
            decx::gemm::GPUK::cu_GEMM_fp32_specWH_anyL << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_dst->Mat.ptr), _A->Pitch() / 4, _B->Pitch() / 4, _B->Height(),
                decx::utils::ceil<uint>(_A->Pitch(), 16));
        }
        else {
            decx::gemm::GPUK::cu_GEMM_fp32_spec << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_dst->Mat.ptr), _A->Pitch() / 4, _B->Pitch() / 4, _B->Height() / 16);
        }
    }
}



static void decx::dev_sGEMM_part_ABC(_GPU_Matrix* _A, _GPU_Matrix* _B, _GPU_Matrix* _C,
    _GPU_Matrix* _dst, decx::cuda_stream* S)
{
    const uint block = 256;
    const dim3 grid(decx::utils::ceil<uint>(_A->Height(), 16 * 8), decx::utils::ceil<uint>(_B->Pitch(), 16 * 8));

    if ((_B->Pitch() % 128) || (_A->Height() % 128)) {        // dstdims CAN NOT be divided into integers
        if (_B->Height() % 16) {
            decx::gemm::GPUK::cu_GEMM_fp32_ABC_anyWH_anyL << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_C->Mat.ptr), reinterpret_cast<float4*>(_dst->Mat.ptr),
                _A->Pitch() / 4, _B->Pitch() / 4, _A->Height(), _B->Height(),
                decx::utils::ceil<uint>(_A->Pitch(), 16));
        }
        else {
            decx::gemm::GPUK::cu_GEMM_fp32_ABC_anyWH_specL << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_C->Mat.ptr), reinterpret_cast<float4*>(_dst->Mat.ptr),
                _A->Pitch() / 4, _B->Pitch() / 4, _A->Height(), _A->Pitch() / 16);
        }
    }
    else {
        if (_B->Height() % 16) {
            decx::gemm::GPUK::cu_GEMM_fp32_ABC_specWH_anyL << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_C->Mat.ptr), reinterpret_cast<float4*>(_dst->Mat.ptr),
                _A->Pitch() / 4, _B->Pitch() / 4, _B->Height(),
                decx::utils::ceil<uint>(_A->Pitch(), 16));
        }
        else {
            decx::gemm::GPUK::cu_GEMM_fp32_ABC_spec << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_C->Mat.ptr), reinterpret_cast<float4*>(_dst->Mat.ptr),
                _A->Pitch() / 4, _B->Pitch() / 4, _B->Height() / 16);
        }
    }
}



#endif