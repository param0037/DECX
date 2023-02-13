/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _GEMM_FP16_KERNEL_CALLERS_H_
#define _GEMM_FP16_KERNEL_CALLERS_H_

#include "GEMM_fp16.cuh"
#include "../../../../classes/GPU_Matrix.h"
#include "GEMM_fp16_accurate.cuh"
#include "../../../GEMM_utils.h"
#include "../../../../core/cudaStream_management/cudaStream_queue.h"


namespace decx
{
    /**
    * @param DA, DB, Ddst are the device memories with dimensions that already fit in the kernel demands
    * --x128 aligned on both hA and wB and x16 aligned on K (wA = hB)
    * @param pitch_A, pitch_B are the widths of DA and DB (true widths) (in float)
    * @param hA is the height of DA (x128 aligned)
    */
    static void hGEMM_part(de::Half* DA, de::Half* DB, de::Half* Ddst,
        const int pitch_A, const int pitch_B, const int hA, decx::cuda_stream* S);


    /**
    * @param DA, DB, Ddst are the device memories with dimensions that already fit in the kernel demands
    * --x128 aligned on both hA and wB and x16 aligned on K (wA = hB)
    * @param pitch_A, pitch_B are the widths of DA and DB (true widths) (in float)
    * @param hA is the height of DA (x128 aligned)
    */
    static void hGEMM_part_ABC(de::Half* DA, de::Half* DB, de::Half* DC, de::Half* Ddst,
        const int pitch_A, const int pitch_B, const int hA, decx::cuda_stream* S);



    static void hGEMM_part_accu(de::Half* DA, de::Half* DB, de::Half* Ddst,
        const int pitch_A, const int pitch_B, const int hA, decx::cuda_stream* S);



    static void hGEMM_part_ABC_accu(de::Half* DA, de::Half* DB, de::Half* DC, de::Half* Ddst,
        const int pitch_A, const int pitch_B, const int hA, decx::cuda_stream* S);



    static void hGEMM_caller_overall(de::Half* DA, de::Half* DB, de::Half* Ddst,
        const int pitch_A, const int pitch_B, const int hA, decx::cuda_stream* S, const int flag);



    static void hGEMM_caller_ABC_overall(de::Half* DA, de::Half* DB, de::Half* DC, de::Half* Ddst,
        const int pitch_A, const int pitch_B, const int hA, decx::cuda_stream* S, const int flag);



    static void dev_hGEMM_part(
        _GPU_Matrix* _A, _GPU_Matrix* _B, _GPU_Matrix* _dst, decx::cuda_stream* S);


    static void dev_hGEMM_part_accu(
        _GPU_Matrix* _A, _GPU_Matrix* _B, _GPU_Matrix* _dst, decx::cuda_stream* S);


    static void dev_hGEMM_caller_overall(
        _GPU_Matrix* _A, _GPU_Matrix* _B, _GPU_Matrix* _dst, decx::cuda_stream* S, const int flag);


    static void dev_hGEMM_part_ABC(
        _GPU_Matrix* _A, _GPU_Matrix* _B, _GPU_Matrix* _C, _GPU_Matrix* _dst, decx::cuda_stream* S);


    static void dev_hGEMM_part_ABC_accu(
        _GPU_Matrix* _A, _GPU_Matrix* _B, _GPU_Matrix* _C, _GPU_Matrix* _dst, decx::cuda_stream* S);


    static void dev_hGEMM_ABC_caller_overall(
        _GPU_Matrix* _A, _GPU_Matrix* _B, _GPU_Matrix* _C, _GPU_Matrix* _dst,
        decx::cuda_stream* S, const int flag);
}



static
void decx::hGEMM_part(de::Half* DA, de::Half* DB, de::Half* Ddst,
    const int pitch_A, const int pitch_B, const int hA, decx::cuda_stream* S)
{
    int threads = GEMM_BlockDim * GEMM_BlockDim;
    dim3 grid(hA / (GEMM_BlockDim * 8), pitch_B / (GEMM_BlockDim * 8));

    const int __iter = pitch_A / GEMM_BlockDim;

    decx::gemm::GPUK::cu_GEMM_fp16_spec << <grid, threads, 0, S->get_raw_stream_ref() >> > (
        reinterpret_cast<float4*>(DA),
        reinterpret_cast<float4*>(DB),
        reinterpret_cast<float4*>(Ddst), pitch_A / 8, pitch_B / 8, __iter);
}




static
void decx::hGEMM_part_ABC(de::Half* DA, de::Half* DB, de::Half* DC, de::Half* Ddst,
    const int pitch_A, const int pitch_B, const int hA, decx::cuda_stream* S)
{
    int threads = GEMM_BlockDim * GEMM_BlockDim;
    dim3 grid(hA / (GEMM_BlockDim * 8), pitch_B / (GEMM_BlockDim * 8));

    const int __iter = pitch_A / GEMM_BlockDim;

    decx::gemm::GPUK::cu_GEMM_fp16_ABC_spec << <grid, threads, 0, S->get_raw_stream_ref() >> > (
        reinterpret_cast<float4*>(DA),
        reinterpret_cast<float4*>(DB),
        reinterpret_cast<float4*>(DC),
        reinterpret_cast<float4*>(Ddst), pitch_A / 8, pitch_B / 8, __iter);
}



static
void decx::hGEMM_part_ABC_accu(de::Half* DA, de::Half* DB, de::Half* DC, de::Half* Ddst,
    const int pitch_A, const int pitch_B, const int hA, decx::cuda_stream* S)
{
    int threads = GEMM_BlockDim * GEMM_BlockDim;
    dim3 grid(hA / (GEMM_BlockDim * 8), pitch_B / (GEMM_BlockDim * 8));

    const int __iter = pitch_A / GEMM_BlockDim;

    decx::gemm::GPUK::cu_GEMM_fp16_ABC_spec_accu << <grid, threads, 0, S->get_raw_stream_ref() >> > (
        reinterpret_cast<float4*>(DA),
        reinterpret_cast<float4*>(DB),
        reinterpret_cast<float4*>(DC),
        reinterpret_cast<float4*>(Ddst), pitch_A / 8, pitch_B / 8, __iter);
}


static
void decx::hGEMM_part_accu(de::Half* DA, de::Half* DB, de::Half* Ddst,
    const int pitch_A, const int pitch_B, const int hA, decx::cuda_stream* S)
{
    int threads = GEMM_BlockDim * GEMM_BlockDim;
    dim3 grid(hA / (GEMM_BlockDim * 8), pitch_B / (GEMM_BlockDim * 8));

    const int __iter = pitch_A / GEMM_BlockDim;

    decx::gemm::GPUK::cu_GEMM_fp16_spec_accu << <grid, threads, 0, S->get_raw_stream_ref() >> > (
        reinterpret_cast<float4*>(DA),
        reinterpret_cast<float4*>(DB),
        reinterpret_cast<float4*>(Ddst), pitch_A / 8, pitch_B / 8, __iter);
}



static void decx::hGEMM_caller_overall(de::Half* DA, de::Half* DB, de::Half* Ddst,
    const int pitch_A, const int pitch_B, const int hA, decx::cuda_stream* S, const int flag)
{
    switch (flag)
    {
    case de::GEMM_properties::HALF_GEMM_DIRECT:
        decx::hGEMM_part(DA, DB, Ddst, pitch_A, pitch_B, hA, S);
        break;

    case de::GEMM_properties::HALF_GEMM_ACCURATE:
        decx::hGEMM_part_accu(DA, DB, Ddst, pitch_A, pitch_B, hA, S);
        break;
    default:
        break;
    }
}


static void decx::hGEMM_caller_ABC_overall(de::Half* DA, de::Half* DB, de::Half* DC, de::Half* Ddst,
    const int pitch_A, const int pitch_B, const int hA, decx::cuda_stream* S, const int flag)
{
    switch (flag)
    {
    case de::GEMM_properties::HALF_GEMM_DIRECT:
        decx::hGEMM_part_ABC(DA, DB, DC, Ddst, pitch_A, pitch_B, hA, S);
        break;

    case de::GEMM_properties::HALF_GEMM_ACCURATE:
        decx::hGEMM_part_ABC_accu(DA, DB, DC, Ddst, pitch_A, pitch_B, hA, S);
        break;
    default:
        break;
    }
}



static void decx::dev_hGEMM_part(
    _GPU_Matrix* _A, _GPU_Matrix* _B, _GPU_Matrix* _dst, decx::cuda_stream* S)
{
    const uint block = 256;
    const dim3 grid(decx::utils::ceil<uint>(_A->height, 16 * 8), decx::utils::ceil<uint>(_B->pitch, 16 * 8));

    if (_B->pitch % 128 || _A->height % 128) {
        if (_B->height % 16) {
            decx::gemm::GPUK::cu_GEMM_fp16_anyWH_anyL << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_dst->Mat.ptr), _A->pitch / 8, _B->pitch / 8, _A->height, _B->height,
                decx::utils::ceil<uint>(_A->pitch, 16));
        }
        else {
            decx::gemm::GPUK::cu_GEMM_fp16_anyWH_specL << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_dst->Mat.ptr), _A->pitch / 8, _B->pitch / 8, _A->height, _A->pitch / 16);
        }
    }
    else {
        if (_B->height % 16) {
            decx::gemm::GPUK::cu_GEMM_fp16_specWH_anyL << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_dst->Mat.ptr), _A->pitch / 8, _B->pitch / 8, _B->height,
                decx::utils::ceil<uint>(_A->pitch, 16));
        }
        else {
            decx::gemm::GPUK::cu_GEMM_fp16_spec << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_dst->Mat.ptr), _A->pitch / 8, _B->pitch / 8, _A->pitch / 16);
        }
    }
}



static void decx::dev_hGEMM_part_accu(
    _GPU_Matrix* _A, _GPU_Matrix* _B, _GPU_Matrix* _dst, decx::cuda_stream* S)
{
    const uint block = 256;
    const dim3 grid(decx::utils::ceil<uint>(_A->height, 16 * 8), decx::utils::ceil<uint>(_B->pitch, 16 * 8));

    if (_B->pitch % 128 || _A->height % 128) {
        if (_B->height % 16) {
            decx::gemm::GPUK::cu_GEMM_fp16_anyWH_anyL_accu << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_dst->Mat.ptr), _A->pitch / 8, _B->pitch / 8, _A->height, _B->height,
                decx::utils::ceil<uint>(_A->pitch, 16));
        }
        else {
            decx::gemm::GPUK::cu_GEMM_fp16_anyWH_specL_accu << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_dst->Mat.ptr), _A->pitch / 8, _B->pitch / 8, _A->height, _A->pitch / 16);
        }
    }
    else {
        if (_B->height % 16) {
            decx::gemm::GPUK::cu_GEMM_fp16_specWH_anyL_accu << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_dst->Mat.ptr), _A->pitch / 8, _B->pitch / 8, _B->height,
                decx::utils::ceil<uint>(_A->pitch, 16));
        }
        else {
            decx::gemm::GPUK::cu_GEMM_fp16_spec_accu << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_dst->Mat.ptr), _A->pitch / 8, _B->pitch / 8, _A->pitch / 16);
        }
    }
}



static
void decx::dev_hGEMM_caller_overall(
    _GPU_Matrix* _A, _GPU_Matrix* _B, _GPU_Matrix* _dst, decx::cuda_stream* S, const int flag)
{
    switch (flag)
    {
    case de::GEMM_properties::HALF_GEMM_DIRECT:
        decx::dev_hGEMM_part(_A, _B, _dst, S);
        break;

    case de::GEMM_properties::HALF_GEMM_ACCURATE:
        decx::dev_hGEMM_part_accu(_A, _B, _dst, S);
        break;
    default:
        break;
    }
}




static void decx::dev_hGEMM_part_ABC(
    _GPU_Matrix* _A, _GPU_Matrix* _B, _GPU_Matrix* _C, _GPU_Matrix* _dst, decx::cuda_stream* S)
{
    const uint block = 256;
    const dim3 grid(decx::utils::ceil<uint>(_A->height, 16 * 8), decx::utils::ceil<uint>(_B->pitch, 16 * 8));

    if (_B->pitch % 128 || _A->height % 128) {
        if (_B->height % 16) {
            decx::gemm::GPUK::cu_GEMM_fp16_ABC_anyWH_anyL << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_C->Mat.ptr), reinterpret_cast<float4*>(_dst->Mat.ptr),
                _A->pitch / 8, _B->pitch / 8, _A->height, _B->height,
                decx::utils::ceil<uint>(_A->pitch, 16));
        }
        else {
            decx::gemm::GPUK::cu_GEMM_fp16_ABC_anyWH_specL << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_C->Mat.ptr), reinterpret_cast<float4*>(_dst->Mat.ptr),
                _A->pitch / 8, _B->pitch / 8, _A->height, _A->pitch / 16);
        }
    }
    else {
        if (_B->height % 16) {
            decx::gemm::GPUK::cu_GEMM_fp16_ABC_specWH_anyL << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_C->Mat.ptr), reinterpret_cast<float4*>(_dst->Mat.ptr),
                _A->pitch / 8, _B->pitch / 8, _B->height,
                decx::utils::ceil<uint>(_A->pitch, 16));
        }
        else {
            decx::gemm::GPUK::cu_GEMM_fp16_ABC_spec << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_C->Mat.ptr), reinterpret_cast<float4*>(_dst->Mat.ptr),
                _A->pitch / 8, _B->pitch / 8, _A->pitch / 16);
        }
    }
}



static void decx::dev_hGEMM_part_ABC_accu(_GPU_Matrix* _A, _GPU_Matrix* _B,
    _GPU_Matrix* _C, _GPU_Matrix* _dst, decx::cuda_stream* S)
{
    const uint block = 256;
    const dim3 grid(decx::utils::ceil<uint>(_A->height, 16 * 8), decx::utils::ceil<uint>(_B->pitch, 16 * 8));

    if (_B->pitch % 128 || _A->height % 128) {
        if (_B->height % 16) {
            decx::gemm::GPUK::cu_GEMM_fp16_ABC_anyWH_anyL_accu << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_C->Mat.ptr), reinterpret_cast<float4*>(_dst->Mat.ptr),
                _A->pitch / 8, _B->pitch / 8, _A->height, _B->height,
                decx::utils::ceil<uint>(_A->pitch, 16));
        }
        else {
            decx::gemm::GPUK::cu_GEMM_fp16_ABC_anyWH_specL_accu << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_C->Mat.ptr), reinterpret_cast<float4*>(_dst->Mat.ptr),
                _A->pitch / 8, _B->pitch / 8, _A->height, _A->pitch / 16);
        }
    }
    else {
        if (_B->height % 16) {
            decx::gemm::GPUK::cu_GEMM_fp16_ABC_specWH_anyL_accu << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_C->Mat.ptr), reinterpret_cast<float4*>(_dst->Mat.ptr),
                _A->pitch / 8, _B->pitch / 8, _B->height,
                decx::utils::ceil<uint>(_A->pitch, 16));
        }
        else {
            decx::gemm::GPUK::cu_GEMM_fp16_ABC_spec_accu << <grid, block, 0, S->get_raw_stream_ref() >> > (
                reinterpret_cast<float4*>(_A->Mat.ptr), reinterpret_cast<float4*>(_B->Mat.ptr),
                reinterpret_cast<float4*>(_C->Mat.ptr), reinterpret_cast<float4*>(_dst->Mat.ptr),
                _A->pitch / 8, _B->pitch / 8, _A->pitch / 16);
        }
    }
}



static void decx::dev_hGEMM_ABC_caller_overall(
    _GPU_Matrix* _A, _GPU_Matrix* _B, _GPU_Matrix* _C, _GPU_Matrix* _dst,
    decx::cuda_stream* S, const int flag)
{
    switch (flag)
    {
    case de::GEMM_properties::HALF_GEMM_DIRECT:
        decx::dev_hGEMM_part_ABC(_A, _B, _C, _dst, S);
        break;

    case de::GEMM_properties::HALF_GEMM_ACCURATE:
        decx::dev_hGEMM_part_ABC_accu(_A, _B, _C, _dst, S);
        break;
    default:
        break;
    }
}



#endif
