/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _MATRIX_REDUCE_SUM_CUH_
#define _MATRIX_REDUCE_SUM_CUH_


#include "../Matrix_reduce.h"
#include "../../float_half_convert.h"
#include "../../../../classes/Matrix.h"
#include "../../../../classes/Vector.h"
#include "../../../../classes/GPU_Matrix.h"
#include "../../../../classes/GPU_Vector.h"


namespace decx
{
    namespace reduce
    {
        static void matrix_reduce2D_full_sum_fp32(decx::_Matrix* src, float* res);
        static void dev_matrix_reduce2D_full_sum_fp32(decx::_GPU_Matrix* src, float* res);


        template <bool _is_reduce_h>
        static void matrix_reduce2D_1way_sum_fp32(decx::_Matrix* src, decx::_Vector* dst);
    }
}


template <bool _is_reduce_h>
static void decx::reduce::matrix_reduce2D_1way_sum_fp32(decx::_Matrix* src, decx::_Vector* dst)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        Print_Error_Message(4, CUDA_EVENT_ACCESS_FAIL);
    }
    
    decx::reduce::cuda_reduce2D_1way_configs<float> _configs;
    _configs.generate_configs<_is_reduce_h>(make_uint2(src->Width(), src->Height()), S);
    
    decx::Ptr2D_Info<void> _dt1 = _configs.get_dtmp1();
    
    checkCudaErrors(cudaMemcpy2DAsync(_dt1._ptr.ptr,                    _dt1._dims.x * sizeof(float),
                                      src->Mat.ptr,                     src->Pitch() * sizeof(float),
                                      src->Width() * sizeof(float),     src->Height(),               
                                      cudaMemcpyHostToDevice,           S->get_raw_stream_ref()));

    if (_is_reduce_h) {
        decx::reduce::reduce_sum2D_h_fp32_Async(&_configs, S);
    }
    else {
        decx::reduce::reduce_sum2D_v_fp32_Async(&_configs, S);
    }

    checkCudaErrors(cudaMemcpyAsync(dst->Vec.ptr, _configs.get_leading_ptr(), dst->Len() * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();
}



static void decx::reduce::matrix_reduce2D_full_sum_fp32(decx::_Matrix* src, float* res)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        Print_Error_Message(4, CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    uint32_t nominated_alloc_x = decx::utils::ceil<uint32_t>(src->Width(), 4) * 4;

    decx::reduce::cuda_reduce1D_configs<float> _kp_configs;
    _kp_configs.generate_configs(nominated_alloc_x * src->Height(), S);

    checkCudaErrors(cudaMemcpy2DAsync(_kp_configs.get_dev_tmp1().ptr,   nominated_alloc_x * sizeof(float),
                                      src->Mat.ptr,                     src->Pitch() * sizeof(float),
                                      src->Width() * sizeof(float),     src->Height(),
                                      cudaMemcpyHostToDevice,           S->get_raw_stream_ref()));

    decx::reduce::reduce_sum2D_full_fp32_Async<false>(&_kp_configs, make_uint2(src->Width(), src->Height()), nominated_alloc_x / 4, S);

    checkCudaErrors(cudaMemcpyAsync(res, _kp_configs.get_leading_MIF().mem, 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    _kp_configs.release_buffer();
}



static void decx::reduce::dev_matrix_reduce2D_full_sum_fp32(decx::_GPU_Matrix* src, float* res)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        Print_Error_Message(4, CUDA_EVENT_ACCESS_FAIL);
        return;
    }

    uint32_t nominated_alloc_x = decx::utils::ceil<uint32_t>(src->Width(), 4) * 4 * 2;

    /*
    * For on-GPU process, the number of element that a block process of 1D is 2-times larger than
    * that of flatten kernel (e.g. float -> 256 * 4 (1D kernel) = 2 * ((32 * 4 * 8) (flatten))
    */
    decx::reduce::cuda_reduce1D_configs<float> _kp_configs;
    _kp_configs.generate_configs(src->Mat, nominated_alloc_x * src->Height(), S);

    decx::reduce::reduce_sum2D_full_fp32_Async<true>(&_kp_configs, make_uint2(src->Width(), src->Height()), src->Pitch() / 4, S);

    checkCudaErrors(cudaMemcpyAsync(res, _kp_configs.get_leading_MIF().mem, 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();

    _kp_configs.release_buffer();
}


#endif