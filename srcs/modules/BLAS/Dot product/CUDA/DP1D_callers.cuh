/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _DP1D_CALLERS_CUH_
#define _DP1D_CALLERS_CUH_


#include "../../../classes/Vector.h"
#include "../../../classes/GPU_Vector.h"
#include "../../../classes/DecxNumber.h"
#include "DP1D_config.cuh"



namespace decx
{
    namespace dot
    {
        /**
        * @param dev_A : The device pointer where vector A is stored
        * @param dev_B : The device pointer where vector B is stored
        * @param _actual_len : The actual length of the vector, measured in element
        * @param _kp_configs : The pointer of reduction summation configuration, don't need to be initialized
        * @param S : The pointer of CUDA stream
        *
        * @return The pointer where the result being stored
        */
        const void* cuda_DP1D_fp32_caller_Async(decx::dot::cuda_DP1D_configs<float>* _configs, decx::cuda_stream* S);


        const void* cuda_DP1D_fp16_caller_Async(decx::dot::cuda_DP1D_configs<de::Half>* _configs, decx::cuda_stream* S, const uint32_t _fp16_accu);


        const void* cuda_DP1D_fp64_caller_Async(decx::dot::cuda_DP1D_configs<double>* _configs, decx::cuda_stream* S);


        const void* cuda_DP1D_cplxf_caller_Async(decx::dot::cuda_DP1D_configs<double>* _configs, decx::cuda_stream* S);
    }
}





namespace decx
{
    namespace dot
    {
        static void vector_dot_fp32(decx::_Vector* A, decx::_Vector* B, de::DecxNumber* res);
        static void dev_vector_dot_fp32(decx::_GPU_Vector* A, decx::_GPU_Vector* B, de::DecxNumber* res);


        static void vector_dot_fp16(decx::_Vector* A, decx::_Vector* B, de::DecxNumber* res, const uint32_t _fp16_accu);
        static void dev_vector_dot_fp16(decx::_GPU_Vector* A, decx::_GPU_Vector* B, de::DecxNumber* res, const uint32_t _fp16_accu);


        static void vector_dot_fp64(decx::_Vector* A, decx::_Vector* B, de::DecxNumber* res);
        static void dev_vector_dot_fp64(decx::_GPU_Vector* A, decx::_GPU_Vector* B, de::DecxNumber* res);


        static void vector_dot_cplxf(decx::_Vector* A, decx::_Vector* B, de::DecxNumber* res);
        static void dev_vector_dot_cplxf(decx::_GPU_Vector* A, decx::_GPU_Vector* B, de::DecxNumber* res);
    }
}


static void decx::dot::vector_dot_fp32(decx::_Vector* A, decx::_Vector* B, de::DecxNumber* res)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        
        return;
    }

    decx::dot::cuda_DP1D_configs<float> _configs(A->Len(), S);

    checkCudaErrors(cudaMemcpyAsync(_configs._dev_A.ptr, A->Vec.ptr, A->Len() * sizeof(float), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));
    checkCudaErrors(cudaMemcpyAsync(_configs._dev_B.ptr, B->Vec.ptr, B->Len() * sizeof(float), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));

    const void* res_ptr = decx::dot::cuda_DP1D_fp32_caller_Async(&_configs, S);
    
    if (res_ptr == NULL) {
        Print_Error_Message(4, INTERNAL_ERROR);
        return;
    }
    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr(), res_ptr, 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_FP32_);

    E->event_record(S);
    E->synchronize();
}




static void decx::dot::dev_vector_dot_fp32(decx::_GPU_Vector* A, decx::_GPU_Vector* B, de::DecxNumber* res)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        
        return;
    }

    decx::dot::cuda_DP1D_configs<float> _configs(A->Vec, B->Vec, A->Len(), S);

    const void* res_ptr = decx::dot::cuda_DP1D_fp32_caller_Async(&_configs, S);

    if (res_ptr == NULL) {
        Print_Error_Message(4, INTERNAL_ERROR);
        return;
    }
    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr(), res_ptr, 1 * sizeof(float), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_FP32_);

    E->event_record(S);
    E->synchronize();
}




static void decx::dot::vector_dot_fp16(decx::_Vector* A, decx::_Vector* B, de::DecxNumber* res, const uint32_t _fp16_accu)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        
        return;
    }

    decx::dot::cuda_DP1D_configs<de::Half> _configs(A->Len(), S, _fp16_accu);

    checkCudaErrors(cudaMemcpyAsync(_configs._dev_A.ptr, A->Vec.ptr, A->Len() * sizeof(de::Half), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));
    checkCudaErrors(cudaMemcpyAsync(_configs._dev_B.ptr, B->Vec.ptr, B->Len() * sizeof(de::Half), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));

    const void* res_ptr = decx::dot::cuda_DP1D_fp16_caller_Async(&_configs, S, _fp16_accu);

    if (res_ptr == NULL) {
        Print_Error_Message(4, INTERNAL_ERROR);
        return;
    }

    uint8_t dst_ele_size = (_fp16_accu == decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1 ? sizeof(float) : sizeof(de::Half));
    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr(), res_ptr, 1 * dst_ele_size, cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(_fp16_accu == decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1 ? 
        de::_DATA_TYPES_FLAGS_::_FP32_ :
        de::_DATA_TYPES_FLAGS_::_FP16_);

    E->event_record(S);
    E->synchronize();
}




static void decx::dot::dev_vector_dot_fp16(decx::_GPU_Vector* A, decx::_GPU_Vector* B, de::DecxNumber* res, const uint32_t _fp16_accu)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        
        return;
    }

    decx::dot::cuda_DP1D_configs<de::Half> _configs(A->Vec, B->Vec, A->Len(), S, _fp16_accu);

    const void* res_ptr = decx::dot::cuda_DP1D_fp16_caller_Async(&_configs, S, _fp16_accu);

    if (res_ptr == NULL) {
        Print_Error_Message(4, INTERNAL_ERROR);
        return;
    }

    uint8_t dst_ele_size = (_fp16_accu == decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1 ? sizeof(float) : sizeof(de::Half));
    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr(), res_ptr, 1 * dst_ele_size, cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(_fp16_accu == decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1 ?
        de::_DATA_TYPES_FLAGS_::_FP32_ :
        de::_DATA_TYPES_FLAGS_::_FP16_);

    E->event_record(S);
    E->synchronize();
}




static void decx::dot::vector_dot_fp64(decx::_Vector* A, decx::_Vector* B, de::DecxNumber* res)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        
        return;
    }

    decx::dot::cuda_DP1D_configs<double> _configs(A->Len(), S);

    checkCudaErrors(cudaMemcpyAsync(_configs._dev_A.ptr, A->Vec.ptr, A->Len() * sizeof(double), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));
    checkCudaErrors(cudaMemcpyAsync(_configs._dev_B.ptr, B->Vec.ptr, B->Len() * sizeof(double), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));

    const void* res_ptr = decx::dot::cuda_DP1D_fp64_caller_Async(&_configs, S);

    if (res_ptr == NULL) {
        Print_Error_Message(4, INTERNAL_ERROR);
        return;
    }
    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr(), res_ptr, 1 * sizeof(double), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_FP64_);

    E->event_record(S);
    E->synchronize();
}




static void decx::dot::dev_vector_dot_fp64(decx::_GPU_Vector* A, decx::_GPU_Vector* B, de::DecxNumber* res)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        
        return;
    }

    decx::dot::cuda_DP1D_configs<double> _configs(A->Vec, B->Vec, A->Len(), S);

    const void* res_ptr = decx::dot::cuda_DP1D_fp64_caller_Async(&_configs, S);

    if (res_ptr == NULL) {
        Print_Error_Message(4, INTERNAL_ERROR);
        return;
    }
    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr(), res_ptr, 1 * sizeof(double), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_FP64_);

    E->event_record(S);
    E->synchronize();
}




static void decx::dot::vector_dot_cplxf(decx::_Vector* A, decx::_Vector* B, de::DecxNumber* res)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        
        return;
    }

    decx::dot::cuda_DP1D_configs<double> _configs(A->Len(), S);

    checkCudaErrors(cudaMemcpyAsync(_configs._dev_A.ptr, A->Vec.ptr, A->Len() * sizeof(de::CPf), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));
    checkCudaErrors(cudaMemcpyAsync(_configs._dev_B.ptr, B->Vec.ptr, B->Len() * sizeof(de::CPf), cudaMemcpyHostToDevice,
        S->get_raw_stream_ref()));

    const void* res_ptr = decx::dot::cuda_DP1D_cplxf_caller_Async(&_configs, S);

    if (res_ptr == NULL) {
        Print_Error_Message(4, INTERNAL_ERROR);
        return;
    }
    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr(), res_ptr, 1 * sizeof(de::CPf), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_);

    E->event_record(S);
    E->synchronize();
}




static void decx::dot::dev_vector_dot_cplxf(decx::_GPU_Vector* A, decx::_GPU_Vector* B, de::DecxNumber* res)
{
    decx::cuda_stream* S = NULL;
    S = decx::cuda::get_cuda_stream_ptr(cudaStreamNonBlocking);
    if (S == NULL) {
        
        return;
    }
    decx::cuda_event* E = NULL;
    E = decx::cuda::get_cuda_event_ptr(cudaEventBlockingSync);
    if (E == NULL) {
        
        return;
    }

    decx::dot::cuda_DP1D_configs<double> _configs(A->Vec, B->Vec, A->Len(), S);

    const void* res_ptr = decx::dot::cuda_DP1D_cplxf_caller_Async(&_configs, S);

    if (res_ptr == NULL) {
        Print_Error_Message(4, INTERNAL_ERROR);
        return;
    }
    checkCudaErrors(cudaMemcpyAsync(res->get_data_ptr(), res_ptr, 1 * sizeof(de::CPf), cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    res->set_type_flag(de::_DATA_TYPES_FLAGS_::_COMPLEX_F32_);

    E->event_record(S);
    E->synchronize();
}



#endif