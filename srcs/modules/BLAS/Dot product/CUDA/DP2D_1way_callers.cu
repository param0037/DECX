/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "DP2D_1way_callers.cuh"


template <bool _is_reduce_h>
const void* decx::dot::cuda_DP2D_1way_fp32_caller_Async(decx::dot::cuda_DP2D_configs<float>* _configs, decx::cuda_stream* S)
{
    const void* res_ptr = _configs->postproc_needed() ? _configs->get_configs_ptr<float>()->get_src()._ptr.ptr : _configs->_dev_dst.ptr;

    if (_is_reduce_h) {
        decx::dot::GPUK::cu_block_dot2D_1way_h_fp32 << < _configs->get_1st_kernel_config(),
            dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_), 0, S->get_raw_stream_ref() >> > (
                (float4*)_configs->_dev_A.ptr,
                (float4*)_configs->_dev_B.ptr,
                (float*)res_ptr,
                decx::utils::ceil<uint32_t>(_configs->get_actual_proc_dims().x, 4),
                decx::utils::ceil<uint32_t>(_configs->get_1st_kernel_config().x, 4) * 4,
                _configs->get_actual_proc_dims());
    }
    else {
        decx::dot::GPUK::cu_block_dot2D_1way_v_fp32 << < _configs->get_1st_kernel_config(),
            dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_), 0, S->get_raw_stream_ref() >> > (
                (float4*)_configs->_dev_A.ptr,
                (float4*)_configs->_dev_B.ptr,
                (float4*)res_ptr,
                decx::utils::ceil<uint32_t>(_configs->get_actual_proc_dims().x, 4),
                decx::utils::ceil<uint32_t>(_configs->get_actual_proc_dims().x, 4),
                _configs->get_actual_proc_dims());
    }

    if (_configs->postproc_needed()) {
        decx::reduce::cuda_reduce2D_1way_configs<float>* _postproc_configs = _configs->get_configs_ptr<float>();

        if (_is_reduce_h) {
            decx::reduce::reduce_sum2D_h_fp32_Async(_postproc_configs, S);
        }
        else {
            decx::reduce::reduce_sum2D_v_fp32_Async(_postproc_configs, S);
        }

        return _postproc_configs->get_dst();
    }
    else {
        return _configs->_dev_dst.ptr;
    }
}

template const void* decx::dot::cuda_DP2D_1way_fp32_caller_Async<true>(decx::dot::cuda_DP2D_configs<float>*, decx::cuda_stream*);
template const void* decx::dot::cuda_DP2D_1way_fp32_caller_Async<false>(decx::dot::cuda_DP2D_configs<float>*, decx::cuda_stream*);


#define _DOT2D_1WAY_H_FP16_PARAM(_dst_type)                                                                 \
    (float4*)_configs->_dev_A.ptr, (float4*)_configs->_dev_B.ptr, (_dst_type*)res_ptr,                      \
    decx::utils::ceil<uint32_t>(_configs->get_actual_proc_dims().x, 8),                                     \
    decx::utils::ceil<uint32_t>(_configs->get_1st_kernel_config().x, (sizeof(float4) / sizeof(_dst_type)))  \
    * (sizeof(float4) / sizeof(_dst_type)),                                                                 \
    _configs->get_actual_proc_dims()                                                                        \


#define _DOT2D_1WAY_V_FP16_PARAM(_dst_type)                                                                 \
    (float4*)_configs->_dev_A.ptr, (float4*)_configs->_dev_B.ptr, (float4*)res_ptr,                         \
    decx::utils::ceil<uint32_t>(_configs->get_actual_proc_dims().x, 8),                                     \
    decx::utils::ceil<uint32_t>(_configs->get_actual_proc_dims().x, (sizeof(float4) / sizeof(_dst_type))),  \
    _configs->get_actual_proc_dims()                                                                        \
    


template <bool _is_reduce_h>
const void* decx::dot::cuda_DP2D_1way_fp16_caller_Async(decx::dot::cuda_DP2D_configs<de::Half>* _configs, const uint32_t _fp16_accu, decx::cuda_stream* S)
{
    const void* res_ptr = NULL;
    if (_fp16_accu == decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1) {
        res_ptr = _configs->postproc_needed() ? _configs->get_configs_ptr<float>()->get_src()._ptr.ptr : _configs->_dev_dst.ptr;
    }
    else {
        res_ptr = _configs->postproc_needed() ? _configs->get_configs_ptr<de::Half>()->get_src()._ptr.ptr : _configs->_dev_dst.ptr;
    }

    if (_is_reduce_h) {
        switch (_fp16_accu)
        {
        case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1:
            decx::dot::GPUK::cu_block_dot2D_1way_h_fp16_L1 << < _configs->get_1st_kernel_config(),
                dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_), 0, S->get_raw_stream_ref() >> > (_DOT2D_1WAY_H_FP16_PARAM(float));
            break;

        case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L2:
            decx::dot::GPUK::cu_block_dot2D_1way_h_fp16_L2 << < _configs->get_1st_kernel_config(),
                dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_), 0, S->get_raw_stream_ref() >> > (_DOT2D_1WAY_H_FP16_PARAM(half));
            break;

        case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L3:
            decx::dot::GPUK::cu_block_dot2D_1way_h_fp16_L3 << < _configs->get_1st_kernel_config(),
                dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_), 0, S->get_raw_stream_ref() >> > (_DOT2D_1WAY_H_FP16_PARAM(half));
            break;
        default:
            break;
        }
    }
    else {
        switch (_fp16_accu)
        {
        case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1:
            decx::dot::GPUK::cu_block_dot2D_1way_v_fp16_L1 << < _configs->get_1st_kernel_config(),
                dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_), 0, S->get_raw_stream_ref() >> > (_DOT2D_1WAY_V_FP16_PARAM(float));
            break;

        case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L2:
            decx::dot::GPUK::cu_block_dot2D_1way_v_fp16_L2 << < _configs->get_1st_kernel_config(),
                dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_), 0, S->get_raw_stream_ref() >> > (_DOT2D_1WAY_V_FP16_PARAM(half));
            break;

        case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L3:
            decx::dot::GPUK::cu_block_dot2D_1way_v_fp16_L3 << < _configs->get_1st_kernel_config(),
                dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_), 0, S->get_raw_stream_ref() >> > (_DOT2D_1WAY_V_FP16_PARAM(half));
            break;
        default:
            break;
        }
    }

    if (_configs->postproc_needed()) {
        if (_fp16_accu == decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1) {
            decx::reduce::cuda_reduce2D_1way_configs<float>* _postproc_configs = _configs->get_configs_ptr<float>();

            if (_is_reduce_h)   decx::reduce::reduce_sum2D_h_fp32_Async(_postproc_configs, S);
            else    decx::reduce::reduce_sum2D_v_fp32_Async(_postproc_configs, S);

            return _postproc_configs->get_dst();
        }
        else {
            decx::reduce::cuda_reduce2D_1way_configs<de::Half>* _postproc_configs = _configs->get_configs_ptr<de::Half>();

            if (_is_reduce_h)   decx::reduce::reduce_sum2D_h_fp16_Async(_postproc_configs, S, _fp16_accu);
            else    decx::reduce::reduce_sum2D_v_fp16_Async(_postproc_configs, S, _fp16_accu);

            return _postproc_configs->get_dst();
        }
    }
    else {
        return _configs->_dev_dst.ptr;
    }
}

template const void* decx::dot::cuda_DP2D_1way_fp16_caller_Async<true>(decx::dot::cuda_DP2D_configs<de::Half>*, const uint32_t, decx::cuda_stream*);
template const void* decx::dot::cuda_DP2D_1way_fp16_caller_Async<false>(decx::dot::cuda_DP2D_configs<de::Half>*, const uint32_t, decx::cuda_stream*);

#ifdef _DOT2D_1WAY_H_FP16_PARAM
#undef _DOT2D_1WAY_H_FP16_PARAM
#endif

#ifdef _DOT2D_1WAY_V_FP16_PARAM
#undef _DOT2D_1WAY_V_FP16_PARAM
#endif

// Host

template <bool _is_reduce_h>
void decx::dot::matrix_dot_1way_fp32(decx::_Matrix* A, decx::_Matrix* B, decx::_Vector* dst)
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

    decx::dot::cuda_DP2D_configs<float> _configs;
    _configs.generate_config<_is_reduce_h>(make_uint2(A->Width(), A->Height()), S);
    _configs.alloc_buffers<_is_reduce_h>(S, 0);

    // Copy the data from matrices on host to the memory on device
    // Matrix A
    checkCudaErrors(cudaMemcpy2DAsync(_configs._dev_A.ptr,          _configs._dev_mat_dims.x * sizeof(float),
                                      A->Mat.ptr,                   A->Pitch() * sizeof(float), 
                                      A->Width() * sizeof(float),   A->Height(), 
                                      cudaMemcpyHostToDevice,       S->get_raw_stream_ref()));
    // Matrix B
    checkCudaErrors(cudaMemcpy2DAsync(_configs._dev_B.ptr,          _configs._dev_mat_dims.x * sizeof(float),
                                      B->Mat.ptr,                   B->Pitch() * sizeof(float), 
                                      B->Width() * sizeof(float),   B->Height(), 
                                      cudaMemcpyHostToDevice,       S->get_raw_stream_ref()));

    const void* res_ptr = decx::dot::cuda_DP2D_1way_fp32_caller_Async<_is_reduce_h>(&_configs, S);

    if (res_ptr == NULL) {
        Print_Error_Message(4, INTERNAL_ERROR);
        return;
    }
    const uint64_t _cpy_size = (_is_reduce_h ? A->Height() : A->Width()) * sizeof(float);
    checkCudaErrors(cudaMemcpyAsync(dst->Vec.ptr, res_ptr, _cpy_size, cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));
    
    E->event_record(S);
    E->synchronize();
}

template void decx::dot::matrix_dot_1way_fp32<true>(decx::_Matrix* A, decx::_Matrix* B, decx::_Vector* dst);
template void decx::dot::matrix_dot_1way_fp32<false>(decx::_Matrix* A, decx::_Matrix* B, decx::_Vector* dst);




template <bool _is_reduce_h>
void decx::dot::matrix_dot_1way_fp16(decx::_Matrix* A, decx::_Matrix* B, decx::_Vector* dst, const uint32_t _fp16_accu)
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

    decx::dot::cuda_DP2D_configs<de::Half> _configs;
    _configs.generate_config<_is_reduce_h>(make_uint2(A->Width(), A->Height()), S, _fp16_accu);
    _configs.alloc_buffers<_is_reduce_h>(S, _fp16_accu);

    // Copy the data from matrices on host to the memory on device
    // Matrix A
    checkCudaErrors(cudaMemcpy2DAsync(_configs._dev_A.ptr,              _configs._dev_mat_dims.x * sizeof(de::Half),
                                      A->Mat.ptr,                       A->Pitch() * sizeof(de::Half),
                                      A->Width() * sizeof(de::Half),    A->Height(),
                                      cudaMemcpyHostToDevice,           S->get_raw_stream_ref()));
    // Matrix B
    checkCudaErrors(cudaMemcpy2DAsync(_configs._dev_B.ptr,              _configs._dev_mat_dims.x * sizeof(de::Half),
                                      B->Mat.ptr,                       B->Pitch() * sizeof(de::Half),
                                      B->Width() * sizeof(de::Half),    B->Height(),
                                      cudaMemcpyHostToDevice,           S->get_raw_stream_ref()));

    const void* res_ptr = decx::dot::cuda_DP2D_1way_fp16_caller_Async<_is_reduce_h>(&_configs, _fp16_accu, S);

    if (res_ptr == NULL) {
        Print_Error_Message(4, INTERNAL_ERROR);
        return;
    }
    const uint8_t _dst_ele_size = _fp16_accu == decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1 ? sizeof(float) : sizeof(de::Half);
    const uint64_t _cpy_size = (_is_reduce_h ? A->Height() : A->Width()) * _dst_ele_size;
    checkCudaErrors(cudaMemcpyAsync(dst->Vec.ptr, res_ptr, _cpy_size, cudaMemcpyDeviceToHost, S->get_raw_stream_ref()));

    E->event_record(S);
    E->synchronize();
}

template void decx::dot::matrix_dot_1way_fp16<true>(decx::_Matrix*, decx::_Matrix*, decx::_Vector*, const uint32_t);
template void decx::dot::matrix_dot_1way_fp16<false>(decx::_Matrix*, decx::_Matrix*, decx::_Vector*, const uint32_t);


#ifdef _DOT2D_1WAY_H_FP16_PARAM
#undef _DOT2D_1WAY_H_FP16_PARAM
#endif


#ifdef _DOT2D_1WAY_V_FP16_PARAM
#undef _DOT2D_1WAY_V_FP16_PARAM
#endif