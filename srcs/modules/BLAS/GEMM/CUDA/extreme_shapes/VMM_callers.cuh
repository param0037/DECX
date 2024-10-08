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


#ifndef _VMM_CALLERS_CUH_
#define _VMM_CALLERS_CUH_


#include "VMM_kernels.cuh"
#include "../../../Dot product/CUDA/2D/DP2D_1way.cuh"


namespace decx
{
namespace blas {
    template <bool _is_reduce_h>
    void generate_VMM_config_fp32(decx::blas::cuda_DP2D_configs<float>* _configs, const uint2 proc_dims, decx::cuda_stream* S);


    template <bool _is_reduce_h>
    void generate_VMM_config_fp16(decx::blas::cuda_DP2D_configs<de::Half>* _configs, const uint2 proc_dims, decx::cuda_stream* S,
        const uint32_t _fp16_accu);
}
}



namespace decx
{
    template <bool _is_reduce_h>
    static void* _VMM_fp32_caller_async(decx::blas::cuda_DP2D_configs<float>* _configs, decx::cuda_stream* S);


    template <bool _is_reduce_h>
    static void* _VMM_fp16_caller_async(decx::blas::cuda_DP2D_configs<de::Half>* _configs, decx::cuda_stream* S, 
        const uint32_t _fp16_accu);
}


template <bool _is_reduce_h>
static void* decx::_VMM_fp32_caller_async(decx::blas::cuda_DP2D_configs<float>* _configs, decx::cuda_stream* S)
{
    const void* res_ptr = _configs->postproc_needed() ? _configs->get_configs_ptr<float>()->get_src()._ptr.ptr : _configs->_dev_dst.ptr;
    
    if (_is_reduce_h) {
        decx::GPUK::cu_mat_m_vec_fp32 << < _configs->get_1st_kernel_config(),
            dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_), 0, S->get_raw_stream_ref() >> > (
                (float4*)_configs->_dev_A.ptr,
                (float4*)_configs->_dev_B.ptr,
                (float*)res_ptr,
                decx::utils::ceil<uint32_t>(_configs->get_actual_proc_dims().x, 4),
                decx::utils::ceil<uint32_t>(_configs->get_1st_kernel_config().x, 4) * 4,
                _configs->get_actual_proc_dims());
    }
    else {
        decx::GPUK::cu_vec_m_mat_fp32 << < _configs->get_1st_kernel_config(),
            dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_), 0, S->get_raw_stream_ref() >> > (
                (float*)_configs->_dev_B.ptr,
                (float4*)_configs->_dev_A.ptr,
                (float4*)res_ptr,
                decx::utils::ceil<uint32_t>(_configs->get_actual_proc_dims().x, 4),
                decx::utils::ceil<uint32_t>(_configs->get_actual_proc_dims().x, 4),
                _configs->get_actual_proc_dims());
    }

    if (_configs->postproc_needed()) {
        decx::reduce::cuda_reduce2D_1way_configs<float>* _postproc_configs = _configs->get_configs_ptr<float>();
        if(_is_reduce_h) {
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


#define _VMM_1WAY_H_FP16_PARAM(_dst_type)                                                                   \
    (float4*)_configs->_dev_A.ptr, (float4*)_configs->_dev_B.ptr, (_dst_type*)res_ptr,                      \
    decx::utils::ceil<uint32_t>(_configs->get_actual_proc_dims().x, 8),                                     \
    decx::utils::ceil<uint32_t>(_configs->get_1st_kernel_config().x, (sizeof(float4) / sizeof(_dst_type)))  \
    * (sizeof(float4) / sizeof(_dst_type)),                                                                 \
    _configs->get_actual_proc_dims()                                                                        \


#define _VMM_1WAY_V_FP16_PARAM(_dst_type)                                                                   \
    (__half*)_configs->_dev_B.ptr, (float4*)_configs->_dev_A.ptr, (float4*)res_ptr,                         \
    decx::utils::ceil<uint32_t>(_configs->get_actual_proc_dims().x, 8),                                     \
    decx::utils::ceil<uint32_t>(_configs->get_actual_proc_dims().x, (sizeof(float4) / sizeof(_dst_type))),  \
    _configs->get_actual_proc_dims()                                                                        \



template <bool _is_reduce_h>
static void* decx::_VMM_fp16_caller_async(decx::blas::cuda_DP2D_configs<de::Half>* _configs, decx::cuda_stream* S,
    const uint32_t _fp16_accu)
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
            decx::GPUK::cu_mat_m_vec_fp16_L1 << < _configs->get_1st_kernel_config(),
                dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_), 0, S->get_raw_stream_ref() >> > (_VMM_1WAY_H_FP16_PARAM(float));
            break;

        case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L2:
            decx::GPUK::cu_mat_m_vec_fp16_L2 << < _configs->get_1st_kernel_config(),
                dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_), 0, S->get_raw_stream_ref() >> > (_VMM_1WAY_H_FP16_PARAM(half));
            break;

        case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L3:
            decx::GPUK::cu_mat_m_vec_fp16_L3 << < _configs->get_1st_kernel_config(),
                dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_), 0, S->get_raw_stream_ref() >> > (_VMM_1WAY_H_FP16_PARAM(half));
            break;
        default:
            break;
        }
    }
    else {
        switch (_fp16_accu)
        {
        case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1:
            decx::GPUK::cu_vec_m_mat_fp16_L1 << < _configs->get_1st_kernel_config(),
                dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_), 0, S->get_raw_stream_ref() >> > (_VMM_1WAY_V_FP16_PARAM(float));
            break;

        case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L2:
            decx::GPUK::cu_vec_m_mat_fp16_L2 << < _configs->get_1st_kernel_config(),
                dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_), 0, S->get_raw_stream_ref() >> > (_VMM_1WAY_V_FP16_PARAM(half));
            break;

        case decx::Fp16_Accuracy_Levels::Fp16_Accurate_L3:
            decx::GPUK::cu_vec_m_mat_fp16_L3 << < _configs->get_1st_kernel_config(),
                dim3(_REDUCE2D_BLOCK_DIM_X_, _REDUCE2D_BLOCK_DIM_Y_), 0, S->get_raw_stream_ref() >> > (_VMM_1WAY_V_FP16_PARAM(half));
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



#endif