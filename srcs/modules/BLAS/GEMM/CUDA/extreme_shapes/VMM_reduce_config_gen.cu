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

#include "../../../Dot product/CUDA/2D/DP2D_1way.cuh"
#include "VMM_callers.cuh"


template <bool _is_reduce_h>
void decx::blas::generate_VMM_config_fp32(decx::blas::cuda_DP2D_configs<float>* _configs, const uint2 proc_dims, decx::cuda_stream* S)
{
    _configs->generate_config<_is_reduce_h>(proc_dims, S);

    if (decx::alloc::_device_malloc(&(_configs->_dev_A), _configs->_dev_mat_dims.x * _configs->_dev_mat_dims.y * sizeof(float), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&(_configs->_dev_B), decx::utils::align(_is_reduce_h ? proc_dims.x : proc_dims.y, _CU_REDUCE1D_MEM_ALIGN_4B_)
        * sizeof(float), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }
    if (!_configs->postproc_needed()) {
        uint32_t _alloc_dst_size = 0;
        _alloc_dst_size = decx::utils::align<uint32_t>(_is_reduce_h ? proc_dims.y : proc_dims.x, _CU_REDUCE1D_MEM_ALIGN_4B_) * sizeof(float);

        if (decx::alloc::_device_malloc(&(_configs->_dev_dst), _alloc_dst_size, true, S)) {
            Print_Error_Message(4, DEV_ALLOC_FAIL);
            return;
        }
    }
}

template void decx::blas::generate_VMM_config_fp32<true>(decx::blas::cuda_DP2D_configs<float>*, const uint2, decx::cuda_stream*);
template void decx::blas::generate_VMM_config_fp32<false>(decx::blas::cuda_DP2D_configs<float>*, const uint2, decx::cuda_stream*);


template <bool _is_reduce_h>
void decx::blas::generate_VMM_config_fp16(decx::blas::cuda_DP2D_configs<de::Half>* _configs, const uint2 proc_dims, decx::cuda_stream* S,
    const uint32_t _fp16_accu)
{
    _configs->generate_config<_is_reduce_h>(proc_dims, S, _fp16_accu);

    if (decx::alloc::_device_malloc(&(_configs->_dev_A), _configs->_dev_mat_dims.x * _configs->_dev_mat_dims.y * sizeof(de::Half), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }
    if (decx::alloc::_device_malloc(&(_configs->_dev_B), decx::utils::align(_is_reduce_h ? proc_dims.x : proc_dims.y, _CU_REDUCE1D_MEM_ALIGN_4B_)
        * sizeof(de::Half), true, S)) {
        Print_Error_Message(4, DEV_ALLOC_FAIL);
        return;
    }
    if (!_configs->postproc_needed()) {
        uint32_t _alloc_dst_size = 0;
        if (_fp16_accu == decx::Fp16_Accuracy_Levels::Fp16_Accurate_L1) {
            _alloc_dst_size = decx::utils::align<uint32_t>(proc_dims.x, _CU_REDUCE1D_MEM_ALIGN_4B_) * sizeof(float);
        }
        else {
            _alloc_dst_size = decx::utils::align<uint32_t>(proc_dims.x, _CU_REDUCE1D_MEM_ALIGN_4B_) * sizeof(de::Half);
        }
        if (decx::alloc::_device_malloc(&_configs->_dev_dst, _alloc_dst_size, true, S)) {
            Print_Error_Message(4, DEV_ALLOC_FAIL);
            return;
        }
    }
}

template void decx::blas::generate_VMM_config_fp16<true>(decx::blas::cuda_DP2D_configs<de::Half>*, const uint2, decx::cuda_stream*, const uint32_t);
template void decx::blas::generate_VMM_config_fp16<false>(decx::blas::cuda_DP2D_configs<de::Half>*, const uint2, decx::cuda_stream*, const uint32_t);
