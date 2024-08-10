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

#include "cuda_GEMM_LS_planner.cuh"


template <typename _data_type>
float decx::blas::cuda_GEMM_LS_planner<_data_type>::
estimate_occupancy(const decx::blas::CGKP* _kernel_info) const
{
    const uint64_t& regs_per_SM = this->_device_prop->regsPerMultiprocessor;
    const uint64_t& shmem_per_SM = this->_device_prop->sharedMemPerMultiprocessor;

    const uint64_t regs_per_block = _kernel_info->_regs_per_thread * _kernel_info->_thread_per_block;

    // Calculate the occupancy of registers on SM
    uint32_t _blocks_per_SM = min(regs_per_SM / regs_per_block, 
                                  shmem_per_SM / _kernel_info->_shared_per_block);

    // If _blocks_per_SM == 0, it means that either register or shared memory size exceeds the hardware limit.
    // CUDA RT library will report error.
    // Hence, assume _block_per_SM >= 1 in this case.
    if (_blocks_per_SM == 0){
        return -1.f;
    }
    float _occupancy = min((float)(_blocks_per_SM * regs_per_block) / (float)regs_per_SM,
                           (float)(_blocks_per_SM * _kernel_info->_shared_per_block) / (float)shmem_per_SM);
                           
    return _occupancy;
}

template float decx::blas::cuda_GEMM_LS_planner<float>::estimate_occupancy(const decx::blas::CGKP*) const;
template float decx::blas::cuda_GEMM_LS_planner<double>::estimate_occupancy(const decx::blas::CGKP*) const;
template float decx::blas::cuda_GEMM_LS_planner<de::Half>::estimate_occupancy(const decx::blas::CGKP*) const;
template float decx::blas::cuda_GEMM_LS_planner<de::CPf>::estimate_occupancy(const decx::blas::CGKP*) const;
template float decx::blas::cuda_GEMM_LS_planner<de::CPd>::estimate_occupancy(const decx::blas::CGKP*) const;


template <typename _data_type>
bool decx::blas::cuda_GEMM_LS_planner<_data_type>::validate_kernel(const decx::blas::CGKP* _kernel_info) const
{
    const uint64_t regs_per_block = _kernel_info->_regs_per_thread * _kernel_info->_thread_per_block;
    return (this->_device_prop->regsPerMultiprocessor > regs_per_block) &&
           (this->_device_prop->sharedMemPerMultiprocessor > _kernel_info->_shared_per_block);
}


template <typename _data_type>
float decx::blas::cuda_GEMM_LS_planner<_data_type>::
padding_efficiency(const uint2 proc_dims_v1, const uint32_t L, const decx::blas::CGKP* _kernel_info) const
{
    uint2 cover_dims_dst = make_uint2(decx::utils::align<uint32_t>(proc_dims_v1.x, _kernel_info->_LWH.y),
                                 decx::utils::align<uint32_t>(proc_dims_v1.y, _kernel_info->_LWH.z));
                                 
    float efficiency = ((float)proc_dims_v1.x / (float)cover_dims_dst.x) * 
                       ((float)proc_dims_v1.y / (float)cover_dims_dst.y);

    efficiency *= ((float)L / (float)decx::utils::align<uint32_t>(L, _kernel_info->_LWH.x));
    
    return efficiency;
}


#define _TRANSPOSE_A_BONUS_ 1.2


template <typename _data_type>
float decx::blas::cuda_GEMM_LS_planner<_data_type>::
kernel_assessment(const uint2 proc_dims_v1, const uint32_t L, const decx::blas::CGKP* _kernel_info) const
{
    if (!this->validate_kernel(_kernel_info)){
        return -1.f;
    }
    const float _occupancy = this->estimate_occupancy(_kernel_info);
    const float _padding_eff = this->padding_efficiency(proc_dims_v1, L, _kernel_info);

    return (_occupancy * _padding_eff) * (_kernel_info->_transpose_A ? _TRANSPOSE_A_BONUS_ : 1.f);
}


template<> void 
decx::blas::cuda_GEMM_LS_planner<float>::plan(const decx::_matrix_layout* A_layout, 
                                           const decx::_matrix_layout* B_layout,
                                           const decx::_matrix_layout* dst_layout, 
                                           de::DH* handle,
                                           decx::cuda_stream* S,
                                           const uint64_t aux_memory_budget)
{
    this->_device_prop = &decx::cuda::_get_cuda_prop();

    this->_A_layout = *A_layout;
    this->_B_layout = *B_layout;
    this->_dst_layout = *dst_layout;

    const uint32_t _L = this->_A_layout.width;
    const uint2 proc_dims_v1 = make_uint2(this->_B_layout.width, this->_A_layout.height);

    const uint32_t pitch_AT = decx::utils::align<uint32_t>(this->_A_layout.height, 128);
    this->_AT._dims = make_uint2(pitch_AT, this->_B_layout.width);
    const uint64_t AT_size = this->_AT._dims.x * this->_AT._dims.y * sizeof(float);

    uint32_t considered_kernels = 0;

    if (AT_size > aux_memory_budget || aux_memory_budget == 0) {
        considered_kernels = 9;
    }
    else{
        considered_kernels = 4;
    }

    // Find the kernel assessed to the highest score
    this->_kernel_id = 0;
    float _assessment = this->kernel_assessment(proc_dims_v1, _L, decx::blas::cuda_GEMM_LS_planner<float>::_kernel_props);
    if (_assessment == -1.f){
        _assessment = 0;
    }
    
    for (uint32_t i = 1; i < considered_kernels; ++i)
    {
        float new_assessment = this->kernel_assessment(proc_dims_v1, _L, decx::blas::cuda_GEMM_LS_planner<float>::_kernel_props + i);
        if (new_assessment != -1.f){
            if (new_assessment > _assessment){
                _assessment = new_assessment;
                this->_kernel_id = i;
            }
        }
    }

    if (this->_kernel_id > 3){      // requires a transposed form of matrix A
        // Assess the remaining device memory
        if (decx::alloc::_device_malloc(&this->_AT._ptr, AT_size, true, S)){
            decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            DEV_ALLOC_FAIL);
            return;
        }
    }
}



template<> void 
decx::blas::cuda_GEMM_LS_planner<de::Half>::plan(const decx::_matrix_layout* A_layout, 
                                              const decx::_matrix_layout* B_layout,
                                              const decx::_matrix_layout* dst_layout, 
                                              de::DH* handle,
                                              decx::cuda_stream* S,
                                              const uint64_t aux_memory_budget)
{
    this->_device_prop = &decx::cuda::_get_cuda_prop();

    this->_A_layout = *A_layout;
    this->_B_layout = *B_layout;
    this->_dst_layout = *dst_layout;

    const uint32_t _L = this->_A_layout.width;
    const uint2 proc_dims_v1 = make_uint2(this->_B_layout.width, this->_A_layout.height);

    const uint32_t pitch_AT = decx::utils::align<uint32_t>(this->_A_layout.height, 256);
    this->_AT._dims = make_uint2(pitch_AT, this->_B_layout.width);
    const uint64_t AT_size = this->_AT._dims.x * this->_AT._dims.y * sizeof(de::Half);

    uint32_t considered_kernels = 0;

    if (AT_size > aux_memory_budget || aux_memory_budget == 0) {
        considered_kernels = 3;
    }
    else{
        considered_kernels = 2;
    }

    // Find the kernel assessed to the highest score
    this->_kernel_id = 0;
    float _assessment = this->kernel_assessment(proc_dims_v1, _L, decx::blas::cuda_GEMM_LS_planner<de::Half>::_kernel_props);
    if (_assessment == -1.f){
        _assessment = 0;
    }
    
    for (uint32_t i = 1; i < considered_kernels; ++i)
    {
        float new_assessment = this->kernel_assessment(proc_dims_v1, _L, decx::blas::cuda_GEMM_LS_planner<de::Half>::_kernel_props + i);
        if (new_assessment != -1.f){
            if (new_assessment > _assessment){
                _assessment = new_assessment;
                this->_kernel_id = i;
            }
        }
    }
    
    if (this->_kernel_id > 1){      // requires a transposed form of matrix A
        // Assess the remaining device memory
        if (decx::alloc::_device_malloc(&this->_AT._ptr, AT_size, true, S)){
            decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_ALLOCATION,
            DEV_ALLOC_FAIL);
            return;
        }
    }
}


template <> void 
decx::blas::cuda_GEMM_LS_planner<_CUDA_GEMM_LS_PLANNER_GENERAL_TYPE_>::
validate(const decx::_GPU_Matrix* A,    const decx::_GPU_Matrix* B,
         const decx::_GPU_Matrix* C,    de::DH* handle,
         const de::Number* alpha,       const de::Number* beta)
{
    if (A->Width() != B->Height()){
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_DimsNotMatching,
            "The width of matrix A and height of matrix B must be identical");
    }

    if (C == NULL){
        if(A->Type() != B->Type()){
            decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_TYPE_MOT_MATCH,
                "Types of the input matrices must be matched");
        }
    }
    else{
        if(A->Type() ^ B->Type() ^ C->Type()){
            decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_TYPE_MOT_MATCH,
                "Types of the input matrices must be matched");
        }
    }

    if (alpha != NULL && beta != NULL){
        if (alpha->Type() != A->Type() || beta->Type() != A->Type()){
            decx::err::handle_error_info_modify(de::GetLastError(), decx::DECX_error_types::DECX_FAIL_TYPE_MOT_MATCH,
                "The input scalar alpha and beta must match the type of input matrices");
            return;
        }
    }
}


template <typename _data_type>
void decx::blas::cuda_GEMM_LS_planner<_data_type>::release(decx::blas::cuda_GEMM_LS_planner<_data_type>* _fake_this)
{
    if (_fake_this->_AT._ptr.ptr != NULL){
        decx::alloc::_device_dealloc(&_fake_this->_AT._ptr);
    }
}

template void decx::blas::cuda_GEMM_LS_planner<float>::release(decx::blas::cuda_GEMM_LS_planner<float>*);
template void decx::blas::cuda_GEMM_LS_planner<de::Half>::release(decx::blas::cuda_GEMM_LS_planner<de::Half>*);
template void decx::blas::cuda_GEMM_LS_planner<de::CPf>::release(decx::blas::cuda_GEMM_LS_planner<de::CPf>*);
template void decx::blas::cuda_GEMM_LS_planner<double>::release(decx::blas::cuda_GEMM_LS_planner<double>*);
template void decx::blas::cuda_GEMM_LS_planner<de::CPd>::release(decx::blas::cuda_GEMM_LS_planner<de::CPd>*);


template <typename _data_type>
bool decx::blas::cuda_GEMM_LS_planner<_data_type>::changed(const decx::_matrix_layout* A_layout, 
                                                        const decx::_matrix_layout* B_layout) const
{
    bool A_changed = (this->_A_layout.width != A_layout->width) ||
           (this->_A_layout.height != A_layout->height);

    bool B_changed = (this->_B_layout.width != B_layout->width) ||
           (this->_B_layout.height != B_layout->height);

    return A_changed || B_changed;
}

template bool decx::blas::cuda_GEMM_LS_planner<float>::changed(const decx::_matrix_layout*, const decx::_matrix_layout*) const;
template bool decx::blas::cuda_GEMM_LS_planner<de::Half>::changed(const decx::_matrix_layout*, const decx::_matrix_layout*) const;
template bool decx::blas::cuda_GEMM_LS_planner<double>::changed(const decx::_matrix_layout*, const decx::_matrix_layout*) const;
template bool decx::blas::cuda_GEMM_LS_planner<de::CPf>::changed(const decx::_matrix_layout*, const decx::_matrix_layout*) const;
template bool decx::blas::cuda_GEMM_LS_planner<de::CPd>::changed(const decx::_matrix_layout*, const decx::_matrix_layout*) const;
