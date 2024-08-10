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

#ifndef _CUDA_GEMM_LS_PLANNER_CUH_
#define _CUDA_GEMM_LS_PLANNER_CUH_

#include "../../../../../../common/basic.h"
#include "../../../../../core/allocators.h"
#include "../../../../../../common/Classes/GPU_Matrix.h"
#include "../../../../../../common/Classes/Number.h"
#include "../../../../../core/configs/config.h"
#include "../../../../../../common/error.h"

#include "../../../../../core/cudaStream_management/cudaStream_queue.h"
#include "../../../../../core/cudaStream_management/cudaEvent_queue.h"
#include "../../../../../core/resources_manager/decx_resource.h"

namespace decx
{
namespace blas{
    template <typename _data_type>
    class cuda_GEMM_LS_planner;


    typedef struct cu_GEMM_kernel_prop_t
    {
        void* _kernel_ptr;              // Pointer to the kernel
        uint64_t _regs_per_thread;      // Used # of 32-bit registers per thread
        uint64_t _shared_per_block;     // Used size of shared memory per block (Byte)
        uint32_t _thread_per_block;     // Nominated # of threads per block
        float _mio_compute_ratio;       // ratio of memory io to compute
        uint3 _LWH;
        uint32_t _transpose_A;
    }CGKP;


    namespace GPUK {
    template <typename _data_type>
    using _cu_GEMM_kernel_ptr = void(const void*, const void*, void*, const uint2, 
        const uint32_t, const uint32_t, const uint32_t, const uint32_t, decx::cuda_stream*,
        const void*, const _data_type, const _data_type);
    }

    extern decx::ResourceHandle g_cu_GEMM_fp32_planner;
    extern decx::ResourceHandle g_cu_GEMM_fp16_planner;
    extern decx::ResourceHandle g_cu_GEMM_fp64_planner;
    extern decx::ResourceHandle g_cu_GEMM_cplxd_planner;
}
}

// Randomly assign a valid type (non-void)
#define _CUDA_GEMM_LS_PLANNER_GENERAL_TYPE_ float


template <typename _data_type>
class decx::blas::cuda_GEMM_LS_planner
{
private:
    uint2 _proc_dims;

    decx::Ptr2D_Info<void> _AT;

    decx::_matrix_layout _A_layout;
    decx::_matrix_layout _B_layout;
    decx::_matrix_layout _dst_layout;

    static decx::blas::CGKP _kernel_props[9];

    const cudaDeviceProp* _device_prop;

    uint32_t _kernel_id;

    /**
     * @return the estimated occupancy.
     * @param _kernel_info The pointer of the kernel properties structure.
    */
    float estimate_occupancy(const decx::blas::CGKP* _kernel_info) const;


    bool validate_kernel(const decx::blas::CGKP* _kernel_info) const;


    float padding_efficiency(const uint2 proc_dims_v1, const uint32_t L, const decx::blas::CGKP* _kernel_info) const;

    /**
     * @brief This private memberfunction assess the candidate GEMM kernels. The assessment is done by :
     *        mark = occupancy * padding_efficiency * K. K = <macro>_TRANSPOSE_A_BONUS_ if this is a kernel requires the
     *        trabsposed form of matrix A; otherwise 1.
    */
    float kernel_assessment(const uint2 proc_dims_v1, const uint32_t L, const decx::blas::CGKP* _kernel_info) const;

public:
    cuda_GEMM_LS_planner() {
        memset(this, 0, sizeof(decx::blas::cuda_GEMM_LS_planner<_data_type>));
    }


    static void validate(const decx::_GPU_Matrix* A, const decx::_GPU_Matrix* B,
        const decx::_GPU_Matrix* C, de::DH* handle, const de::Number* alpha = NULL, const de::Number* beta = NULL);


    bool changed(const decx::_matrix_layout* A_layout, const decx::_matrix_layout* B_layout) const;

    /**
     * @param A_layout The layout descriptor of matrix A.
     * @param B_layout The layout descriptor of matrix B.
     * @param dst_layout The layout descriptor of the destinated matrix.
     * @param handle de::DH* to carry any error produced.
     * @param S decx::cuda_stream* object, carrying the cudaStream.
     * @param aux_memory_budget Budget for additional memory used, mainly for transposing of matrix A to accelerate
     *                          the kernel MIO performance. The default value is 0, for non-conservating memory use.
    */
    void _CRSR_ plan(const decx::_matrix_layout* A_layout, const decx::_matrix_layout* B_layout,
        const decx::_matrix_layout* dst_layout, de::DH* handle, decx::cuda_stream* S, uint64_t aux_memory_budget = 0);

    
    void run(decx::_GPU_Matrix* A, decx::_GPU_Matrix* B, decx::_GPU_Matrix* dst, decx::cuda_stream* S);


    void run(decx::_GPU_Matrix* A, decx::_GPU_Matrix* B, decx::_GPU_Matrix* C, decx::_GPU_Matrix* dst, 
        const _data_type alpha, const _data_type beta, decx::cuda_stream* S);


    static void release(decx::blas::cuda_GEMM_LS_planner<_data_type>* _fake_this);


    template <typename _Dtype>
    decx::blas::cuda_GEMM_LS_planner<_Dtype>* _reinterpret_cast() {
        return (decx::blas::cuda_GEMM_LS_planner<_Dtype>*)this;
    }


};


#endif