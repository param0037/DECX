/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CUDA_CONV2D_FP32_IM2COL_PLANNER_CUH_
#define _CUDA_CONV2D_FP32_IM2COL_PLANNER_CUH_


#include "im2col_fp32.cuh"
#include "im2col_GEMM_fp32.cuh"
#include "../../../../classes/GPU_Tensor.h"
#include "../../../../classes/Tensor.h"
#include "../../../../classes/GPU_TensorArray.h"
#include "../../../../BLAS/basic_process/transpose/CUDA/transpose_kernels.cuh"
#include "../../../../BLAS/basic_process/extension/extend_flags.h"


namespace decx
{
    namespace nn {
        class cuda_conv2D_fp32_im2col_planner;
    }
}


class decx::nn::cuda_conv2D_fp32_im2col_planner
{
private:
    uint2 _strides;
    const decx::_tensor_layout* _src_layout;
    const decx::_tensor_layout* _kernel_layout;
    uint32_t _kernel_tensor_num;
    const decx::_tensor_layout* _dst_layout;

    decx::bp::extend_label _ext_method;

    decx::PtrInfo<float4> _ext_src_buf;     // Allocated if extend method == border_zero
    decx::PtrInfo<void> _im2col_buf;

    decx::PtrInfo<void> _shrinked_kernel, _transposed_kernel;

public:
    cuda_conv2D_fp32_im2col_planner() {}


    void _CRSR_ plan(const decx::_tensor_layout* src_layout, const decx::_GPU_TensorArray* kernel_layout,
        const decx::_tensor_layout* dst_layout, const decx::bp::extend_label ext_method, const uint2 strides, 
        decx::cuda_stream* S, de::DH *handle);
};


#endif