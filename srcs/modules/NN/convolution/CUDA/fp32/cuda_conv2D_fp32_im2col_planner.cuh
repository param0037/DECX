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


        template <typename _data_type>
        struct cuda_conv2D_im2col_kernel_arrange;
    }
}


template <typename _data_type>
struct decx::nn::cuda_conv2D_im2col_kernel_arrange
{
    decx::PtrInfo<void> _shrinked_kernel, _transposed_kernel;

    uint2 _eq_kernel_dims_2D;
    uint2 _transp_ker_dims;
    uint32_t _kernel_tensor_num;

    const decx::_tensor_layout* _kernel_layout;

    cudaMemcpy3DParms _kernel_cpy_params;


    void _CRSR_ init(const decx::_GPU_TensorArray* kernel, decx::cuda_stream* S, de::DH* handle);


    void arrange_kernel(const decx::_GPU_TensorArray* kernel, decx::cuda_stream* S);


    void release();
};



/**
* The execution order should be : plan -> dst_dims_query -> run 
*/
class decx::nn::cuda_conv2D_fp32_im2col_planner
{
public:
    uint2 _strides;
    const decx::_tensor_layout* _src_layout;

    decx::nn::cuda_conv2D_im2col_kernel_arrange<float> _kernel_manager;

    // [D, W, H]
    uint3 _dst_dims;
    
    de::extend_label _ext_method;

    
    ulong2 _im2col_buf_dims;
    decx::Ptr2D_Info<float4> _ext_src_buf;     // Allocated if extend method == border_zero
    decx::PtrInfo<void> _im2col_buf;
    

    dim3 _grid_i2c, _block_i2c;
    dim3 _grid_gemm, _block_gemm;


    void _cpy_src_ext(decx::_GPU_Tensor* src, decx::cuda_stream* S) const;


public:
    cuda_conv2D_fp32_im2col_planner() {}


    void _CRSR_ plan(const decx::_tensor_layout* src_layout, const decx::_GPU_TensorArray* kernel,
        const de::extend_label ext_method, const uint2 strides, 
        decx::cuda_stream* S, de::DH *handle);


    void _CRSR_ run(decx::_GPU_Tensor* src, decx::_GPU_TensorArray* kernel, decx::_GPU_Tensor* dst, 
        decx::cuda_stream* S, de::DH* handle);

    // [D, W, H]
    const uint3& dst_dims_query() const;
};


#endif