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


#ifndef _CUDA_CONV2D_FP32_IM2COL_PLANNER_CUH_
#define _CUDA_CONV2D_FP32_IM2COL_PLANNER_CUH_


#include "im2col_fp32.cuh"
#include "im2col_GEMM_fp32.cuh"
#include "../../../../classes/GPU_Tensor.h"
#include "../../../../classes/Tensor.h"
#include "../../../../classes/GPU_TensorArray.h"
#include "../../../../BLAS/basic_process/transpose/CUDA/transpose_kernels.cuh"
#include "../../../../BLAS/basic_process/extension/extend_flags.h"
#include "../../../../core/utils/Fixed_Length_Array.h"


namespace decx
{
    namespace nn {
        class cuda_conv2D_fp32_im2col_planner;


        template <typename _data_type>
        struct cuda_conv2D_im2col_kernel_arrange;


        struct cuda_conv2D_im2col_kernel_params;


        void InitCUDAConv2DResource();
    }
}


struct decx::nn::cuda_conv2D_im2col_kernel_params
{
    void* _src_loc;
    void* _dst_loc;
    
    uint32_t _proc_H;
    uint64_t _im2col_bufW;

    dim3 _grid_i2c, _block_i2c;
    dim3 _grid_gemm, _block_gemm;
};


#define _MAX_IM2COL_TILE_SIZE_ (uint64_t)2048*2048*16


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
    const decx::_tensor_layout* _dst_layout;

    decx::nn::cuda_conv2D_im2col_kernel_arrange<float> _kernel_manager;

    uint32_t _I2C_wpitch;       // 128 bytes aligned
    uint32_t _wpitchsrc_proc_v1;

    // [D, W, H]
    uint3 _dst_dims;
    
    de::extend_label _ext_method;

    decx::utils::Fixed_Length_Array<decx::nn::cuda_conv2D_im2col_kernel_params> _params_array;
    
    decx::Ptr2D_Info<float4> _ext_src_buf;     // Allocated if extend method == border_zero
    decx::PtrInfo<void> _im2col_buf;

    ulong2 _im2col_buf_alloc;
    

    void _cpy_src_ext(decx::_GPU_Tensor* src, decx::cuda_stream* S) const;


    void _kernel_launch_config(const uint32_t _proc_idx, const uint32_t _proc_h);


    void _flush_im2col_buf(decx::cuda_stream* S, const bool _is_top);

public:
    cuda_conv2D_fp32_im2col_planner();


    void _CRSR_ plan(const decx::_tensor_layout* src_layout, const decx::_GPU_TensorArray* kernel,
        const de::extend_label ext_method, const uint2 strides, 
        decx::cuda_stream* S, de::DH *handle);


    bool changed(const decx::_tensor_layout* src_layout, const decx::_GPU_TensorArray* kernel,
        const de::extend_label ext_method, const uint2 strides) const;


    template <bool _boundless_T, bool _boundless_B>
    void _CRSR_ run_single_frag_BC(const uint32_t _proc_idx, decx::cuda_stream* S);


    void _CRSR_ run_single_frag_NB(const uint32_t _proc_idx, decx::cuda_stream* S);


    void _CRSR_ run(decx::_GPU_Tensor* src, decx::_GPU_TensorArray* kernel, decx::_GPU_Tensor* dst,
        decx::cuda_stream* S, de::DH* handle);


    void update_dst_layout(const decx::_tensor_layout* dst_layout);


    // [D, W, H]
    const uint3& dst_dims_query() const;


    void release();
};


namespace decx
{
    namespace nn {
        extern decx::nn::cuda_conv2D_fp32_im2col_planner* _conv2_fp32_planner;
    }
}


#endif