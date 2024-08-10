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


#include "../common/cuda_filter2D_planner.cuh"
#include "../common/filter2D_kernel.cuh"


decx::ResourceHandle decx::dsp::_cuda_filter2D_fp32;


template <> template <uint32_t _ext_w> void decx::dsp::cuda_Filter2D_planner<float>::
_cu_Filter2D_fp32_caller(const decx::dsp::cuda_Filter2D_planner<float>* _fake_this, 
                           const float4* src,
                           const float* kernel, 
                           float4* dst, 
                           const uint32_t pitchdst_v1, 
                           decx::cuda_stream* S)
{
    if (_fake_this->_conv_border_method != de::extend_label::_EXTEND_NONE_) 
    {
        checkCudaErrors(cudaMemcpy2DAsync((float*)_fake_this->_ext_src._ptr.ptr + (_fake_this->_kernel_layout->width >> 1),
            _fake_this->_ext_src._dims.x * sizeof(float),
            src,
            _fake_this->_src_layout->pitch * sizeof(float),
            _fake_this->_src_layout->width * sizeof(float),
            _fake_this->_src_layout->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));
        
        decx::dsp::GPUK::cu_filter2D_BC_fp32<_ext_w> << <_fake_this->_grid, _fake_this->_block, 
                                                    0, S->get_raw_stream_ref() >> > (
            (float4*)_fake_this->_ext_src._ptr.ptr,
            (float*)kernel,
            (float4*)dst,
            _fake_this->_ext_src._dims.x / 4,
            pitchdst_v1 / 4,
            make_uint3(_fake_this->_kernel_layout->width, 
                       _fake_this->_kernel_layout->height, 
                       _fake_this->_kernel_layout->pitch),
            _fake_this->_dst_dims);
    }
    else
    {
        decx::dsp::GPUK::cu_filter2D_NB_fp32<_ext_w> << <_fake_this->_grid, _fake_this->_block, 
                                                    0, S->get_raw_stream_ref() >> > (
            (float4*)src,
            (float*)kernel,
            (float4*)dst,
            _fake_this->_src_layout->pitch / 4,
            pitchdst_v1 / 4,
            make_uint3(_fake_this->_kernel_layout->width, 
                       _fake_this->_kernel_layout->height, 
                       _fake_this->_kernel_layout->pitch),
            _fake_this->_dst_dims);
    }
}



namespace decx
{
namespace dsp {
    decx::dsp::_cu_F2_FP32_Kcaller _cu_F2_FP32_Kcallers[32] = 
    {
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<4>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<8>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<12>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<16>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<20>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<24>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<28>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<32>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<36>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<40>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<44>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<48>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<52>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<56>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<60>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<64>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<68>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<72>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<76>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<80>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<84>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<88>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<92>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<96>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<100>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<104>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<108>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<112>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<116>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<120>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<124>,
        &decx::dsp::cuda_Filter2D_planner<float>::_cu_Filter2D_fp32_caller<128>,
    };
}
}


template <> void
decx::dsp::cuda_Filter2D_planner<float>::run(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel,
    decx::_GPU_Matrix* dst, decx::cuda_stream* S, de::DH* handle)
{
    _cu_F2_FP32_Kcaller _kernel_ptr = decx::dsp::_cu_F2_FP32_Kcallers[(this->_kernel_layout->width - 2) / 4];

    _kernel_ptr(this, (float4*)src->Mat.ptr, (float*)kernel->Mat.ptr, (float4*)dst->Mat.ptr, dst->get_layout().pitch, S);
}
