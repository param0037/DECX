/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "../cuda_Filter2D_planner.cuh"
#include "../filter2D_kernel.cuh"


decx::ResourceHandle decx::dsp::_cuda_filter2D_fp64;


template <> template <uint32_t _ext_w> void decx::dsp::cuda_Filter2D_planner<double>::
_cu_Filter2D_fp64_caller(const decx::dsp::cuda_Filter2D_planner<double>* _fake_this, 
                           const double2* src,
                           const double* kernel, 
                           double2* dst, 
                           const uint32_t pitchdst_v1, 
                           decx::cuda_stream* S)
{
    if (_fake_this->_conv_border_method != de::extend_label::_EXTEND_NONE_) 
    {
        checkCudaErrors(cudaMemcpy2DAsync((double*)_fake_this->_ext_src._ptr.ptr + (_fake_this->_kernel_layout->width >> 1),
            _fake_this->_ext_src._dims.x * sizeof(double),
            src,
            _fake_this->_src_layout->pitch * sizeof(double),
            _fake_this->_src_layout->width * sizeof(double),
            _fake_this->_src_layout->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));
        
        decx::dsp::GPUK::cu_filter2D_BC_fp64<_ext_w> << <_fake_this->_grid, _fake_this->_block, 
                                                    0, S->get_raw_stream_ref() >> > (
            (double2*)_fake_this->_ext_src._ptr.ptr,
            (double*)kernel,
            (double2*)dst,
            _fake_this->_ext_src._dims.x / 2,
            pitchdst_v1 / 2,
            make_uint3(_fake_this->_kernel_layout->width, 
                       _fake_this->_kernel_layout->height, 
                       _fake_this->_kernel_layout->pitch),
            _fake_this->_dst_dims);
    }
    else {
        decx::dsp::GPUK::cu_filter2D_NB_fp64<_ext_w> << <_fake_this->_grid, _fake_this->_block, 
                                                    0, S->get_raw_stream_ref() >> > (
            (double2*)src,
            (double*)kernel,
            (double2*)dst,
            _fake_this->_src_layout->pitch / 2,
            pitchdst_v1 / 2,
            make_uint3(_fake_this->_kernel_layout->width, 
                       _fake_this->_kernel_layout->height, 
                       _fake_this->_kernel_layout->pitch),
            _fake_this->_dst_dims);
    }
}



namespace decx
{
namespace dsp {
    decx::dsp::_cu_F2_FP64_Kcaller _cu_F2_FP64_Kcallers[32] = 
    {
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<2>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<4>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<6>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<8>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<10>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<12>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<14>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<16>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<18>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<20>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<22>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<24>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<26>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<28>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<30>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<32>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<34>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<36>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<38>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<40>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<42>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<44>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<46>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<48>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<50>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<52>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<54>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<56>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<58>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<60>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<62>,
        &decx::dsp::cuda_Filter2D_planner<double>::_cu_Filter2D_fp64_caller<64>,
    };
}
}



template <> void
decx::dsp::cuda_Filter2D_planner<double>::run(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel,
    decx::_GPU_Matrix* dst, decx::cuda_stream* S, de::DH* handle)
{
    _cu_F2_FP64_Kcaller _kernel_ptr = decx::dsp::_cu_F2_FP64_Kcallers[(this->_kernel_layout->width - 2) / 2];

    _kernel_ptr(this, (double2*)src->Mat.ptr, (double*)kernel->Mat.ptr, (double2*)dst->Mat.ptr, dst->get_layout().pitch, S);
}
