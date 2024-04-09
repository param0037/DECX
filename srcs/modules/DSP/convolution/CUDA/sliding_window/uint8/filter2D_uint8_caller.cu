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


template <> template <uint8_t _ext_w> 
void decx::dsp::cuda_Filter2D_planner<uint8_t>::
_cu_Filter2D_NB_u8_x_caller(const decx::dsp::cuda_Filter2D_planner<uint8_t>* _fake_this,
                            const double* src, 
                            const void* kernel,
                            void* dst, 
                            const uint32_t pitchdst_v1, 
                            decx::cuda_stream* S)
{
    switch (_fake_this->_output_type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::GPUK::cu_filter2D_NB_u8_Kfp32_fp32<_ext_w> << <_fake_this->_grid, _fake_this->_block, 
                                                              0, S->get_raw_stream_ref() >> > (
            src,
            (float*)kernel,
            (float4*)dst,
            _fake_this->_src_layout->pitch / 8,
            pitchdst_v1 / 8,
            make_uint3(_fake_this->_kernel_layout->width, 
                       _fake_this->_kernel_layout->height, 
                       _fake_this->_kernel_layout->pitch),
            _fake_this->_dst_dims);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::dsp::GPUK::cu_filter2D_NB_u8_Kfp32_u8<_ext_w> << <_fake_this->_grid, _fake_this->_block,
                                                            0, S->get_raw_stream_ref() >> > (
            src,
            (float*)kernel,
            (double*)dst,
            _fake_this->_src_layout->pitch / 8,
            pitchdst_v1 / 8,
            make_uint3(_fake_this->_kernel_layout->width, 
                       _fake_this->_kernel_layout->height, 
                       _fake_this->_kernel_layout->pitch),
            _fake_this->_dst_dims);
        break;

    default:
        break;
    }
}



template <> template <uint8_t _ext_w> 
void decx::dsp::cuda_Filter2D_planner<uint8_t>::
_cu_Filter2D_BC_u8_x_caller(const decx::dsp::cuda_Filter2D_planner<uint8_t>* _fake_this,
                            const double* src, 
                            const void* kernel,
                            void* dst, 
                            const uint32_t pitchdst_v1, 
                            decx::cuda_stream* S)
{
    switch (_fake_this->_output_type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::dsp::GPUK::cu_filter2D_BC_u8_Kfp32_fp32<_ext_w> << <_fake_this->_grid, _fake_this->_block, 
                                                                  0, S->get_raw_stream_ref() >> > (
            src,
            (float*)kernel,
            (float4*)dst,
            _fake_this->_ext_src._dims.x / 8,
            pitchdst_v1 / 8,
            make_uint3(_fake_this->_kernel_layout->width, 
                       _fake_this->_kernel_layout->height, 
                       _fake_this->_kernel_layout->pitch),
            _fake_this->_dst_dims);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        decx::dsp::GPUK::cu_filter2D_BC_u8_Kfp32_u8<_ext_w> << <_fake_this->_grid, _fake_this->_block, 
                                                                0, S->get_raw_stream_ref() >> > (
            src,
            (float*)kernel,
            (double*)dst,
            _fake_this->_ext_src._dims.x / 8,
            pitchdst_v1 / 8,
            make_uint3(_fake_this->_kernel_layout->width, 
                       _fake_this->_kernel_layout->height, 
                       _fake_this->_kernel_layout->pitch),
            _fake_this->_dst_dims);
        break;

    default:
        break;
    }
}



namespace decx
{
namespace dsp {
    static decx::dsp::_cu_F2_U8_Kcaller _cu_F2_U8_Kcallers[2][4] = { 
        {
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<8>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<16>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<24>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<32>,
        }, {
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<8>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<16>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<24>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<32>,
        }
    };
}
}


template <> void
decx::dsp::cuda_Filter2D_planner<uint8_t>::run(decx::_GPU_Matrix* src, decx::_GPU_Matrix* kernel,
    decx::_GPU_Matrix* dst, decx::cuda_stream* S, de::DH* handle)
{
    if (this->_conv_border_method != de::extend_label::_EXTEND_NONE_) 
    {
        checkCudaErrors(cudaMemcpy2DAsync((uint8_t*)this->_ext_src._ptr.ptr + (this->_kernel_layout->width >> 1),
            _ext_src._dims.x * sizeof(uint8_t),
            src->Mat.ptr,
            this->_src_layout->pitch * sizeof(uint8_t),
            this->_src_layout->width * sizeof(uint8_t),
            this->_src_layout->height,
            cudaMemcpyDeviceToDevice,
            S->get_raw_stream_ref()));
        
        decx::dsp::cuda_Filter2D_planner<uint8_t>::
            _cu_Filter2D_BC_u8_x_caller<32>(this, (double*)_ext_src._ptr.ptr, kernel->Mat.ptr, dst->Mat.ptr,
            dst->get_layout().pitch, S);
    }
    else
    {
        decx::dsp::cuda_Filter2D_planner<uint8_t>::
            _cu_Filter2D_NB_u8_x_caller<32>(this, (double*)src->Mat.ptr, kernel->Mat.ptr, dst->Mat.ptr,
            dst->get_layout().pitch, S);
    }
}
