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


decx::ResourceHandle decx::dsp::_cuda_filter2D_u8;


template <> template <uint32_t _ext_w> 
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



template <> template <uint32_t _ext_w> 
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
    decx::dsp::_cu_F2_U8_Kcaller _cu_F2_U8_Kcallers[2][32] = {
        {
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<8>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<16>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<24>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<32>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<40>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<48>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<56>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<64>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<72>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<80>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<88>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<96>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<104>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<112>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<120>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<128>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<136>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<144>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<152>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<160>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<168>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<176>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<184>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<192>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<200>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<208>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<216>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<224>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<232>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<240>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<248>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_NB_u8_x_caller<256>,
        }, {
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<8>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<16>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<24>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<32>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<40>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<48>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<56>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<64>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<72>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<80>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<88>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<96>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<104>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<112>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<120>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<128>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<136>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<144>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<152>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<160>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<168>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<176>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<184>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<192>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<200>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<208>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<216>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<224>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<232>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<240>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<248>,
            &decx::dsp::cuda_Filter2D_planner<uint8_t>::_cu_Filter2D_BC_u8_x_caller<256>,
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

        decx::dsp::_cu_F2_U8_Kcaller _kernel_ptr = decx::dsp::_cu_F2_U8_Kcallers[1][(this->_kernel_layout->width - 2) / 8];

        _kernel_ptr(this, (double*)_ext_src._ptr.ptr, kernel->Mat.ptr, dst->Mat.ptr,
            dst->get_layout().pitch, S);
    }
    else
    {
        decx::dsp::_cu_F2_U8_Kcaller _kernel_ptr = decx::dsp::_cu_F2_U8_Kcallers[0][(this->_kernel_layout->width - 2) / 8];

        _kernel_ptr(this, (double*)src->Mat.ptr, kernel->Mat.ptr, dst->Mat.ptr,
            dst->get_layout().pitch, S);
    }
}
