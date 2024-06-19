/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "CPU_FFT3D_planner.h"
#include "../FFT.h"
#include "../../FFT_commons.h"
#include "FFT3D_kernels.h"


template <>
template <typename _type_in>
void _CRSR_ decx::dsp::fft::cpu_FFT3D_planner<float>::Forward(decx::_Tensor* src, decx::_Tensor* dst) const
{
    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());

    // FFT along depth
    decx::dsp::fft::_FFT3D_H_entire_rows_cplxf<_type_in, false>((const _type_in*)src->Tens.ptr,
        (de::CPf*)this->get_tmp1_ptr(),
        this,
        &t1D,
        decx::dsp::fft::FFT_directions::_FFT_AlongD);

    // Transpose multi-channel
    decx::bp::transpose_MK_2x2_caller((double*)this->get_tmp1_ptr(), 
        (double*)this->get_tmp2_ptr(),
        this->_FFT_D._pitchdst, 
        this->_FFT_W._pitchsrc, 
        &this->_transp_config_MC,
        &t1D);

    // FFT along width
    decx::dsp::fft::_FFT3D_H_entire_rows_cplxf<de::CPf, false>((de::CPf*)this->get_tmp2_ptr(), 
        (de::CPf*)this->get_tmp1_ptr(), 
        this,
        &t1D, 
        decx::dsp::fft::FFT_directions::_FFT_AlongW);

    // Transpose multi-channel back
    decx::bp::transpose_MK_2x2_caller((double*)this->get_tmp1_ptr(),
        (double*)this->get_tmp2_ptr(),
        this->_FFT_W._pitchdst,
        this->_FFT_D._pitchdst,
        &this->_transp_config_MC_back,
        &t1D);

    // Transpose [DPxW, H] to [DH, DPXW]
    decx::bp::transpose_2x2_caller((double*)this->get_tmp2_ptr(), 
                                   (double*)this->get_tmp1_ptr(),
                                   this->_FFT_D._pitchdst * src->Width(), 
                                   this->_FFT_H._pitchsrc, 
                                   &this->_transp_config, 
                                   &t1D);
    
    decx::dsp::fft::_FFT3D_H_entire_rows_cplxf<de::CPf, true>((de::CPf*)this->get_tmp1_ptr(),
        (de::CPf*)this->get_tmp2_ptr(),
        this,
        &t1D,
        decx::dsp::fft::FFT_directions::_FFT_AlongH);
    
    decx::bp::transpose_2x2_caller((double*)this->get_tmp2_ptr(),
                                   (double*)dst->Tens.ptr,
                                   this->_FFT_H._pitchdst, 
                                   dst->get_layout().dp_x_wp, 
                                   &this->_transp_config_back, 
                                   &t1D);
}

template void _CRSR_ decx::dsp::fft::cpu_FFT3D_planner<float>::Forward<float>(decx::_Tensor*, decx::_Tensor*) const;
template void _CRSR_ decx::dsp::fft::cpu_FFT3D_planner<float>::Forward<de::CPf>(decx::_Tensor*, decx::_Tensor*) const;
template void _CRSR_ decx::dsp::fft::cpu_FFT3D_planner<float>::Forward<uint8_t>(decx::_Tensor*, decx::_Tensor*) const;


template <>
template <typename _type_out>
void _CRSR_ decx::dsp::fft::cpu_FFT3D_planner<float>::Inverse(decx::_Tensor* src, decx::_Tensor* dst) const
{
    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());

    // FFT along depth
    decx::dsp::fft::_IFFT3D_H_entire_rows_cplxf<de::CPf>((const de::CPf*)src->Tens.ptr,
        (de::CPf*)this->get_tmp1_ptr(),
        this,
        &t1D,
        decx::dsp::fft::FFT_directions::_FFT_AlongD);

    // Transpose multi-channel
    decx::bp::transpose_MK_2x2_caller((double*)this->get_tmp1_ptr(), 
        (double*)this->get_tmp2_ptr(),
        this->_FFT_D._pitchdst, 
        this->_FFT_W._pitchsrc, 
        &this->_transp_config_MC,
        &t1D);

    // FFT along width
    decx::dsp::fft::_IFFT3D_H_entire_rows_cplxf<de::CPf>((de::CPf*)this->get_tmp2_ptr(), 
        (de::CPf*)this->get_tmp1_ptr(), 
        this,
        &t1D, 
        decx::dsp::fft::FFT_directions::_FFT_AlongW);

    // Transpose multi-channel back
    decx::bp::transpose_MK_2x2_caller((double*)this->get_tmp1_ptr(),
        (double*)this->get_tmp2_ptr(),
        this->_FFT_W._pitchdst,
        this->_FFT_D._pitchdst,
        &this->_transp_config_MC_back,
        &t1D);

    // Transpose [DPxW, H] to [DH, DPXW]
    decx::bp::transpose_2x2_caller((double*)this->get_tmp2_ptr(), 
                                   (double*)this->get_tmp1_ptr(),
                                   this->_FFT_D._pitchdst * src->Width(), 
                                   this->_FFT_H._pitchsrc, 
                                   &this->_transp_config, 
                                   &t1D);
    
    decx::dsp::fft::_IFFT3D_H_entire_rows_cplxf<_type_out>((de::CPf*)this->get_tmp1_ptr(),
        (_type_out*)this->get_tmp2_ptr(),
        this,
        &t1D,
        decx::dsp::fft::FFT_directions::_FFT_AlongH);

    decx::bp::_cpu_transpose_config<sizeof(_type_out)>* transp_config_back =
        (decx::bp::_cpu_transpose_config<sizeof(_type_out)>*)(&this->_transp_config_back);
    
    if constexpr (std::is_same_v<_type_out, float>) {
        decx::bp::transpose_4x4_caller((float*)this->get_tmp2_ptr(),
                                       (float*)dst->Tens.ptr,
                                       this->_FFT_H._pitchdst, 
                                       dst->get_layout().dp_x_wp, 
                                       transp_config_back, 
                                       &t1D);
    }
    else if constexpr (std::is_same_v<_type_out, uint8_t>) {
        decx::bp::transpose_8x8_caller((double*)this->get_tmp2_ptr(),
                                       (double*)dst->Tens.ptr,
                                       this->_FFT_H._pitchdst, 
                                       dst->get_layout().dp_x_wp, 
                                       transp_config_back, 
                                       &t1D);
    }
    else {
        decx::bp::transpose_2x2_caller((double*)this->get_tmp2_ptr(),
                                       (double*)dst->Tens.ptr,
                                       this->_FFT_H._pitchdst, 
                                       dst->get_layout().dp_x_wp, 
                                       transp_config_back, 
                                       &t1D);
    }
}

template void _CRSR_ decx::dsp::fft::cpu_FFT3D_planner<float>::Inverse<float>(decx::_Tensor*, decx::_Tensor*) const;
template void _CRSR_ decx::dsp::fft::cpu_FFT3D_planner<float>::Inverse<de::CPf>(decx::_Tensor*, decx::_Tensor*) const;
template void _CRSR_ decx::dsp::fft::cpu_FFT3D_planner<float>::Inverse<uint8_t>(decx::_Tensor*, decx::_Tensor*) const;


decx::ResourceHandle decx::dsp::fft::FFT3D_cplxf32_planner;
decx::ResourceHandle decx::dsp::fft::IFFT3D_cplxf32_planner;
