/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "FFT2D_kernels.h"
#include "../../../../BLAS/basic_process/transpose/CPU/transpose_exec.h"
#include "CPU_FFT2D_planner.h"


template <>
template <typename _type_in>
void decx::dsp::fft::cpu_FFT2D_planner<float>::Forward(decx::_Matrix* src, 
                                                       decx::_Matrix* dst,
                                                       decx::utils::_thread_arrange_1D* t1D) const
{
    // Horizontal FFT
    decx::dsp::fft::_FFT2D_H_entire_rows_cplxf<_type_in, false>((_type_in*)src->Mat.ptr,                             
                                                                (double*)this->get_tmp1_ptr(), 
                                                                this,                                            
                                                                src->Pitch(), 
                                                                decx::utils::ceil<uint32_t>(src->Width(), 4) * 4,   
                                                                t1D, true);
    // Transpose
    decx::bp::transpose_2x2_caller((double*)this->get_tmp1_ptr(),                     
                                   (double*)this->get_tmp2_ptr(),
                                   decx::utils::ceil<uint32_t>(src->Width(), 4) * 4,   
                                   decx::utils::ceil<uint32_t>(src->Height(), 4) * 4, 
                                   &this->_transpose_config_1st,                              
                                   t1D);
    // Horizontal FFT
    decx::dsp::fft::_FFT2D_H_entire_rows_cplxf<double, true>((double*)this->get_tmp2_ptr(),         (double*)this->get_tmp1_ptr(), 
                                               this,                                                decx::utils::ceil<uint32_t>(src->Height(), 4) * 4,  
                                               decx::utils::ceil<uint32_t>(src->Height(), 4) * 4,   t1D, false);
    // Transpose
    decx::bp::transpose_2x2_caller((double*)this->get_tmp1_ptr(),                       (double*)dst->Mat.ptr,
                                   decx::utils::ceil<uint32_t>(src->Height(), 4) * 4,   dst->Pitch(), 
                                   &this->_transpose_config_2nd,                        t1D);
}

template void decx::dsp::fft::cpu_FFT2D_planner<float>::Forward<float>(decx::_Matrix*, decx::_Matrix*, decx::utils::_thread_arrange_1D*) const;
template void decx::dsp::fft::cpu_FFT2D_planner<float>::Forward<double>(decx::_Matrix*, decx::_Matrix*, decx::utils::_thread_arrange_1D*) const;
template void decx::dsp::fft::cpu_FFT2D_planner<float>::Forward<uint8_t>(decx::_Matrix*, decx::_Matrix*, decx::utils::_thread_arrange_1D*) const;



template <>
template <typename _type_out>
void decx::dsp::fft::cpu_FFT2D_planner<float>::Inverse(decx::_Matrix* src, 
                                                       decx::_Matrix* dst,
                                                       decx::utils::_thread_arrange_1D* t1D) const
{
    // Horizontal FFT
    decx::dsp::fft::_IFFT2D_H_entire_rows_cplxf<double>((double*)src->Mat.ptr,                
                                                        (double*)this->get_tmp1_ptr(), 
                                                        this,                                            
                                                        src->Pitch(), 
                                                        decx::utils::ceil<uint32_t>(src->Width(), 4) * 4,    
                                                        t1D, true);
                                                        
    // Transpose
    decx::bp::transpose_2x2_caller((double*)this->get_tmp1_ptr(),                     
                                   (double*)this->get_tmp2_ptr(),
                                   decx::utils::ceil<uint32_t>(src->Width(), 4) * 4,    
                                   decx::utils::ceil<uint32_t>(src->Height(), 4) * 4, 
                                   &this->_transpose_config_1st,                              
                                   t1D);
                                   
    // Horizontal FFT
    const uint8_t _STG_alignment = decx::dsp::fft::cpu_FFT2D_planner<float>::get_alignment_FFT_last_dimension<_type_out>();
    decx::dsp::fft::_IFFT2D_H_entire_rows_cplxf<_type_out>((double*)this->get_tmp2_ptr(),       
                                                           (_type_out*)this->get_tmp1_ptr(), 
                                                           this,                                            
                                                           decx::utils::ceil<uint32_t>(src->Height(), 4) * 4,  
                                                           decx::utils::ceil<uint32_t>(src->Height(), _STG_alignment) * _STG_alignment,
                                                           t1D, false);
                                                           
    // Transpose
    const decx::bp::_cpu_transpose_config<sizeof(_type_out)>* _transp_config_2nd_ptr =
        reinterpret_cast<const decx::bp::_cpu_transpose_config<sizeof(_type_out)>*>(&this->_transpose_config_2nd);

    if constexpr (std::is_same_v<_type_out, double>){
        decx::bp::transpose_2x2_caller((double*)this->get_tmp1_ptr(),                     (double*)dst->Mat.ptr,
                                       decx::utils::ceil<uint32_t>(src->Height(), 4) * 4,  dst->Pitch(), 
                                       _transp_config_2nd_ptr, t1D);
    }
    else if constexpr (std::is_same_v<_type_out, uint8_t>) {
        decx::bp::transpose_8x8_caller((double*)this->get_tmp1_ptr(),                     (double*)dst->Mat.ptr,
                                       decx::utils::ceil<uint32_t>(src->Height(), _STG_alignment) * _STG_alignment, dst->Pitch(),
                                       _transp_config_2nd_ptr, t1D);
    }
    else {
        decx::bp::transpose_4x4_caller((float*)this->get_tmp1_ptr(),                      (float*)dst->Mat.ptr,
                                       decx::utils::ceil<uint32_t>(src->Height(), _STG_alignment) * _STG_alignment, dst->Pitch(),
                                       _transp_config_2nd_ptr, t1D);
    }
}

template void decx::dsp::fft::cpu_FFT2D_planner<float>::Inverse<float>(decx::_Matrix*, decx::_Matrix*, decx::utils::_thread_arrange_1D*) const;
template void decx::dsp::fft::cpu_FFT2D_planner<float>::Inverse<double>(decx::_Matrix*, decx::_Matrix*, decx::utils::_thread_arrange_1D*) const;
template void decx::dsp::fft::cpu_FFT2D_planner<float>::Inverse<uint8_t>(decx::_Matrix*, decx::_Matrix*, decx::utils::_thread_arrange_1D*) const;


decx::dsp::fft::cpu_FFT2D_planner<float>* decx::dsp::fft::cpu_FFT2D_cplxf32_planner;
decx::dsp::fft::cpu_FFT2D_planner<float>* decx::dsp::fft::cpu_IFFT2D_cplxf32_planner;
