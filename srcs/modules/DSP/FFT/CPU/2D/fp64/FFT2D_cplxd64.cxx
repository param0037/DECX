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


#include "../FFT2D_kernels.h"
#include "../../../../../../common/Basic_process/transpose/CPU/transpose2D_config.h"
#include "../CPU_FFT2D_planner.h"


decx::ResourceHandle decx::dsp::fft::g_cpu_FFT2D_cplxd64_planner;
decx::ResourceHandle decx::dsp::fft::g_cpu_IFFT2D_cplxd64_planner;

template <>
template <typename _type_in>
void decx::dsp::fft::cpu_FFT2D_planner<double>::Forward(decx::_Matrix* src, 
                                                       decx::_Matrix* dst,
                                                       decx::utils::_thread_arrange_1D* t1D) const
{
    // The alignment is always 2 in case where _op_data_type = de::CPd
    // Horizontal FFT
    decx::dsp::fft::_FFT2D_H_entire_rows_cplxd<_type_in, false>((_type_in*)src->Mat.ptr,                             
                                                                (de::CPd*)this->get_tmp1_ptr(), 
                                                                this,                                            
                                                                src->Pitch(), 
                                                                decx::utils::align<uint32_t>(src->Width(), 2),   
                                                                t1D, true);
    // Transpose
    this->_transpose_config_1st.transpose_16b_caller((de::CPd*)this->get_tmp1_ptr(),
                                                     (de::CPd*)this->get_tmp2_ptr(),
                                                     decx::utils::align<uint32_t>(src->Width(), 2),
                                                     decx::utils::align<uint32_t>(src->Height(), 2),
                                                     t1D);

    // Horizontal FFT
    decx::dsp::fft::_FFT2D_H_entire_rows_cplxd<de::CPd, true>((de::CPd*)this->get_tmp2_ptr(),       (de::CPd*)this->get_tmp1_ptr(), 
                                               this,                                                decx::utils::align<uint32_t>(src->Height(), 2),  
                                               decx::utils::align<uint32_t>(src->Height(), 2),   t1D, false);
    // Transpose
    this->_transpose_config_2nd.transpose_16b_caller((de::CPd*)this->get_tmp1_ptr(), 
                                                     (de::CPd*)dst->Mat.ptr,
                                                     decx::utils::align<uint32_t>(src->Height(), 2), 
                                                     dst->Pitch(), 
                                                     t1D);
}

template void decx::dsp::fft::cpu_FFT2D_planner<double>::Forward<double>(decx::_Matrix*, decx::_Matrix*, decx::utils::_thread_arrange_1D*) const;
template void decx::dsp::fft::cpu_FFT2D_planner<double>::Forward<de::CPd>(decx::_Matrix*, decx::_Matrix*, decx::utils::_thread_arrange_1D*) const;
template void decx::dsp::fft::cpu_FFT2D_planner<double>::Forward<uint8_t>(decx::_Matrix*, decx::_Matrix*, decx::utils::_thread_arrange_1D*) const;



template <>
template <typename _type_out>
void decx::dsp::fft::cpu_FFT2D_planner<double>::Inverse(decx::_Matrix* src, 
                                                       decx::_Matrix* dst,
                                                       decx::utils::_thread_arrange_1D* t1D) const
{
    // The alignment is always 2 in case where _op_data_type = de::CPd
    // Horizontal FFT
    decx::dsp::fft::_IFFT2D_H_entire_rows_cplxd<de::CPd>((de::CPd*)src->Mat.ptr,                
                                                        (de::CPd*)this->get_tmp1_ptr(), 
                                                        this,                                            
                                                        src->Pitch(), 
                                                        decx::utils::align<uint32_t>(src->Width(), 2),    
                                                        t1D, true);
                                                        
    // Transpose
    this->_transpose_config_1st.transpose_16b_caller((de::CPd*)this->get_tmp1_ptr(),
        (de::CPd*)this->get_tmp2_ptr(),
        decx::utils::align<uint32_t>(src->Width(), 2),
        decx::utils::align<uint32_t>(src->Height(), 2),
        t1D);

    // Horizontal FFT
    const uint8_t _STG_alignment = decx::dsp::fft::cpu_FFT2D_planner<double>::get_alignment_FFT_last_dimension<_type_out>();

    decx::dsp::fft::_IFFT2D_H_entire_rows_cplxd<_type_out>((de::CPd*)this->get_tmp2_ptr(),       
                                                           (_type_out*)this->get_tmp1_ptr(), 
                                                           this,                                    
                                                           decx::utils::align<uint32_t>(src->Height(), 2),  
                                                           decx::utils::align<uint32_t>(src->Height(), _STG_alignment),
                                                           t1D, false);
                                                           
    // Transpose
    if constexpr (std::is_same_v<_type_out, de::CPd>){
        this->_transpose_config_2nd.
            transpose_16b_caller((de::CPd*)this->get_tmp1_ptr(), 
                                (de::CPd*)dst->Mat.ptr,
                                decx::utils::align<uint32_t>(src->Height(), 2), 
                                dst->Pitch(),
                                t1D);
    }
    else if constexpr (std::is_same_v<_type_out, uint8_t>) {
        this->_transpose_config_2nd.
            transpose_1b_caller((uint64_t*)this->get_tmp1_ptr(), 
                                (uint64_t*)dst->Mat.ptr,
                                decx::utils::align<uint32_t>(src->Height(), _STG_alignment) / 8, 
                                dst->Pitch() / 8,
                                t1D);
    }
    else {
        this->_transpose_config_2nd.
            transpose_8b_caller((double*)this->get_tmp1_ptr(), 
                                (double*)dst->Mat.ptr,
                                decx::utils::align<uint32_t>(src->Height(), _STG_alignment), 
                                dst->Pitch(),
                                t1D);
    }
}

template void decx::dsp::fft::cpu_FFT2D_planner<double>::Inverse<double>(decx::_Matrix*, decx::_Matrix*, decx::utils::_thread_arrange_1D*) const;
template void decx::dsp::fft::cpu_FFT2D_planner<double>::Inverse<de::CPd>(decx::_Matrix*, decx::_Matrix*, decx::utils::_thread_arrange_1D*) const;
template void decx::dsp::fft::cpu_FFT2D_planner<double>::Inverse<uint8_t>(decx::_Matrix*, decx::_Matrix*, decx::utils::_thread_arrange_1D*) const;
