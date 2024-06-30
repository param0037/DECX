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


#include "../../FFT.h"
#include "../CPU_FFT1D_planner.h"
#include "../FFT1D_kernels.h"


template <>
template <typename _type_in>
void decx::dsp::fft::cpu_FFT1D_planner<float>::Forward(decx::_Vector* src, decx::_Vector* dst, decx::utils::_thread_arrange_1D* t1D) const
{
    decx::utils::double_buffer_manager _double_buffer(this->get_tmp1_ptr(), 
                                                      this->get_tmp2_ptr());
    
    decx::dsp::fft::FIMT1D _FIMT1D(this->get_signal_len(), 
                                   this->get_smaller_FFT_info_ptr(0)->get_signal_len(),
                                   this->get_smaller_FFT_info_ptr(1)->get_signal_len());
    
    if (this->get_kernel_call_num() > 1) {
        decx::dsp::fft::_FFT1D_cplxf32_1st<false, _type_in>((const _type_in*)src->Vec.ptr, 
                                                     (de::CPf*)_double_buffer._MIF1.mem, 
                                                     this, t1D, &_FIMT1D);

        _double_buffer.reset_buffer1_leading();
    }
    else {
        decx::dsp::fft::_FFT1D_cplxf32_1st<false, _type_in>((const _type_in*)src->Vec.ptr, 
                                                     (de::CPf*)dst->Vec.ptr, 
                                                     this, t1D, NULL);
    }

    for (uint32_t i = 1; i < this->get_kernel_call_num(); ++i)
    {
        if (i < this->get_kernel_call_num() - 1) {
            _FIMT1D.update(this->get_smaller_FFT_info_ptr(i + 1)->get_signal_len());

            decx::dsp::fft::_FFT1D_cplxf32_mid<de::CPf, false>(_double_buffer.get_leading_ptr<const de::CPf>(), 
                                                          _double_buffer.get_lagging_ptr<de::CPf>(), 
                                                          this, t1D, 
                                                          i, &_FIMT1D);
        }
        else {
            decx::dsp::fft::_FFT1D_cplxf32_mid<de::CPf, true>(_double_buffer.get_leading_ptr<const de::CPf>(), 
                                                         (de::CPf*)dst->Vec.ptr, 
                                                         this, t1D, 
                                                         i, NULL);
        }

        _double_buffer.update_states();
    }
}

template void decx::dsp::fft::cpu_FFT1D_planner<float>::Forward<float>(decx::_Vector*, decx::_Vector*, decx::utils::_thread_arrange_1D*) const;
template void decx::dsp::fft::cpu_FFT1D_planner<float>::Forward<de::CPf>(decx::_Vector*, decx::_Vector*, decx::utils::_thread_arrange_1D*) const;



template <>
template <typename _type_out>
void decx::dsp::fft::cpu_FFT1D_planner<float>::Inverse(decx::_Vector* src, decx::_Vector* dst, decx::utils::_thread_arrange_1D* t1D) const
{
    decx::utils::double_buffer_manager _double_buffer(this->get_tmp1_ptr(), 
                                                      this->get_tmp2_ptr());
    
    decx::dsp::fft::FIMT1D _FIMT1D(this->get_signal_len(), 
                                   this->get_smaller_FFT_info_ptr(0)->get_signal_len(),
                                   this->get_smaller_FFT_info_ptr(1)->get_signal_len());
    
    if (this->get_kernel_call_num() > 1) {
        decx::dsp::fft::_FFT1D_cplxf32_1st<true, de::CPf>((const de::CPf*)src->Vec.ptr,
                                                   (de::CPf*)_double_buffer._MIF1.mem, 
                                                   this, t1D, &_FIMT1D);

        _double_buffer.reset_buffer1_leading();
    }
    else {
        decx::dsp::fft::_FFT1D_cplxf32_1st<true, de::CPf>((const de::CPf*)src->Vec.ptr, 
                                                   (de::CPf*)dst->Vec.ptr, 
                                                   this, t1D, NULL);
    }

    for (uint32_t i = 1; i < this->get_kernel_call_num(); ++i)
    {
        if (i < this->get_kernel_call_num() - 1) {
            _FIMT1D.update(this->get_smaller_FFT_info_ptr(i + 1)->get_signal_len());

            decx::dsp::fft::_FFT1D_cplxf32_mid<de::CPf, false>(_double_buffer.get_leading_ptr<const de::CPf>(),
                                                          _double_buffer.get_lagging_ptr<de::CPf>(), 
                                                          this, t1D, 
                                                          i, &_FIMT1D);
        }
        else {
            decx::dsp::fft::_FFT1D_cplxf32_mid<_type_out, false>(_double_buffer.get_leading_ptr<const de::CPf>(),
                                                         (_type_out*)dst->Vec.ptr,
                                                         this, t1D, 
                                                         i, NULL);
        }

        _double_buffer.update_states();
    }
}

template void decx::dsp::fft::cpu_FFT1D_planner<float>::Inverse<float>(decx::_Vector*, decx::_Vector*, decx::utils::_thread_arrange_1D*) const;
template void decx::dsp::fft::cpu_FFT1D_planner<float>::Inverse<de::CPf>(decx::_Vector*, decx::_Vector*, decx::utils::_thread_arrange_1D*) const;


decx::ResourceHandle decx::dsp::fft::g_cpu_FFT1D_cplxf32_planner;
decx::ResourceHandle decx::dsp::fft::g_cpu_IFFT1D_cplxf32_planner;
