/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "../FFT1D.h"
#include "../CPU_FFT1D_planner.h"
#include "../FFT1D_kernels.h"
#include "../FFT1D_kernel_utils.h"


template <>
template <typename _type_in>
void decx::dsp::fft::cpu_FFT1D_planner<double>::Forward(decx::_Vector* src, decx::_Vector* dst, decx::utils::_thread_arrange_1D* t1D) const
{
    decx::utils::double_buffer_manager _double_buffer(this->get_tmp1_ptr(), 
                                                      this->get_tmp2_ptr());
    
    decx::dsp::fft::FIMT1D _FIMT1D(this->get_signal_len(), 
                                   this->get_smaller_FFT_info_ptr(0)->get_signal_len(),
                                   this->get_smaller_FFT_info_ptr(1)->get_signal_len());

    if (this->get_kernel_call_num() > 1) {
        decx::dsp::fft::_FFT1D_cplxd64_1st<false, _type_in>((const _type_in*)src->Vec.ptr, 
                                                     (de::CPd*)_double_buffer._MIF1.mem, 
                                                     this, t1D, &_FIMT1D);

        _double_buffer.reset_buffer1_leading();
    }
    else {
        decx::dsp::fft::_FFT1D_cplxd64_1st<false, _type_in>((const _type_in*)src->Vec.ptr, 
                                                     (de::CPd*)dst->Vec.ptr, 
                                                     this, t1D, NULL);
    }

    for (uint32_t i = 1; i < this->get_kernel_call_num(); ++i)
    {
        if (i < this->get_kernel_call_num() - 1) {
            _FIMT1D.update(this->get_smaller_FFT_info_ptr(i + 1)->get_signal_len());

            decx::dsp::fft::_FFT1D_cplxd64_mid<de::CPd, false>(_double_buffer.get_leading_ptr<const de::CPd>(), 
                                                          _double_buffer.get_lagging_ptr<de::CPd>(), 
                                                          this, t1D, 
                                                          i, &_FIMT1D);
        }
        else {
            decx::dsp::fft::_FFT1D_cplxd64_mid<de::CPd, true>(_double_buffer.get_leading_ptr<const de::CPd>(), 
                                                         (de::CPd*)dst->Vec.ptr, 
                                                         this, t1D, 
                                                         i, NULL);
        }

        _double_buffer.update_states();
    }
}

template void decx::dsp::fft::cpu_FFT1D_planner<double>::Forward<double>(decx::_Vector*, decx::_Vector*, decx::utils::_thread_arrange_1D*) const;
template void decx::dsp::fft::cpu_FFT1D_planner<double>::Forward<de::CPd>(decx::_Vector*, decx::_Vector*, decx::utils::_thread_arrange_1D*) const;




template <>
template <typename _type_out>
void decx::dsp::fft::cpu_FFT1D_planner<double>::Inverse(decx::_Vector* src, decx::_Vector* dst, decx::utils::_thread_arrange_1D* t1D) const
{
    decx::utils::double_buffer_manager _double_buffer(this->get_tmp1_ptr(), 
                                                      this->get_tmp2_ptr());
    
    decx::dsp::fft::FIMT1D _FIMT1D(this->get_signal_len(), 
                                   this->get_smaller_FFT_info_ptr(0)->get_signal_len(),
                                   this->get_smaller_FFT_info_ptr(1)->get_signal_len());
    
    if (this->get_kernel_call_num() > 1) {
        decx::dsp::fft::_FFT1D_cplxd64_1st<true, de::CPd>((const de::CPd*)src->Vec.ptr, 
                                                   (de::CPd*)_double_buffer._MIF1.mem, 
                                                   this, t1D, &_FIMT1D);

        _double_buffer.reset_buffer1_leading();
    }
    else {
        decx::dsp::fft::_FFT1D_cplxd64_1st<true, de::CPd>((const de::CPd*)src->Vec.ptr, 
                                                   (de::CPd*)dst->Vec.ptr, 
                                                   this, t1D, NULL);
    }

    for (uint32_t i = 1; i < this->get_kernel_call_num(); ++i)
    {
        if (i < this->get_kernel_call_num() - 1) {
            _FIMT1D.update(this->get_smaller_FFT_info_ptr(i + 1)->get_signal_len());

            decx::dsp::fft::_FFT1D_cplxd64_mid<de::CPd, false>(_double_buffer.get_leading_ptr<const de::CPd>(),
                                                          _double_buffer.get_lagging_ptr<de::CPd>(), 
                                                          this, t1D, 
                                                          i, &_FIMT1D);
        }
        else {
            decx::dsp::fft::_FFT1D_cplxd64_mid<_type_out, false>(_double_buffer.get_leading_ptr<const de::CPd>(),
                                                         (_type_out*)dst->Vec.ptr,
                                                         this, t1D, 
                                                         i, NULL);
        }

        _double_buffer.update_states();
    }
}

template void decx::dsp::fft::cpu_FFT1D_planner<double>::Inverse<double>(decx::_Vector*, decx::_Vector*, decx::utils::_thread_arrange_1D*) const;
template void decx::dsp::fft::cpu_FFT1D_planner<double>::Inverse<de::CPd>(decx::_Vector*, decx::_Vector*, decx::utils::_thread_arrange_1D*) const;


decx::ResourceHandle decx::dsp::fft::cpu_FFT1D_cplxd64_planner;
decx::ResourceHandle decx::dsp::fft::cpu_IFFT1D_cplxd64_planner;
