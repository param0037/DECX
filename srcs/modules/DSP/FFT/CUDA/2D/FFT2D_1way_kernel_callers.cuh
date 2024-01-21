/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _FFT2D_1WAY_KERNEL_CALLERS_CUH_
#define _FFT2D_1WAY_KERNEL_CALLERS_CUH_


#include "FFT2D_kernels.cuh"


namespace decx
{
namespace dsp {
namespace fft 
{
    template <typename _type_in, bool _div>
    void FFT2D_cplxf_1st_1way_caller(const void* src, decx::utils::double_buffer_manager* _double_buffer,
        const decx::dsp::fft::_FFT2D_1way_config* _FFT_info, decx::cuda_stream* S);


    template <bool _div, bool _conj, typename _type_out>
    void FFT2D_C2C_cplxf_1way_caller(decx::utils::double_buffer_manager* _double_buffer,
        const decx::dsp::fft::_FFT2D_1way_config* _FFT_info, decx::cuda_stream* S);
}
}
}



#endif