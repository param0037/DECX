/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _FFT1D_KERNEL_CALLERS_CUH_
#define _FFT1D_KERNEL_CALLERS_CUH_

#include "FFT1D_1st_kernels_dense.cuh"
#include "../2D/FFT2D_kernels.cuh"


namespace decx
{
namespace dsp {
namespace fft 
{
    /**
    * @brief : The function accepts pointer of double buffer, and sets tmp1 as the first priority.
    */
    template <typename _type_in, bool _div>
    void FFT1D_partition_cplxf_1st_caller(const void* src, decx::utils::double_buffer_manager* _double_buffer,
        const decx::dsp::fft::_FFT2D_1way_config* _FFT_info, decx::cuda_stream* S, const uint64_t _signal_len_total = 0);


    template <bool _div, bool _conj, typename _type_out>
    void FFT1D_partition_cplxf_end_caller(decx::utils::double_buffer_manager* _double_buffer, void* dst,
        const decx::dsp::fft::_FFT2D_1way_config* _FFT_info, decx::cuda_stream* S);
}
}
}


#endif