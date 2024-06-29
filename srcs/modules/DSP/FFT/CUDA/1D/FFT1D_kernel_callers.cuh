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


#ifndef _FFT1D_KERNEL_CALLERS_CUH_
#define _FFT1D_KERNEL_CALLERS_CUH_

#include "FFT1D_1st_kernels_dense.cuh"
#include "../2D/FFT2D_kernels.cuh"


namespace decx
{
namespace dsp {
namespace fft 
{
    // ------------------------------------------------------ float ------------------------------------------------------
    /**
    * @brief : The function accepts pointer of double buffer, and sets tmp1 as the first priority.
    */
    template <typename _type_in, bool _div>
    void FFT1D_partition_cplxf_1st_caller(const void* src, decx::utils::double_buffer_manager* _double_buffer,
        const decx::dsp::fft::_FFT2D_1way_config* _FFT_info, decx::cuda_stream* S, const uint64_t _signal_len_total = 0);


    template <bool _div, bool _conj, typename _type_out>
    void FFT1D_partition_cplxf_end_caller(decx::utils::double_buffer_manager* _double_buffer, void* dst,
        const decx::dsp::fft::_FFT2D_1way_config* _FFT_info, decx::cuda_stream* S);

    // ------------------------------------------------------ double ------------------------------------------------------

    /**
    * @brief : The function accepts pointer of double buffer, and sets tmp1 as the first priority.
    */
    template <typename _type_in, bool _div>
    void FFT1D_partition_cplxd_1st_caller(const void* src, decx::utils::double_buffer_manager* _double_buffer,
        const decx::dsp::fft::_FFT2D_1way_config* _FFT_info, decx::cuda_stream* S, const uint64_t _signal_len_total = 0);


    template <bool _div, bool _conj, typename _type_out>
    void FFT1D_partition_cplxd_end_caller(decx::utils::double_buffer_manager* _double_buffer, void* dst,
        const decx::dsp::fft::_FFT2D_1way_config* _FFT_info, decx::cuda_stream* S);
}
}
}


#endif