/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _FFT3D_MIDPROC_CALLER_CUH_
#define _FFT3D_MIDPROC_CALLER_CUH_


#include "FFT3D_kernels.cuh"
#include "FFT3D_planner.cuh"
#include "../../FFT_commons.h"


namespace decx
{
namespace dsp {
    namespace fft 
    {
        template <bool _div>
        void FFT3D_cplxf_1st_1way_caller(decx::utils::double_buffer_manager* _double_buffer,
            const decx::dsp::fft::_cuda_FFT3D_mid_config* _FFT_info, decx::cuda_stream* S);
    }
}
}


#endif
