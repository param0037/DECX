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


#ifndef _FFT1D_KERNELS_H_
#define _FFT1D_KERNELS_H_


#include "../../../../../common/basic.h"
#include "../../../../core/thread_management/thread_pool.h"
#include "../../../../core/thread_management/thread_arrange.h"
#include "../FFT_common/CPU_FFT_tiles.h"
#include "../../../../../common/FMGR/fragment_arrangment.h"
#include "CPU_FFT1D_planner.h"


namespace decx
{
namespace dsp {
namespace fft {
    namespace CPUK {
        // Radix-4
        _THREAD_CALL_ void
        _FFT1D_R4_cplxf32_1st_R2C(const float* __restrict src, de::CPf* __restrict dst, const uint32_t signal_length);


        _THREAD_CALL_ void
        _FFT1D_R4_cplxf32_1st_C2C(const de::CPf* __restrict src, de::CPf* __restrict dst, const uint32_t signal_length);


        _THREAD_CALL_ void
        _FFT1D_R4_cplxf32_mid_C2C(const de::CPf* __restrict src, de::CPf* __restrict dst, const de::CPf* __restrict _W_table,
            const decx::dsp::fft::FKI1D* _kernel_info);


        // Radix-3
        _THREAD_CALL_ void
        _FFT1D_R3_cplxf32_1st_R2C(const float* __restrict src, de::CPf* __restrict dst, const uint32_t signal_length);


        _THREAD_CALL_ void
        _FFT1D_R3_cplxf32_1st_C2C(const de::CPf* __restrict src, de::CPf* __restrict dst, const uint32_t signal_length);


        _THREAD_CALL_ void
        _FFT1D_R3_cplxf32_mid_C2C(const de::CPf* __restrict src, de::CPf* __restrict dst, const de::CPf* __restrict _W_table,
            const decx::dsp::fft::FKI1D* _kernel_info);


        // Radix-5
        _THREAD_CALL_ void
        _FFT1D_R5_cplxf32_1st_R2C(const float* __restrict src, de::CPf* __restrict dst, const uint32_t signal_length);


        _THREAD_CALL_ void
        _FFT1D_R5_cplxf32_1st_C2C(const de::CPf* __restrict src, de::CPf* __restrict dst, const uint32_t signal_length);


        _THREAD_CALL_ void
        _FFT1D_R5_cplxf32_mid_C2C(const de::CPf* __restrict src, de::CPf* __restrict dst, const de::CPf* __restrict _W_table,
            const decx::dsp::fft::FKI1D* _kernel_info);


        // Radix-2
        _THREAD_CALL_ void
        _FFT1D_R2_cplxf32_1st_R2C(const float* __restrict src, de::CPf* __restrict dst, const uint32_t signal_length);


        _THREAD_CALL_ void
        _FFT1D_R2_cplxf32_1st_C2C(const de::CPf* __restrict src, de::CPf* __restrict dst, const uint32_t signal_length);


        _THREAD_CALL_ void
        _FFT1D_R2_cplxf32_mid_C2C(const de::CPf* __restrict src, de::CPf* __restrict dst, const de::CPf* __restrict _W_table,
            const decx::dsp::fft::FKI1D* _kernel_info);

    }

    namespace CPUK{
        _THREAD_CALL_ void
        _FFT1D_caller_cplxf32_1st_R2C(const float* __restrict src, de::CPf* __restrict dst, const decx::dsp::fft::_FFT1D_kernel_info* _kernel_info);


        _THREAD_CALL_ void
        _FFT1D_caller_cplxf32_1st_C2C(const de::CPf* __restrict src, de::CPf* __restrict dst, const decx::dsp::fft::_FFT1D_kernel_info* _kernel_info);


        _THREAD_CALL_ void
        _FFT1D_caller_cplxf32_mid_C2C(const de::CPf* __restrict src, de::CPf* __restrict dst, const de::CPf* __restrict _W_table,
            const decx::dsp::fft::_FFT1D_kernel_info* _kernel_info);



        template <bool _IFFT, typename _type_in> _THREAD_FUNCTION_ void
        _FFT1D_smaller_1st_cplxf32(const _type_in* __restrict src, de::CPf* __restrict dst, const decx::dsp::fft::FKT1D* _tiles,
            const uint64_t _signal_length, const decx::dsp::fft::cpu_FFT1D_smaller<float>* _FFT_info, const uint32_t FFT_call_times,
            const uint32_t FFT_call_time_start, const decx::dsp::fft::FIMT1D* _Twd_info);

        /**
        * The function goes a fragment of a warp, but reaches every warp
        */
        template <typename _type_out, bool _conj> _THREAD_FUNCTION_ void
        _FFT1D_smaller_mid_cplxf32_C2C(const de::CPf* __restrict src, _type_out* __restrict dst, void* __restrict _tmp1, void* __restrict _tmp2,
            const decx::dsp::fft::FKI1D* _global_kernel_info, const decx::dsp::fft::cpu_FFT1D_smaller<float>* _FFT_info, 
            const uint32_t FFT_call_time_start_v1, const uint32_t _FFT_times, const decx::dsp::fft::FIMT1D* _Twd_info);
    }
}
}
}



namespace decx
{
namespace dsp {
    namespace fft {
        template <bool _IFFT, typename _type_in>
        void _FFT1D_cplxf32_1st(const _type_in* __restrict src, de::CPf* __restrict dst,
            const decx::dsp::fft::cpu_FFT1D_planner<float>* _FFT_frame, decx::utils::_thr_1D* t1D, 
            const decx::dsp::fft::FIMT1D* _Twd_info);


        template <bool _IFFT, typename _type_in>
        void _FFT1D_cplxd64_1st(const _type_in* __restrict src, de::CPd* __restrict dst,
            const decx::dsp::fft::cpu_FFT1D_planner<double>* _FFT_frame, decx::utils::_thr_1D* t1D,
            const decx::dsp::fft::FIMT1D* _Twd_info);


        template <typename _type_out, bool _conj>
        void _FFT1D_cplxf32_mid(const de::CPf* __restrict src, _type_out* __restrict dst,
            const decx::dsp::fft::cpu_FFT1D_planner<float>* _FFT_frame, decx::utils::_thr_1D* t1D, const uint32_t _call_order,
            const decx::dsp::fft::FIMT1D* _Twd_info);


        template <typename _type_out, bool _conj>
        void _FFT1D_cplxd64_mid(const de::CPd* __restrict src, _type_out* __restrict dst,
            const decx::dsp::fft::cpu_FFT1D_planner<double>* _FFT_frame, decx::utils::_thr_1D* t1D, const uint32_t _call_order,
            const decx::dsp::fft::FIMT1D* _Twd_info);
    }
}
}


// --------------------------------------------------------- double ---------------------------------------------------------



namespace decx
{
namespace dsp {
namespace fft {
    namespace CPUK {
        // Radix-4
        _THREAD_CALL_ void
        _FFT1D_R4_cplxd64_1st_R2C(const double* __restrict src, de::CPd* __restrict dst, const uint32_t signal_length);


        _THREAD_CALL_ void
        _FFT1D_R4_cplxd64_1st_C2C(const de::CPd* __restrict src, de::CPd* __restrict dst, const uint32_t signal_length);


        _THREAD_CALL_ void
        _FFT1D_R4_cplxd64_mid_C2C(const de::CPd* __restrict src, de::CPd* __restrict dst, const de::CPd* __restrict _W_table,
            const decx::dsp::fft::FKI1D* _kernel_info);


        // Radix-3
        _THREAD_CALL_ void
        _FFT1D_R3_cplxd64_1st_R2C(const double* __restrict src, de::CPd* __restrict dst, const uint32_t signal_length);


        _THREAD_CALL_ void
        _FFT1D_R3_cplxd64_1st_C2C(const de::CPd* __restrict src, de::CPd* __restrict dst, const uint32_t signal_length);


        _THREAD_CALL_ void
        _FFT1D_R3_cplxd64_mid_C2C(const de::CPd* __restrict src, de::CPd* __restrict dst, const de::CPd* __restrict _W_table,
            const decx::dsp::fft::FKI1D* _kernel_info);


        // Radix-5
        _THREAD_CALL_ void
        _FFT1D_R5_cplxd64_1st_R2C(const double* __restrict src, de::CPd* __restrict dst, const uint32_t signal_length);


        _THREAD_CALL_ void
        _FFT1D_R5_cplxd64_1st_C2C(const de::CPd* __restrict src, de::CPd* __restrict dst, const uint32_t signal_length);


        _THREAD_CALL_ void
        _FFT1D_R5_cplxd64_mid_C2C(const de::CPd* __restrict src, de::CPd* __restrict dst, const de::CPd* __restrict _W_table,
            const decx::dsp::fft::FKI1D* _kernel_info);


        // Radix-2
        _THREAD_CALL_ void
        _FFT1D_R2_cplxd64_1st_R2C(const double* __restrict src, de::CPd* __restrict dst, const uint32_t signal_length);


        _THREAD_CALL_ void
        _FFT1D_R2_cplxd64_1st_C2C(const de::CPd* __restrict src, de::CPd* __restrict dst, const uint32_t signal_length);


        _THREAD_CALL_ void
        _FFT1D_R2_cplxd64_mid_C2C(const de::CPd* __restrict src, de::CPd* __restrict dst, const de::CPd* __restrict _W_table,
            const decx::dsp::fft::FKI1D* _kernel_info);

    }

    namespace CPUK{
        _THREAD_CALL_ void
        _FFT1D_caller_cplxd64_1st_R2C(const double* __restrict src, de::CPd* __restrict dst, const decx::dsp::fft::_FFT1D_kernel_info* _kernel_info);


        _THREAD_CALL_ void
        _FFT1D_caller_cplxd64_1st_C2C(const de::CPd* __restrict src, de::CPd* __restrict dst, const decx::dsp::fft::_FFT1D_kernel_info* _kernel_info);


        _THREAD_CALL_ void
        _FFT1D_caller_cplxd64_mid_C2C(const de::CPd* __restrict src, de::CPd* __restrict dst, const de::CPd* __restrict _W_table,
            const decx::dsp::fft::_FFT1D_kernel_info* _kernel_info);



        template <bool _IFFT, typename _type_in> _THREAD_FUNCTION_ void
        _FFT1D_smaller_1st_cplxd64(const _type_in* __restrict src, de::CPd* __restrict dst, const decx::dsp::fft::FKT1D* _tiles,
            const uint64_t _signal_length, const decx::dsp::fft::cpu_FFT1D_smaller<double>* _FFT_info, const uint32_t FFT_call_times,
            const uint32_t FFT_call_time_start, const decx::dsp::fft::FIMT1D* _Twd_info);

        /**
        * The function goes a fragment of a warp, but reaches every warp
        */
        template <typename _type_out, bool _conj> _THREAD_FUNCTION_ void
        _FFT1D_smaller_mid_cplxd64_C2C(const de::CPd* __restrict src, _type_out* __restrict dst, void* __restrict _tmp1, void* __restrict _tmp2,
            const decx::dsp::fft::FKI1D* _global_kernel_info, const decx::dsp::fft::cpu_FFT1D_smaller<double>* _FFT_info, 
            const uint32_t FFT_call_time_start_v1, const uint32_t _FFT_times, const decx::dsp::fft::FIMT1D* _Twd_info);
    }
}
}
}


#endif