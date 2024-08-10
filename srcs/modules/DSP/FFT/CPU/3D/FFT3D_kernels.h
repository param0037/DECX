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


#ifndef _FFT3D_KERNELS_H_
#define _FFT3D_KERNELS_H_


#include "../FFT_common/CPU_FFT_tiles.h"
#include "CPU_FFT3D_planner.h"


namespace decx
{
namespace dsp {
namespace fft {
    namespace CPUK
    {
        template <typename _type_in, bool _conj> _THREAD_FUNCTION_ void
            _FFT3D_smaller_4rows_cplxf(const _type_in* __restrict src, de::CPf* __restrict dst, const decx::dsp::fft::FKT1D* _tiles,
                const uint32_t _pitch_src, const uint32_t _pitch_dst, const uint32_t _proc_H_r1,const uint32_t start_dex,
                const decx::dsp::fft::cpu_FFT3D_subproc<float>* _FFT_info);


        template <typename _type_in, bool _conj> _THREAD_FUNCTION_ void
            _FFT3D_smaller_2rows_cplxd(const _type_in* __restrict src, de::CPd* __restrict dst, const decx::dsp::fft::FKT1D* _tiles,
                const uint32_t _pitch_src, const uint32_t _pitch_dst, const uint32_t _proc_H_r1, const uint32_t start_dex,
                const decx::dsp::fft::cpu_FFT3D_subproc<double>* _FFT_info);


        template <typename _type_out> _THREAD_FUNCTION_ void
            _IFFT3D_smaller_4rows_cplxf(const de::CPf* __restrict src, _type_out* __restrict dst, const decx::dsp::fft::FKT1D* _tiles,
                const uint32_t _pitch_src, const uint32_t _pitch_dst, const uint32_t _proc_H_r1, const uint32_t start_dex,
                const decx::dsp::fft::cpu_FFT3D_subproc<float>* _FFT_info);


        template <typename _type_out> _THREAD_FUNCTION_ void
            _IFFT3D_smaller_2rows_cplxd(const de::CPd* __restrict src, _type_out* __restrict dst, const decx::dsp::fft::FKT1D* _tiles,
                const uint32_t _pitch_src, const uint32_t _pitch_dst, const uint32_t _proc_H_r1, const uint32_t start_dex,
                const decx::dsp::fft::cpu_FFT3D_subproc<double>* _FFT_info);
    }

    template <typename _type_in, bool _conj>
    void _FFT3D_H_entire_rows_cplxf(const _type_in* __restrict src, de::CPf* __restrict dst, const decx::dsp::fft::cpu_FFT3D_planner<float>* planner,
        decx::utils::_thread_arrange_1D* t1D, decx::dsp::fft::FFT_directions _proc_dir);


    template <typename _type_in, bool _conj>
    void _FFT3D_H_entire_rows_cplxd(const _type_in* __restrict src, de::CPd* __restrict dst, const decx::dsp::fft::cpu_FFT3D_planner<double>* planner,
        decx::utils::_thread_arrange_1D* t1D, decx::dsp::fft::FFT_directions _proc_dir);


    template <typename _type_out>
    void _IFFT3D_H_entire_rows_cplxf(const de::CPf* __restrict src, _type_out* __restrict dst, const decx::dsp::fft::cpu_FFT3D_planner<float>* planner,
        decx::utils::_thread_arrange_1D* t1D, decx::dsp::fft::FFT_directions _proc_dir);


    template <typename _type_out>
    void _IFFT3D_H_entire_rows_cplxd(const de::CPd* __restrict src, _type_out* __restrict dst, const decx::dsp::fft::cpu_FFT3D_planner<double>* planner,
        decx::utils::_thread_arrange_1D* t1D, decx::dsp::fft::FFT_directions _proc_dir);
}
}
}

#endif
