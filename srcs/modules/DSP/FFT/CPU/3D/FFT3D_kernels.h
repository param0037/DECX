/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _FFT3D_KERNELS_H_
#define _FFT3D_KERNELS_H_


#include "../CPU_FFT_tiles.h"
#include "CPU_FFT3D_planner.h"


namespace decx
{
namespace dsp {
namespace fft {
    namespace CPUK
    {
        template <typename _type_in, bool _conj> _THREAD_FUNCTION_ void
            _FFT3D_smaller_4rows_cplxf(const _type_in* __restrict src, de::CPf* __restrict dst, const decx::dsp::fft::FKT1D_fp32* _tiles,
                const uint32_t _pitch_src, const uint32_t _pitch_dst, const uint32_t _proc_H_r1,const uint32_t start_dex,
                const decx::dsp::fft::cpu_FFT3D_subproc<float>* _FFT_info);


        template <typename _type_out> _THREAD_FUNCTION_ void
            _IFFT3D_smaller_4rows_cplxf(const de::CPf* __restrict src, _type_out* __restrict dst, const decx::dsp::fft::FKT1D_fp32* _tiles,
                const uint32_t _pitch_src, const uint32_t _pitch_dst, const uint32_t _proc_H_r1, const uint32_t start_dex,
                const decx::dsp::fft::cpu_FFT3D_subproc<float>* _FFT_info);
    }

    template <typename _type_in, bool _conj>
    void _FFT3D_H_entire_rows_cplxf(const _type_in* __restrict src, de::CPf* __restrict dst, const decx::dsp::fft::cpu_FFT3D_planner<float>* planner,
        decx::utils::_thread_arrange_1D* t1D, decx::dsp::fft::FFT_directions _proc_dir);


    template <typename _type_out>
    void _IFFT3D_H_entire_rows_cplxf(const de::CPf* __restrict src, _type_out* __restrict dst, const decx::dsp::fft::cpu_FFT3D_planner<float>* planner,
        decx::utils::_thread_arrange_1D* t1D, decx::dsp::fft::FFT_directions _proc_dir);
}
}
}

#endif
