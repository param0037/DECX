/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _FFT2D_KERNELS_H_
#define _FFT2D_KERNELS_H_


#include "../CPU_FFT_tiles.h"
#include "CPU_FFT2D_planner.h"


namespace decx
{
namespace dsp {
namespace fft {
    namespace CPUK 
    {
        template <typename _type_in, bool _conj> _THREAD_FUNCTION_ void 
        _FFT2D_smaller_4rows_cplxf(const _type_in* __restrict src, de::CPf* __restrict dst, const decx::dsp::fft::FKT1D* _tiles, 
            const uint32_t _pitch_src, const uint32_t _pitch_dst, const uint32_t _proc_H_r1, 
            const decx::dsp::fft::cpu_FFT1D_smaller<float>* _FFT_info);


        template <typename _type_out> _THREAD_FUNCTION_ void 
        _IFFT2D_smaller_4rows_cplxf(const de::CPf* __restrict src, _type_out* __restrict dst, const decx::dsp::fft::FKT1D* _tiles, 
            const uint32_t _pitch_src, const uint32_t _pitch_dst, const uint32_t _proc_H_r1, 
            const decx::dsp::fft::cpu_FFT1D_smaller<float>* _FFT_info);

        template <typename _type_out> _THREAD_FUNCTION_ void
        _IFFT2D_smaller_2rows_cplxd(const de::CPd* __restrict src, _type_out* __restrict dst, const decx::dsp::fft::FKT1D* _tiles,
            const uint32_t _pitch_src, const uint32_t _pitch_dst, const uint32_t _proc_H_r1,
            const decx::dsp::fft::cpu_FFT1D_smaller<double>* _FFT_info);


        template <typename _type_in, bool _conj> _THREAD_FUNCTION_ void 
        _FFT2D_smaller_2rows_cplxd(const _type_in* __restrict src, de::CPd* __restrict dst, const decx::dsp::fft::FKT1D* _tiles, 
            const uint32_t _pitch_src, const uint32_t _pitch_dst, const uint32_t _proc_H_r1, 
            const decx::dsp::fft::cpu_FFT1D_smaller<double>* _FFT_info);
    }

        template <typename _type_in, bool _conj>
        void _FFT2D_H_entire_rows_cplxf(const _type_in* __restrict src, de::CPf* __restrict dst, const decx::dsp::fft::cpu_FFT2D_planner<float>* planner,
            const uint32_t pitch_src, const uint32_t pitch_dst, decx::utils::_thread_arrange_1D* t1D, bool _is_FFTH);


        template <typename _type_in, bool _conj>
        void _FFT2D_H_entire_rows_cplxd(const _type_in* __restrict src, de::CPd* __restrict dst, const decx::dsp::fft::cpu_FFT2D_planner<double>* planner,
            const uint32_t pitch_src, const uint32_t pitch_dst, decx::utils::_thread_arrange_1D* t1D, bool _is_FFTH);


        template <typename _type_out>
        void _IFFT2D_H_entire_rows_cplxf(const de::CPf* __restrict src, _type_out* __restrict dst, const decx::dsp::fft::cpu_FFT2D_planner<float>* planner,
            const uint32_t pitch_src, const uint32_t pitch_dst, decx::utils::_thread_arrange_1D* t1D, bool _is_FFTH);


        template <typename _type_out>
        void _IFFT2D_H_entire_rows_cplxd(const de::CPd* __restrict src, _type_out* __restrict dst, const decx::dsp::fft::cpu_FFT2D_planner<double>* planner,
            const uint32_t pitch_src, const uint32_t pitch_dst, decx::utils::_thread_arrange_1D* t1D, bool _is_FFTH);
    }
}
}


#endif