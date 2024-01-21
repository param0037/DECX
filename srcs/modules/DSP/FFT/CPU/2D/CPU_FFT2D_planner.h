/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CPU_FFT2D_PLANNER_H_
#define _CPU_FFT2D_PLANNER_H_


#include "../1D/CPU_FFT1D_planner.h"
#include "../../../../classes/Matrix.h"


namespace decx
{
namespace dsp {
    namespace fft {
        template <typename _type_in>
        class cpu_FFT2D_planner;
    }
}
}


template <typename _type_in>
class decx::dsp::fft::cpu_FFT2D_planner
{
private:
    uint2 _signal_dims;

    // Horizontal and vertical FFTs
    decx::dsp::fft::cpu_FFT1D_smaller<_type_in> _FFT_H, _FFT_V;

    // Tiles for each thread (2 (double buffers) for each)
    decx::utils::Fixed_Length_Array<decx::dsp::fft::FKT1D_fp32> _tiles;

    // Thread distributions for width and height
    decx::utils::frag_manager _thread_dist_FFTH, _thread_dist_FFTV;

    decx::PtrInfo<void> _tmp1, _tmp2;

public:
    cpu_FFT2D_planner() {}


    _CRSR_ cpu_FFT2D_planner(const uint2 signal_dims, de::DH* handle);


    _CRSR_ void plan(decx::utils::_thread_arrange_1D* t1D, de::DH* handle);


    uint2 get_signal_dims() const;


    void* get_tmp1_ptr() const;
    void* get_tmp2_ptr() const;


    const decx::dsp::fft::cpu_FFT1D_smaller<_type_in>* get_FFTH_info() const;
    const decx::dsp::fft::cpu_FFT1D_smaller<_type_in>* get_FFTV_info() const;


    const decx::utils::frag_manager* get_thread_dist_H() const;
    const decx::utils::frag_manager* get_thread_dist_V() const;


    void release_buffers();


    const decx::dsp::fft::FKT1D_fp32* get_tile_ptr(const uint32_t _id) const;


    template <typename _type_out>
    static inline uint8_t get_alignment_FFT_last_dimension()
    {
        return std::is_same_v<_type_out, uint8_t> ? 8 : 4;
    }


    ~cpu_FFT2D_planner();
};



#endif