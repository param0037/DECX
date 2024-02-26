/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CUDA_FFT2D_PLANNER_CUH_
#define _CUDA_FFT2D_PLANNER_CUH_


#include "../../FFT_commons.h"
#include "../2D/FFT2D_config.cuh"
#include "../../../../classes/Vector.h"
#include "../../../../classes/GPU_Vector.h"


namespace decx
{
namespace dsp {
    namespace fft {
        template <typename _type_in>
        class _cuda_FFT1D_planner;
    }
}
}


template <typename _data_type>
class decx::dsp::fft::_cuda_FFT1D_planner
{
private:
    uint64_t _signal_length;

    decx::PtrInfo<void> _tmp1, _tmp2;

    std::vector<uint32_t> _all_radixes;
    uint32_t _large_FFT_lengths[2];

    decx::dsp::fft::_cuda_FFT2D_planner<_data_type> _FFT2D_layout;


    void _CRSR_ _plan_group_radixes(de::DH* handle, decx::cuda_stream* S);


    //uint64_t _calc_max_required_tmp_size();


public:
    _cuda_FFT1D_planner() {}


    bool changed(const uint64_t signal_len) const;


    void _CRSR_ plan(const uint64_t signal_length, de::DH* handle, decx::cuda_stream* S);

    template <typename _type_in>
    void Forward(decx::_Vector* src, decx::_Vector* dst, decx::cuda_stream* S) const;

    template <typename _type_in>
    void Forward(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, decx::cuda_stream* S) const;


    template <typename _type_out>
    void Inverse(decx::_Vector* src, decx::_Vector* dst, decx::cuda_stream* S) const;

    template <typename _type_out>
    void Inverse(decx::_GPU_Vector* src, decx::_GPU_Vector* dst, decx::cuda_stream* S) const;


    uint64_t get_signal_length() const;


    const decx::dsp::fft::_cuda_FFT2D_planner<_data_type>* get_FFT2D_planner() const;


    uint32_t get_larger_FFT_lengths(const uint8_t _id) const;
};


namespace decx
{
    namespace dsp {
        namespace fft {
            extern decx::dsp::fft::_cuda_FFT1D_planner<float>* cuda_FFT1D_cplxf32_planner;
            extern decx::dsp::fft::_cuda_FFT1D_planner<float>* cuda_IFFT1D_cplxf32_planner;
        }
    }
}


#endif