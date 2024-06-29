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


#ifndef _CUDA_FFT2D_PLANNER_CUH_
#define _CUDA_FFT2D_PLANNER_CUH_


#include "../../FFT_commons.h"
#include "../2D/FFT2D_config.cuh"
#include "../../../../classes/Vector.h"
#include "../../../../classes/GPU_Vector.h"
#include "../../../../core/resources_manager/decx_resource.h"


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

    
    static void release(decx::dsp::fft::_cuda_FFT1D_planner<_data_type>* _fake_this);


    ~_cuda_FFT1D_planner();
};


namespace decx
{
namespace dsp {
    namespace fft {
        extern decx::ResourceHandle cuda_FFT1D_cplxf32_planner;
        extern decx::ResourceHandle cuda_IFFT1D_cplxf32_planner;

        extern decx::ResourceHandle cuda_FFT1D_cplxf64_planner;
        extern decx::ResourceHandle cuda_IFFT1D_cplxf64_planner;
    }
}
}


#endif