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


#ifndef _CPU_FFT2D_PLANNER_H_
#define _CPU_FFT2D_PLANNER_H_


#include "../1D/CPU_FFT1D_planner.h"
#include "../../../../../common/Classes/Matrix.h"
#include "../../../../core/resources_manager/decx_resource.h"


namespace decx
{
namespace dsp {
    namespace fft {
        template <typename _data_type>
        class cpu_FFT2D_planner;
    }
}
}


template <typename _data_type>
class decx::dsp::fft::cpu_FFT2D_planner
{
private:
    uint2 _signal_dims;

    // Horizontal and vertical FFTs
    decx::dsp::fft::cpu_FFT1D_smaller<_data_type> _FFT_H, _FFT_V;

    // Tiles for each thread (2 (double buffers) for each)
    decx::utils::Fixed_Length_Array<decx::dsp::fft::FKT1D> _tiles;

    // Thread distributions for width and height
    decx::utils::frag_manager _thread_dist_FFTH, _thread_dist_FFTV;

    decx::PtrInfo<void> _tmp1, _tmp2;

    uint32_t _input_typesize, _output_typesize;

    decx::blas::_cpu_transpose_config _transpose_config_1st, _transpose_config_2nd;

    uint32_t _concurrency;


    template <typename _type_out>
    void _CRSR_ plan_transpose_configs(de::DH* handle);

public:
    cpu_FFT2D_planner() {}


    bool changed(const decx::_matrix_layout* src_layout, const decx::_matrix_layout* dst_layout,
        const uint32_t concurrency) const;

    
    template <typename _type_out> _CRSR_ 
    void plan(const decx::_matrix_layout* src_layout, const decx::_matrix_layout* dst_layout, 
        decx::utils::_thread_arrange_1D* t1D, de::DH* handle);


    uint2 get_signal_dims() const;


    void* get_tmp1_ptr() const;
    void* get_tmp2_ptr() const;


    const decx::dsp::fft::cpu_FFT1D_smaller<_data_type>* get_FFTH_info() const;
    const decx::dsp::fft::cpu_FFT1D_smaller<_data_type>* get_FFTV_info() const;


    const decx::utils::frag_manager* get_thread_dist_H() const;
    const decx::utils::frag_manager* get_thread_dist_V() const;


    static void release_buffers(decx::dsp::fft::cpu_FFT2D_planner<_data_type>* _fake_this);


    const decx::dsp::fft::FKT1D* get_tile_ptr(const uint32_t _id) const;


    template <typename _type_out>
    static inline uint8_t get_alignment_FFT_last_dimension()
    {
        if (sizeof(_data_type) == 4) {
            return std::is_same_v<_type_out, uint8_t> ? 8 : 4;
        }
        else {
            return std::is_same_v<_type_out, uint8_t> ? 8 : 2;
        }
    }

    
    template <typename _type_in>
    void Forward(decx::_Matrix* src, decx::_Matrix* dst, decx::utils::_thread_arrange_1D* t1D) const;

    template <typename _type_out>
    void Inverse(decx::_Matrix* src, decx::_Matrix* dst, decx::utils::_thread_arrange_1D* t1D) const;


    ~cpu_FFT2D_planner();
};


namespace decx
{
namespace dsp {
    namespace fft {
        extern decx::ResourceHandle cpu_FFT2D_cplxf32_planner;
        extern decx::ResourceHandle cpu_IFFT2D_cplxf32_planner;

        extern decx::ResourceHandle g_cpu_FFT2D_cplxd64_planner;
        extern decx::ResourceHandle g_cpu_IFFT2D_cplxd64_planner;
    }
}
}


#endif
