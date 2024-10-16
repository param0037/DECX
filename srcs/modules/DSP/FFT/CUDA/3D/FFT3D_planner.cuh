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


#ifndef _FFT3D_PLANNER_CUH_
#define _FFT3D_PLANNER_CUH_


#include "../../../../../common/basic.h"
#include "../2D/FFT2D_config.cuh"
#include "../../../../core/cudaStream_management/cudaEvent_queue.h"
#include "../../../../core/cudaStream_management/cudaStream_queue.h"
#include "../../../../../common/Classes/Tensor.h"
#include "../../../../../common/Classes/GPU_Tensor.h"
#include "../../../../core/resources_manager/decx_resource.h"


namespace decx
{
namespace dsp {
    namespace fft {
        template <typename _type_in>
        class _cuda_FFT3D_planner;


        typedef struct
        {
            decx::dsp::fft::_FFT2D_1way_config _1way_FFT_conf;
            uint32_t _parallel;
            uint32_t _signal_pitch_src;
            uint32_t _signal_pitch_dst;
        }_cuda_FFT3D_mid_config;

    }
}
}


/**
* If using midproc FFT, the thread distribution and data layout is given below:
* -----------------------------------------------   -           -
*                                                   |           |
*                                                   |       signal_len
*                 lane 0                        signal_pitch    |
*                                                   |           |
* 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0   |           -
* -----------------------------------------------   -           -
*                                                   |           |
*                                                   |       signal_len
*                 lane 1                        signal_pitch    |
*                                                   |           |
* 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0   |           -
* -----------------------------------------------   -
* ...
* -----------------------------------------------
* Signal_pitch should be consistent with the output tensor. 
*/

/*
* On my device, the efficiency improvement is not significant to complete the 
* extra time of cudaMemcpy2DAsync(D2D). Thus it's turned off.
*/
#define _CUDA_FFT3D_restrict_coalesce_ 0

template <typename _data_type>
class decx::dsp::fft::_cuda_FFT3D_planner
{
private:
    uint3 _signal_dims;     // [depth, width, height]
    /**
    * .x : Pitch when performing FFT along depth dimension
    * .y : Pitch when performing FFT along width dimension
    * .z : Pitch when performing FFT along height dimension
    */
    //uint3 _pitch_DWH;

    decx::PtrInfo<void> _tmp1, _tmp2;

    decx::dsp::fft::_FFT2D_1way_config _FFT_D,  // Along depth
                                       _FFT_H;  // Along height
#if _CUDA_FFT3D_restrict_coalesce_
    bool _sync_dpitchdst_needed;
#endif

    uint32_t _input_typesize, _output_typesize;

    decx::dsp::fft::_cuda_FFT3D_mid_config _FFT_W;  // Alongg width


    const decx::dsp::fft::_FFT2D_1way_config* get_FFT_info(const decx::dsp::fft::FFT_directions _dir) const;


    const decx::dsp::fft::_cuda_FFT3D_mid_config* get_midFFT_info() const;


    template <typename _ptr_type>
    _ptr_type* get_tmp1_ptr() const {
        return static_cast<_ptr_type*>(this->_tmp1.ptr);
    }


    template <typename _ptr_type>
    _ptr_type* get_tmp2_ptr() const {
        return static_cast<_ptr_type*>(this->_tmp2.ptr);
    }

public:


    _cuda_FFT3D_planner() {}


    void _CRSR_ plan(const decx::_tensor_layout* _src_layout, const decx::_tensor_layout* _dst_layout,
        de::DH* handle, decx::cuda_stream* S);


    template <typename _type_in>
    void Forward(decx::_GPU_Tensor* src, decx::_GPU_Tensor* dst, decx::cuda_stream* S) const;


    template <typename _type_out>
    void Inverse(decx::_GPU_Tensor* src, decx::_GPU_Tensor* dst, decx::cuda_stream* S) const;


    bool changed(const decx::_tensor_layout* src_layout, const decx::_tensor_layout* dst_layout) const;


    static void release(decx::dsp::fft::_cuda_FFT3D_planner<_data_type>* _fake_this);


    ~_cuda_FFT3D_planner();
};


namespace decx
{
    namespace dsp {
        namespace fft {
            extern decx::ResourceHandle cuda_FFT3D_cplxf32_planner;
            extern decx::ResourceHandle cuda_IFFT3D_cplxf32_planner;

            extern decx::ResourceHandle cuda_FFT3D_cplxd64_planner;
            extern decx::ResourceHandle cuda_IFFT3D_cplxd64_planner;
        }
    }
}



#endif
