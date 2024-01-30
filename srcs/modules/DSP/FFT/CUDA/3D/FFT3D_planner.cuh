/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _FFT3D_PLANNER_CUH_
#define _FFT3D_PLANNER_CUH_


#include "../../../../core/basic.h"
#include "../2D/FFT2D_config.cuh"
#include "../../../../core/cudaStream_management/cudaEvent_queue.h"
#include "../../../../core/cudaStream_management/cudaStream_queue.h"
#include "../../../../classes/Tensor.h"


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


template <typename _type_in>
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

    decx::dsp::fft::_cuda_FFT3D_mid_config _FFT_W;  // Alongg width

public:
    enum FFT_directions {
        _FFT_AlongH = 0,
        _FFT_AlongW = 1,
        _FFT_AlongD = 2
    };


    _cuda_FFT3D_planner() {}


    _CRSR_ _cuda_FFT3D_planner(const uint3 signal_dims);


    void _CRSR_ plan(const decx::_tensor_layout* _src_layout, const decx::_tensor_layout* _dst_layout,
        de::DH* handle, decx::cuda_stream* S);


    const decx::dsp::fft::_FFT2D_1way_config* get_FFT_info(const FFT_directions _dir) const;


    const decx::dsp::fft::_cuda_FFT3D_mid_config* get_midFFT_info() const;


    template <typename _ptr_type>
    _ptr_type* get_tmp1_ptr() const {
        return static_cast<_ptr_type*>(this->_tmp1.ptr);
    }


    template <typename _ptr_type>
    _ptr_type* get_tmp2_ptr() const {
        return static_cast<_ptr_type*>(this->_tmp2.ptr);
    }


    void release();


    ~_cuda_FFT3D_planner();
};



#endif
