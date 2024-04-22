/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _FFT2D_CONFIG_CUH_
#define _FFT2D_CONFIG_CUH_

#include "../../FFT_commons.h"
#include "../../../../classes/GPU_Matrix.h"
#include "../../../../core/cudaStream_management/cudaEvent_queue.h"
#include "../../../../core/cudaStream_management/cudaStream_queue.h"
#include "../../../../core/resources_manager/decx_resource.h"



namespace decx
{
namespace dsp
{
namespace fft 
{
    template <typename _type_in>
    class _cuda_FFT2D_planner;


    class _FFT2D_1way_config;
}
}
}


#define _FFT2D_BLOCK_X_ _WARP_SIZE_
#define _FFT2D_BLOCK_Y_ 8



class decx::dsp::fft::_FFT2D_1way_config
{
private:
    uint32_t _signal_length;

    std::vector<uint32_t> _radix;
    decx::utils::Fixed_Length_Array<decx::dsp::fft::FKI_4_2DK> _kernrel_infos;

public:
    uint32_t _pitchsrc, _pitchdst, _pitchtmp;


    _FFT2D_1way_config() {}


    void plan(const uint32_t signal_length);


    uint32_t get_radix(const uint32_t _index) const;


    const decx::dsp::fft::FKI_4_2DK* get_kernel_info(const uint32_t _index) const;


    uint32_t get_signal_len() const;


    uint32_t partition_num() const;
};




template <typename _data_type>
class decx::dsp::fft::_cuda_FFT2D_planner
{
private:
    uint2 _signal_dims;
    
    decx::PtrInfo<void> _tmp1, _tmp2;

    _FFT2D_1way_config _FFT_H, _FFT_V;

    uint2 _buffer_dims;


public:

    _cuda_FFT2D_planner() {}


    bool changed(const uint2 signal_dims, const uint32_t pitchsrc, const uint32_t pitchdst) const;


    void _CRSR_ plan(const uint2 signal_dims, const uint32_t pitchsrc, const uint32_t pitchdst, de::DH* handle);

    
    const decx::dsp::fft::_FFT2D_1way_config* get_FFT_info(const decx::dsp::fft::FFT_directions _dir) const;


    template <typename _ptr_type>
    _ptr_type* get_tmp1_ptr() const {
        return static_cast<_ptr_type*>(this->_tmp1.ptr);
    }


    template <typename _ptr_type>
    _ptr_type* get_tmp2_ptr() const {
        return static_cast<_ptr_type*>(this->_tmp2.ptr);
    }


    uint2 get_buffer_dims() const;


    template <typename _type_in>
    void Forward(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, decx::cuda_stream* S) const;

    template <typename _type_out>
    void Inverse(decx::_GPU_Matrix* src, decx::_GPU_Matrix* dst, decx::cuda_stream* S) const;


    static void release_buffers(decx::dsp::fft::_cuda_FFT2D_planner<_data_type>* _fake_this);


    ~_cuda_FFT2D_planner();
};


namespace decx
{
    namespace dsp {
        namespace fft {
            extern decx::ResourceHandle cuda_FFT2D_cplxf32_planner;
            extern decx::ResourceHandle cuda_IFFT2D_cplxf32_planner;

            extern decx::ResourceHandle cuda_FFT2D_cplxd64_planner;
            extern decx::ResourceHandle cuda_IFFT2D_cplxd64_planner;
        }
    }
}


#endif