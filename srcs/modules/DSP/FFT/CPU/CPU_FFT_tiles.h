/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _CPU_FFT_TILES_H_
#define _CPU_FFT_TILES_H_


// Include complex number
#include "../../../classes/classes_util.h"

// Include comlex number fast math functions
#include "../../CPU_cpf32_avx.h"
#include "../../CPU_cpd64_avx.h"

#include "CPU_FFT_defs.h"
#include "../../../BLAS/basic_process/transpose/CPU/transpose_exec.h"
#include "../FFT_commons.h"



namespace decx
{
namespace dsp {
    namespace fft 
    {
        struct _FFT1D_kernel_tile_fp32;


        using FKT1D_fp32 = decx::dsp::fft::_FFT1D_kernel_tile_fp32;
    }
}
}

/**
* For Stockham methods
* 
* Since the _tile_length is aligned to 8 (8x fp32 = 256 bits) (aligne to 4 in double-precision cae), don't need to concern about 
* _v1 kernels
* 
* The memory layout of tile is shown below (in the case of radix-3, for example):
* 
* |<------------------------------ _tile_length -------------------------->|
* |<---_tile_frag_len--->|
* |<---- (aligned to 8) ---->|<------ _tile_pitch ----->|
* [***********************000|***********************000|***********************000]
*/
struct decx::dsp::fft::_FFT1D_kernel_tile_fp32
{
    decx::PtrInfo<void> _tmp_ptr;

    /*
    * When data_type == float: in de::CPf;
    * When data_type == double: in de::CPd;
    */
    uint32_t _tile_row_pitch;
    uint32_t _tile_len;
    uint32_t _total_size;

    template <typename _ptr_type>
    _ptr_type* get_tile1() const
    {
        return (_ptr_type*)this->_tmp_ptr.ptr;
    }


    template <typename _ptr_type>
    _ptr_type* get_tile2() const
    {
        return (_ptr_type*)((uint8_t*)this->_tmp_ptr.ptr + this->_total_size / 2);
    }


    void flush() const;


    void _inblock_transpose_vecAdj_2_VecDist_cplxf(decx::utils::double_buffer_manager* __restrict _double_buffer) const;


    void _inblock_transpose_vecDist_2_VecAdj_fp32(decx::utils::double_buffer_manager* __restrict _double_buffer) const;


    void _inblock_transpose_vecDist_2_VecAdj_cplxf(decx::utils::double_buffer_manager* __restrict _double_buffer) const;


    void _inblock_transpose_vecAdj_2_VecDist_cplxd(decx::utils::double_buffer_manager* __restrict _double_buffer) const;


    void _inblock_transpose_vecDist_2_VecAdj_fp64(decx::utils::double_buffer_manager* __restrict _double_buffer) const;


    void _inblock_transpose_vecDist_2_VecAdj_cplxd(decx::utils::double_buffer_manager* __restrict _double_buffer) const;


    template <typename _data_type>
    void allocate_tile(const uint32_t tile_frag_len, de::DH* handle);


    void release();
};



#endif