/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CPU_FFT3D_PLANNER_H_
#define _CPU_FFT3D_PLANNER_H_


#include "../W_table.h"
#include "../CPU_FFT_defs.h"
#include "../CPU_FFT_tiles.h"
#include "../2D/CPU_FFT2D_planner.h"
#include "../../../../classes/Tensor.h"
#include "FFT3D_kernel_utils.h"


namespace decx
{
namespace dsp {
    namespace fft {
        template <typename _type_in>
        class cpu_FFT3D_planner;


        template <typename _type_in>
        struct cpu_FFT3D_subproc;
    }
}
}



/**
* ------------------------ STAGE 1, FFT along depth ------------------------
* 
* When performing FFT on depth dimension, zip the width and height.
* The visualized data structure is shown below:
* phyaddr   viraddr
*                   |---> FFT_D ---> ...  ______________________
*   0         0     |* * * * * * * * * *    |       |
*   1         1     |* * * * * * * * * *  width   wpitch      lane 0
*   2         2     |* * * * * * * * * *  __|       |
*   3         X     |0 0 0 0 0 0 0 0 0 0  __________|___________
*   4         3     |* * * * * * * * * *
*   5         4     |* * * * * * * * * *                      lane 1
*   6         5     |* * * * * * * * * *
*   7         X     |0 0 0 0 0 0 0 0 0 0  ______________________
*                   ...
*                   ...
*   123       X     |0 0 0 0 0 0 0 0 0 0  ______________________
*   124       93    |* * * * * * * * * *
*   125       94    |* * * * * * * * * *                      lane height - 1
*   126       95    |* * * * * * * * * *
*   127       X     |0 0 0 0 0 0 0 0 0 0  ______________________
* 
* Ignore the zero rows when performing FFT calculation
*/

template <typename _data_type>
class decx::dsp::fft::cpu_FFT3D_planner
{
private:
    decx::dsp::fft::cpu_FFT3D_subproc<_data_type> _FFT_D, _FFT_W, _FFT_H;
    uint3 _signal_dims;     // [D, W, H]

    uint3 _aligned_proc_dims;

    const decx::_tensor_layout* _src_layout;
    const decx::_tensor_layout*_dst_layout;

    decx::bp::_cpu_transpose_MK_config<8> _transp_config_MC, _transp_config_MC_back;
    decx::bp::_cpu_transpose_config<sizeof(_data_type) * 2>  _transp_config, _transp_config_back;

    uint32_t _concurrency;

    // Tiles for each thread (2 (double buffers) for each)
    decx::utils::Fixed_Length_Array<decx::dsp::fft::FKT1D_fp32> _tiles;

    decx::PtrInfo<void> _tmp1, _tmp2;


    uint32_t _input_typesize, _output_typesize;


    void _CRSR_ allocate_buffers(de::DH* handle);


    void* get_tmp1_ptr() const;
    void* get_tmp2_ptr() const;


    template <typename type_out>
    void plan_transpose_configs();

public:
    cpu_FFT3D_planner() {}


    //_CRSR_ cpu_FFT3D_planner(const uint3 signal_dims, de::DH* handle);


    _CRSR_ template <typename _type_out>
    void plan(decx::utils::_thread_arrange_1D* t1D, 
        const decx::_tensor_layout* src_layout, const decx::_tensor_layout* dst_layout, de::DH* handle);


    const decx::dsp::fft::cpu_FFT3D_subproc<_data_type>* get_subproc(const decx::dsp::fft::FFT_directions proc_dir) const;


    const decx::dsp::fft::FKT1D_fp32* get_tile_ptr(const uint32_t _id) const;


    template <typename _input_type>
    void _CRSR_ Forward(decx::_Tensor* src, decx::_Tensor* dst) const;

    template <typename _output_type>
    void _CRSR_ Inverse(decx::_Tensor* src, decx::_Tensor* dst) const;


    bool changed(const decx::_tensor_layout* src_layout, const decx::_tensor_layout* dst_layout,
        const uint32_t concurrency) const;


    void release();


    ~cpu_FFT3D_planner();
};



template <typename _data_type>
struct decx::dsp::fft::cpu_FFT3D_subproc
{
    decx::dsp::fft::cpu_FFT1D_smaller<_data_type> _FFT_info;
    decx::utils::frag_manager _f_mgr;
    decx::utils::unpitched_frac_mapping<uint32_t> _FFT_zip_info_LDG, _FFT_zip_info_STG;
    uint32_t _pitchsrc, _pitchdst;
};


namespace decx
{
    namespace dsp {
        namespace fft {
            extern decx::dsp::fft::cpu_FFT3D_planner<float>* FFT3D_cplxf32_planner;
            extern decx::dsp::fft::cpu_FFT3D_planner<float>* IFFT3D_cplxf32_planner;
        }
    }
}


#endif