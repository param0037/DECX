/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CPU_FFT_CONFIGS_H_
#define _CPU_FFT_CONFIGS_H_

#include "../../../core/basic.h"
#include "../../../core/memory_management/MemBlock.h"
#include "../fft_utils.h"


namespace decx
{
    namespace signal
    {
        struct CPU_FFT_Configs;


        bool apart_for_CPU_FFT(int __x, std::vector<int>* res_arr);
    }
}




struct decx::signal::CPU_FFT_Configs
{
    std::vector<int> _base;

    bool _is_2s;

    int _base_num;

    CPU_FFT_Configs() {}

    /*
    * @param vec_len : The effective length of a signal (padding NOT included), i.e.
    * decx::_Vector<type>::length
    */
    bool FFT1D_config_gen(const size_t vec_len, de::DH* handle);

    /*
    * Call only when it is used to configure a 2D (I)FFT process
    */
    void FFT2D_launch_param_gen(const int vec_len, const int _height);


    ~CPU_FFT_Configs();
};





namespace decx
{
    namespace signal
    {
        template <typename T_init, typename T_last>
        struct _CPU_FFT2D_Kernel_Param;


        typedef decx::signal::_CPU_FFT2D_Kernel_Param<float, double> _R2C_PM;
        typedef decx::signal::_CPU_FFT2D_Kernel_Param<double, double> _C2C_PM;
    }
}


template <typename T_init, typename T_last>
struct decx::signal::_CPU_FFT2D_Kernel_Param
{
    T_init* _initial;
    decx::alloc::MIF<double> tmp0, tmp1;
    decx::signal::CPU_FFT_Configs* _conf;
    uint signal_W;
    uint Wsrc;
    uint2 proc_dim;

    _CPU_FFT2D_Kernel_Param();


    _CPU_FFT2D_Kernel_Param(T_init* _init_ptr,
                            decx::alloc::MIF<double> _tmp0,
                            decx::alloc::MIF<double> _tmp1,
                            decx::signal::CPU_FFT_Configs* __conf,
                            const uint _signal_W,
                            const uint _Wsrc,
                            const uint2 _proc_dim);
};


#define _NEW_FFT_PM_R2C_(_init_ptr, tmp0, tmp1, __conf, _signal_W, _Wsrc, _proc_dim)    \
    decx::signal::_R2C_PM(_init_ptr, tmp0, tmp1, __conf, _signal_W, _Wsrc, _proc_dim)  \


#define _NEW_FFT_PM_C2C_(_init_ptr, tmp0, tmp1, __conf, _signal_W, _proc_dim)    \
    decx::signal::_C2C_PM(_init_ptr, tmp0, tmp1, __conf, _signal_W, 0, (_proc_dim))  \


#define _NEW_FFT_PM_C2C_mid(tmp0, tmp1, __conf, _signal_W, _proc_dim)    \
    decx::signal::_C2C_PM(NULL, tmp0, tmp1, __conf, _signal_W, 0, _proc_dim)  \


#endif