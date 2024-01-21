/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CUDA_FFT_CONFIGS_H_
#define _CUDA_FFT_CONFIGS_H_

#include "../../../core/basic.h"
#include "../../../core/memory_management/MemBlock.h"
#include "../FFT_commons.h"



namespace decx
{
    namespace dsp
    {
        struct CUDA_FFT_Configs;


        //bool apart_for_CUDA_FFT(int __x, std::vector<int>* res_arr);
    }
}



struct decx::dsp::CUDA_FFT_Configs
{
    std::vector<uint32_t> _base;

    bool _is_2s;
    int _base_num;

    CUDA_FFT_Configs();

    /*
    * @param vec_len : The effective length of a signal (padding NOT included), i.e.
    * decx::_Vector<type>::length
    */
    void FFT1D_config_gen(const size_t vec_len, de::DH* handle);
};


#endif