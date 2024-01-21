/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "CUDA_FFT_configs.h"

//
//bool decx::dsp::apart_for_CUDA_FFT(int __x, std::vector<int>* res_arr)
//{
//    int prime[4] = { 5, 4, 3, 2 };
//    //int prime[4] = { 4, 2, 3, 5 };
//    int tmp = 0;
//    bool __continue = true;
//    bool round_not_f = true;
//
//    while (__continue)
//    {
//        round_not_f = true;
//        for (int i = 0; i < 4; ++i) {
//            if ((__x % prime[i]) == 0) {
//                (*res_arr).push_back(prime[i]);
//                round_not_f = false;
//
//                __x /= prime[i];
//                break;
//            }
//        }
//        if (round_not_f) {
//            __continue = false;
//        }
//    }
//    if (__x != 1) {
//        (*res_arr).push_back(__x);
//        return false;
//    }
//    else {
//        return true;
//    }
//}



decx::dsp::CUDA_FFT_Configs::CUDA_FFT_Configs() {}


void decx::dsp::CUDA_FFT_Configs::FFT1D_config_gen(const uint64_t vec_len, de::DH* handle)
{
    const int bin_len = _GetHighest(vec_len - 1);
    this->_is_2s = !(vec_len & (~(1 << bin_len)));

    // initialize the base_vector
    this->_base.clear();

    decx::dsp::fft::_radix_apart<false>(vec_len, &this->_base);

    this->_base_num = this->_base.size();

    uint64_t product = this->_base[0];
}