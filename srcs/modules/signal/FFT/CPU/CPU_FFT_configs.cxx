/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "CPU_FFT_configs.h"


bool decx::signal::apart_for_CPU_FFT(int __x, std::vector<int>* res_arr)
{
    //the order is important
    int prime[4] = { 4, 2, 3, 5 };
    int tmp = 0;

    bool __continue = true;
    bool round_not_f = true;

    while (__continue)
    {
        round_not_f = true;
        for (int i = 0; i < 4; ++i) {   // 记得改回来，改成4
            if ((__x % prime[i]) == 0) {
                (*res_arr).push_back(prime[i]);
                round_not_f = false;
                __x /= prime[i];
                break;
            }
        }
        if (round_not_f) {    // ���һ����û���ҵ����ʵ�
            __continue = false;
        }
    }
    if (__x != 1) {      // ˵��__x�޷���ȫ�ֽ�
        (*res_arr).push_back(__x);
        return false;
    }
    else {
        return true;
    }
}


bool decx::signal::CPU_FFT_Configs::FFT1D_config_gen(const size_t vec_len, de::DH* handle)
{
    const int bin_len = _GetHighest(vec_len - 1);
    this->_is_2s = !(vec_len & (~(1 << bin_len)));

    // initialize the base_vector
    this->_base.clear();

    bool crit = decx::signal::apart_for_CPU_FFT(vec_len, &this->_base);
    this->_base_num = this->_base.size();
    return crit;
}



decx::signal::CPU_FFT_Configs::~CPU_FFT_Configs() {}



template<> decx::signal::_R2C_PM::_CPU_FFT2D_Kernel_Param()
{
    this->_initial = NULL;
    this->signal_W = 0;
    this->Wsrc = 0;
    this->proc_dim = make_uint2(0, 0);
}


template<> decx::signal::_R2C_PM::_CPU_FFT2D_Kernel_Param(float* _init_ptr,
                            decx::alloc::MIF<double> _tmp0,
                            decx::alloc::MIF<double> _tmp1,
                            decx::signal::CPU_FFT_Configs* __conf,
                            const uint _signal_W,
                            const uint _Wsrc,
                            const uint2 _proc_dim)
{
    this->_initial = _init_ptr;
    this->tmp0 = _tmp0;
    this->tmp1 = _tmp1;
    this->_conf = __conf;
    this->signal_W = _signal_W;
    this->Wsrc = _Wsrc;
    this->proc_dim = _proc_dim;
}


template<>
decx::signal::_C2C_PM::_CPU_FFT2D_Kernel_Param()
{
    this->_initial = NULL;
    this->signal_W = 0;
    this->Wsrc = 0;
    this->proc_dim = make_uint2(0, 0);
}

template<>
decx::signal::_C2C_PM::_CPU_FFT2D_Kernel_Param(double* _init_ptr,
                            decx::alloc::MIF<double> _tmp0,
                            decx::alloc::MIF<double> _tmp1,
                            decx::signal::CPU_FFT_Configs* __conf,
                            const uint _signal_W,
                            const uint _Wsrc,
                            const uint2 _proc_dim)
{
    this->_initial = _init_ptr;
    this->tmp0 = _tmp0;
    this->tmp1 = _tmp1;
    this->_conf = __conf;
    this->signal_W = _signal_W;
    this->Wsrc = _Wsrc;
    this->proc_dim = _proc_dim;
}