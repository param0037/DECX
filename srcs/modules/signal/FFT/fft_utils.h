/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _FFT_UTILS_H_
#define _FFT_UTILS_H_

#include "../CUDA_cpf32.cuh"


#define FFT2D_BLOCK_SIZE 16

#define _CUDA_FFT1D_BLOCK_SIZE 512
#define _CUDA_FFT_SHMEM_SIZE_ _CUDA_FFT1D_BLOCK_SIZE * 8  // in byte

#define Pi          3.1415926f
#define Two_Pi      6.2831853f
#define Four_Pi     12.5663706f
#define Six_Pi      18.8495559f
#define Eight_Pi    25.1327412f


namespace decx
{
    namespace signal
    {
        static bool check_apart(int __x);
    }
}


static
bool decx::signal::check_apart(int __x)
{
    int prime[4] = { 5, 4, 3, 2 };
    int tmp = 0;
    // ���ж���һ��ȫ���Ҳ������ʵģ�break��whileѭ�����������
    bool __continue = true;
    bool round_not_f = true;

    while (__continue)
    {
        round_not_f = true;
        for (int i = 0; i < 4; ++i) {
            if ((__x % prime[i]) == 0) {
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
        return false;
    }
    else {
        return true;
    }
}


namespace de
{
    namespace signal {
        enum FFT_flags {
            FFT_R2C = 0,
            FFT_C2C = 1,
            IFFT_C2C = 2,
            IFFT_C2R = 3
        };
    }
}


#endif
