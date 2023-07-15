/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _GAUSSIAN_FILTER_EXEC_H_
#define _GAUSSIAN_FILTER_EXEC_H_


#include "../../../core/basic.h"
#include "../../../DSP/convolution/CPU/sliding_window/uint8/conv2_uint8_K_loop_core.h"
#include "../../../core/thread_management/thread_pool.h"
#include "../../../core/utils/fragment_arrangment.h"
#include "../../../core/allocators.h"


namespace decx
{
    namespace vis {
        namespace CPUK {
            _THREAD_FUNCTION_
                void _gaussian_H_uint8_ST(const double* src, const float* kernel, double* dst, const uint2 proc_dim, const uint32_t Wker,
                    const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);
        }

        void _gaussian_H_uint8_caller(const double* src, const float* kernel, float* dst, const uint2 proc_dim, const uint32_t Wker,
            const uint Wsrc, const uint Wdst, const ushort reg_WL, decx::utils::_thr_1D* t1D, const uint _loop);


        void _gaussian_H_uchar4_caller(const float* src, const float* kernel, float* dst, const uint2 proc_dim, const uint32_t Wker,
            const uint Wsrc, const uint Wdst, const ushort reg_WL, decx::utils::_thr_1D* t1D, const uint _loop);


        void _gaussian_V_uint8_caller(const float* src, const float* kernel, double* dst, const uint2 proc_dim, const uint32_t Hker,
            const uint Wsrc, const uint Wdst, decx::utils::_thr_1D* t1D);
    }
}



#endif