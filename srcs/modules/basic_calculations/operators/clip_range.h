/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _CLIP_RANGE_H_
#define _CLIP_RANGE_H_


#include "../../core/thread_management/thread_pool.h"
#include "../../core/thread_management/thread_arrange.h"


namespace decx
{
    namespace calc {
        namespace CPUK {
            _THREAD_FUNCTION_ void _clip_range_fp32(const float* src, const float2 range, float* dst, const uint64_t proc_len);

            _THREAD_FUNCTION_ void _clip_range_fp64(const double* src, const double2 range, double* dst, const uint64_t proc_len);
        }
    }
}



#endif