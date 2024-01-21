/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "VEVID.h"


_THREAD_FUNCTION_ void _VEVID_u8_kernel(const double* __restrict src, float* __restrict dst, 
            const uint32_t pitchsrc_v8, const uint32_t pitchdst_v1, const uint32_t proc_H)
{
    double _recv_pixels;
    decx::utils::simd::xmm256_reg;

    for (uint32_t i = 0; i < proc_H; ++i){
        // for (uint32_t j = 0; j < )
    }
}