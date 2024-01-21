/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "hist_gen_exec.h"


_THREAD_FUNCTION_ void
decx::bp::CPUK::_histgen2D_u8(const uint8_t* __restrict     src,
                              uint64_t* __restrict          _hist,
                              const uint2                   proc_dims,
                              const uint32_t                Wsrc,
                              const uint8_t                 _leagal_space_v32)
{
    uint64_t dex_src = 0;
    uint8_t tmp_v32[32];

    for (int i = 0; i < 256 / 4; ++i) {
        _mm256_store_pd((double*)(_hist + i * 4), _mm256_setzero_pd());
    }

    for (int i = 0; i < proc_dims.y; ++i)
    {
        dex_src = i * Wsrc;
        for (int j = 0; j < proc_dims.x - 1; ++j) {
            _mm256_store_ps((float*)tmp_v32, _mm256_load_ps((float*)(src + dex_src)));
#ifdef __GNUC__
#pragma unroll 32
#endif
            for (int vec_id = 0; vec_id < 32; ++vec_id) {
                ++_hist[tmp_v32[vec_id]];
            }
            dex_src += 32;
        }
        _mm256_store_ps((float*)tmp_v32, _mm256_load_ps((float*)(src + dex_src)));
        for (int vec_id = 0; vec_id < _leagal_space_v32; ++vec_id) {
            ++_hist[tmp_v32[vec_id]];
        }
    }
}