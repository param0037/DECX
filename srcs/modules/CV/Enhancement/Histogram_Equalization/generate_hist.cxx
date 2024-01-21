/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "generate_hist.h"
#include "../../../core/utils/intrinsics_ops.h"



_THREAD_CALL_ inline void
decx::vis::CPUK::_hist_accumulate_vec32(const decx::utils::simd::xmm256_reg _reg_uc32, uint32_t* _hist)
{

}



_THREAD_FUNCTION_ void
decx::vis::CPUK::_generate_hist_uc8(const float* __restrict src, 
                                    uint32_t* __restrict    hist, 
                                    const uint2             proc_dims, 
                                    const uint32_t          pitch)  // in float
{
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, proc_dims.x, 32);

    size_t dex_src = 0;
    decx::utils::simd::xmm256_reg _recv;

    for (int i = 0; i < proc_dims.y; ++i) {
        dex_src = i * pitch;
        for (int j = 0; j < f_mgr.frag_num - 1; ++j) {
            _recv._vf = _mm256_load_ps(src + dex_src);

            
            dex_src += 8;
        }
    }
}