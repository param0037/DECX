/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _ROTATION_H_
#define _ROTATION_H_


#include "../../../classes/Tensor.h"
#include "../../../core/thread_management/thread_pool.h"
#include "../../../core/thread_management/thread_arrange.h"



namespace decx {
    namespace fields {
        namespace CPUK {
            static inline _THREAD_CALL_ __m128
                rotcalc_internal_shuffle(decx::utils::simd::xmm128_reg* _px, decx::utils::simd::xmm128_reg* _py,
                    decx::utils::simd::xmm128_reg* _pz);

            /**
            * @param _proc_dims : ~.x : x_axis dims; ~.y : y_axis dims; ~.y : y_axis dims;
            * @param dp_x_wp : In float;
            * @param d_pitch : In float;
            */
            _THREAD_FUNCTION_ void
                rotation_field3D_fp32(const float* src, float* dst, const uint3 _proc_dims, const uint dp_x_wp, const uint d_pitch);
        }
    }
}


#endif