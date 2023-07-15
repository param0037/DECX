/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _TRANSPOSE_EXEC_H_
#define _TRANSPOSE_EXEC_H_


#include "../../../../core/basic.h"
#include "../../../../core/thread_management/thread_pool.h"
#include "../../../../core/thread_management/thread_arrange.h"
#include "../../../../core/configs/config.h"
#include "../../../../core/utils/fragment_arrangment.h"
#include "../../../../core/utils/intrinsics_ops.h"


namespace decx
{
    namespace bp {
        namespace CPUK {
            /**
            * @param proc_dims_src : .x -> width of processed area, in vec4; ~.y -> height of processed area, in vec4
            * @param Wsrc : width of source matrix (in vec4)
            * @param Wdst : height of destinated matrix (in vec4)
            */
            _THREAD_FUNCTION_ void
                transpose_4x4_b32(const float* src, float* dst, const uint2 proc_dims_src, const uint Wsrc, const uint Wdst, const uint _LW);


            /**
            * @param proc_dims_src : .x -> width of processed area, in vec4; ~.y -> height of processed area, in vec4
            * @param Wsrc : width of source matrix (in vec4)
            * @param Wdst : height of destinated matrix (in vec4)
            */
            _THREAD_FUNCTION_ void
                transpose_4x4_b32_LH(const float* src, float* dst, const uint2 proc_dims_src, const uint Wsrc, const uint Wdst, const uint _LW, const uint _LH);


            /**
            * @param proc_dims_src : .x -> width of processed area, in vec4; ~.y -> height of processed area, in vec4
            * @param Wsrc : width of source matrix (in vec4)
            * @param Wdst : height of destinated matrix (in vec4)
            */
            _THREAD_FUNCTION_ void
                transpose_2x2_b64(const double* src, double* dst, const uint2 proc_dims_src, const uint Wsrc, const uint Wdst, const bool is_LW);


            /**
            * @param proc_dims_src : .x -> width of processed area, in vec4; ~.y -> height of processed area, in vec4
            * @param Wsrc : width of source matrix (in vec4)
            * @param Wdst : height of destinated matrix (in vec4)
            */
            _THREAD_FUNCTION_ void
                transpose_2x2_b64_LH(const double* src, double* dst, const uint2 proc_dims_src, const uint Wsrc, const uint Wdst, const bool is_LW, const bool is_LH);
        }

        /**
        * @param proc_dims_src : .x -> width of processed area, in vec4; ~.y -> height of processed area, in vec4
        * @param Wsrc : width of source matrix (in element)
        * @param Wdst : height of destinated matrix (in element)
        */
        void transpose_4x4_caller(const float* src, float* dst, const uint2 proc_dim_src, const uint Wsrc, const uint Wdst);


        /**
        * @param proc_dims_src : .x -> width of processed area, in vec4; ~.y -> height of processed area, in vec4
        * @param Wsrc : width of source matrix (in element)
        * @param Wdst : height of destinated matrix (in element)
        */
        void transpose_2x2_caller(const double* src, double* dst, const uint2 proc_dim_src, const uint Wsrc, const uint Wdst);
    }
}


#define _AVX_MM128_TRANSPOSE_4X4_(src4, reg4) {             \
    reg4[0] = _mm_shuffle_ps(src4[0], src4[1], 0x44);       \
    reg4[2] = _mm_shuffle_ps(src4[0], src4[1], 0xEE);       \
    reg4[1] = _mm_shuffle_ps(src4[2], src4[3], 0x44);       \
    reg4[3] = _mm_shuffle_ps(src4[2], src4[3], 0xEE);       \
                                                            \
    src4[0] = _mm_shuffle_ps(reg4[0], reg4[1], 0x88);       \
    src4[1] = _mm_shuffle_ps(reg4[0], reg4[1], 0xDD);       \
    src4[2] = _mm_shuffle_ps(reg4[2], reg4[3], 0x88);       \
    src4[3] = _mm_shuffle_ps(reg4[2], reg4[3], 0xDD);       \
}


#define _AVX_MM128_TRANSPOSE_2X2_(src2, dst2) {             \
    dst2[0] = _mm_shuffle_pd(src2[0], src2[1], 0);          \
    dst2[1] = _mm_shuffle_pd(src2[0], src2[1], 3);          \
}


#endif