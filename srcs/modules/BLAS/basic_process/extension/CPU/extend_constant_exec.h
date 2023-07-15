/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _EXTEND_CONSTANT_EXEC_H_
#define _EXTEND_CONSTANT_EXEC_H_


#include "../../../../core/basic.h"
#include "../../../../core/utils/intrinsics_ops.h"
#include "../../../../core/thread_management/thread_pool.h"
#include "extend_reflect_exec_params.h"



namespace decx
{
    namespace bp {
        namespace CPUK {
            _THREAD_FUNCTION_ void _extend_constant1D_b32(const float* src, float* dst, const float val,
                const decx::bp::extend_reflect_exec_params* b_rfct, const size_t _actual_w_v1, const size_t _original_w_v8);


            _THREAD_FUNCTION_ void _extend_constant1D_b64(const double* src, double* dst, const double val,
                const decx::bp::extend_reflect_exec_params* b_rfct, const size_t _actual_w_v1, const size_t _original_w_v4);


            _THREAD_FUNCTION_ void _extend_constant1D_b8(const uint8_t* src, uint8_t* dst, const uint8_t val,
                const decx::bp::extend_reflect_exec_params* b_rfct, const size_t _actual_w_v1, const size_t _original_w_v8);


            _THREAD_FUNCTION_ void _extend_constant1D_b16(const uint16_t* src, uint16_t* dst, const uint16_t val,
                const decx::bp::extend_reflect_exec_params* b_rfct, const size_t _actual_w_v1, const size_t _original_w_v8);


            _THREAD_FUNCTION_ void _extend_H_constant2D_b32(const float* src, float* dst, const float val,
                const decx::bp::extend_reflect_exec_params* b_rfct, const uint32_t Wsrc,
                const uint32_t Wdst, const uint32_t _actual_w_v1, const uint2 _original_dims_v8);


            _THREAD_FUNCTION_ void _extend_H_constant2D_b64(const double* src, double* dst, const double val,
                const decx::bp::extend_reflect_exec_params* b_rfct, const uint32_t Wsrc,
                const uint32_t Wdst, const uint32_t _actual_w_v1, const uint2 _original_dims_v8);


            _THREAD_FUNCTION_ void _extend_H_constant2D_b8(const uint8_t* src, uint8_t* dst, const uint8_t val,
                const decx::bp::extend_reflect_exec_params* b_rfct, const uint32_t Wsrc,
                const uint32_t Wdst, const uint32_t _actual_w_v1, const uint2 _original_dims_v16);


            _THREAD_FUNCTION_ void _extend_H_constant2D_b16(const uint16_t* src, uint16_t* dst, const uint16_t val,
                const decx::bp::extend_reflect_exec_params* b_rfct, const uint32_t Wsrc,
                const uint32_t Wdst, const uint32_t _actual_w_v1, const uint2 _original_dims_v16);


            // vertical extension
            _THREAD_FUNCTION_ void _extend_V_constant2D_m256(float* dst, const __m256 _v_val, const uint32_t _top, const uint32_t _bottom,
                const uint Hsrc, const uint32_t Wdst);
        }
    }
}


#endif