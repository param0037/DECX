/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _IM2COL_H_
#define _IM2COL_H_


#include "../../../core/thread_management/thread_arrange.h"
#include "../../../core/thread_management/thread_pool.h"


namespace decx
{
    namespace conv {
        namespace CPUK {
            _THREAD_CALL_ static void
            _load_to_row_buffer(const float* src, float* row_buf, const uint load_num_v128);


            _THREAD_CALL_ static void
            _row_buffer_load_to_dst(const float* row_buf, float* dst, const uint load_num_v128);


            _THREAD_CALL_ static void
            row_buffer_shift_L(float* row_buf, const uint unit_len_v128, const uint unit_v128_num);


            _THREAD_CALL_ static void
            _im2col_v128_row_ops(const float* src, float* row_buffer, float* I2C_buf, const int _depth_v128, 
                const uint2 ker_dims, const uint ker_wp, const uint src_dp_x_wp, const size_t WI2C, const uint proc_Wdst);


            _THREAD_CALL_ static void
            _im2col_v128_row_ops_stride(const float* src, float* row_buf, float* I2C_buf, const uint stride, const uint _depth_v128, 
                const uint2 ker_dims, const uint ker_wp, const uint src_dp_x_wp, const size_t WI2C, const uint proc_Wdst);


            _THREAD_FUNCTION_ void
            _im2col_v128(const float* src, float* row_buffer, float* I2C_buf, const uint _depth_v128,
                const uint2 ker_dims, const uint ker_wp, const uint src_dp_x_wp, const size_t WI2C, const uint2 proc_dims_dst);


            _THREAD_FUNCTION_ void
            _im2col_v128_stride(const float* src, float* row_buffer, float* I2C_buf, const uint2 strideXY, const uint _depth_v128,
                const uint2 ker_dims, const uint ker_wp, const uint src_dp_x_wp, const size_t WI2C, const uint2 proc_dims_dst);
        }
    }
}


#endif