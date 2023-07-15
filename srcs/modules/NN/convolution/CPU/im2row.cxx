/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "im2row.h"



_THREAD_CALL_ static void
decx::conv::CPUK::_load_to_row_buffer(const float* __restrict src, float* __restrict row_buf, const uint load_num_v128)
{
    for (int i = 0; i < load_num_v128; ++i) {
        _mm_store_ps(row_buf + (i << 2), _mm_load_ps(src + (i << 2)));
    }
}



_THREAD_CALL_ static void
decx::conv::CPUK::_row_buffer_load_to_dst(const float* row_buf, float* dst, const uint load_num_v128)
{
    for (int i = 0; i < load_num_v128; ++i) {
        _mm_store_ps(dst + (i << 2), _mm_load_ps(row_buf + (i << 2))/*_mm_set_ps1(37)*/);
    }
}



_THREAD_CALL_ static void
decx::conv::CPUK::row_buffer_shift_L(float* row_buf, const uint unit_len_v128, const uint unit_v128_num)
{
    __m128 tmp;
    uint dex = 0;
    for (int i = 1; i < unit_v128_num; ++i) {
        dex = i * unit_len_v128 * 4;
        for (int j = 0; j < unit_len_v128; ++j) {
            tmp = _mm_load_ps(row_buf + dex);
            _mm_store_ps(row_buf + dex - ((size_t)unit_len_v128 << 2), tmp);
            dex += 4;
        }
    }
}



_THREAD_CALL_ static void
decx::conv::CPUK::_im2col_v128_row_ops(const float* __restrict            src, 
                                       float* __restrict                  row_buffer,
                                       float* __restrict                  I2C_buf, 
                                       const int                          _depth_v128, 
                                       const uint2                        ker_dims,
                                       const uint                         ker_wp,             // in (how many dpitch)
                                       const uint                         src_dp_x_wp,        // in float (contains depth)(dp_x_wp)
                                       const size_t                       WI2C,               // in float
                                       const uint                         proc_Wdst)
{
    size_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < ker_dims.y; ++i) {
        dex_dst = (size_t)i * (size_t)ker_wp * ((size_t)_depth_v128 << 2);
        dex_src = (size_t)i * src_dp_x_wp;

        decx::conv::CPUK::_load_to_row_buffer(src + dex_src, row_buffer, ker_dims.x * _depth_v128);

        decx::conv::CPUK::_row_buffer_load_to_dst(row_buffer, I2C_buf + dex_dst, ker_dims.x * _depth_v128);

        dex_src += (size_t)_depth_v128 * 4;
        dex_dst += WI2C;

        for (int j = 1; j < proc_Wdst; ++j) {
            decx::conv::CPUK::row_buffer_shift_L(row_buffer, _depth_v128, ker_dims.x);

            decx::conv::CPUK::_load_to_row_buffer(src + dex_src + (size_t)(ker_dims.x - 1) * _depth_v128 * 4, 
                row_buffer + (size_t)(ker_dims.x - 1) * _depth_v128 * 4, _depth_v128);

            decx::conv::CPUK::_row_buffer_load_to_dst(row_buffer, I2C_buf + dex_dst, ker_dims.x * _depth_v128);

            dex_src += ((size_t)_depth_v128 << 2);
            dex_dst += WI2C;
        }
    }
}



_THREAD_CALL_ static void
decx::conv::CPUK::_im2col_v128_row_ops_stride(const float* __restrict            src,
                                              float* __restrict                  row_buf,
                                              float* __restrict                  I2C_buf, 
                                              const uint                         strideX,
                                              const uint                         _depth_v128, 
                                              const uint2                        ker_dims,
                                              const uint                         ker_wp,             // in (how many dpitch)
                                              const uint                         src_dp_x_wp,        // in float (contains depth)(dp_x_wp)
                                              const size_t                       WI2C,               // in float
                                              const uint                         proc_Wdst)
{
    size_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < ker_dims.y; ++i) {
        dex_dst = (size_t)i * (size_t)ker_wp * ((size_t)_depth_v128 << 2);
        dex_src = (size_t)i * src_dp_x_wp;

        for (int j = 0; j < proc_Wdst; ++j) {
            for (int k = 0; k < ker_dims.x; ++k) {
                decx::conv::CPUK::_load_to_row_buffer(src + dex_src + k * ((size_t)_depth_v128 << 2), row_buf,
                    _depth_v128);

                decx::conv::CPUK::_row_buffer_load_to_dst(row_buf, I2C_buf + dex_dst + k * ((size_t)_depth_v128 << 2),
                    _depth_v128);
            }

            dex_src += ((size_t)_depth_v128 << 2) * strideX;
            dex_dst += WI2C;
        }
    }
}



_THREAD_FUNCTION_ void
decx::conv::CPUK::_im2col_v128(const float* __restrict            src, 
                               float* __restrict                  row_buffer,
                               float* __restrict                  I2C_buf, 
                               const uint                         _depth_v128, 
                               const uint2                        ker_dims,
                               const uint                         ker_wp,             // in (how many dpitch)
                               const uint                         src_dp_x_wp,        // in float (contains depth)(dp_x_wp)
                               const size_t                       WI2C,               // in float
                               const uint2                        proc_dims_dst)
{
    size_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < proc_dims_dst.y; ++i) {
        decx::conv::CPUK::_im2col_v128_row_ops(src + dex_src, row_buffer, I2C_buf + dex_dst, _depth_v128, 
            ker_dims, ker_wp, src_dp_x_wp, WI2C, proc_dims_dst.x);

        dex_src += src_dp_x_wp;
        dex_dst += (size_t)proc_dims_dst.x * ((size_t)WI2C);
    }
}



_THREAD_FUNCTION_ void
decx::conv::CPUK::_im2col_v128_stride(const float* __restrict            src, 
                                      float* __restrict                  row_buffer,
                                      float* __restrict                  I2C_buf, 
                                      const uint2                        strideXY,
                                      const uint                         _depth_v128, 
                                      const uint2                        ker_dims,
                                      const uint                         ker_wp,             // in (how many dpitch)
                                      const uint                         src_dp_x_wp,        // in float (contains depth)(dp_x_wp)
                                      const size_t                       WI2C,               // in float
                                      const uint2                        proc_dims_dst)
{
    size_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < proc_dims_dst.y; ++i) {
        decx::conv::CPUK::_im2col_v128_row_ops_stride(src + dex_src, row_buffer, I2C_buf + dex_dst, strideXY.x, _depth_v128, 
            ker_dims, ker_wp, src_dp_x_wp, WI2C, proc_dims_dst.x);

        dex_src += (size_t)src_dp_x_wp * (size_t)strideXY.y;
        dex_dst += (size_t)proc_dims_dst.x * ((size_t)WI2C);
    }
}