/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _GEMM_BLOCK_KERNEL_FP32_H_
#define _GEMM_BLOCK_KERNEL_FP32_H_


#include "../../../../core/basic.h"
#include "../../../../core/thread_management/thread_pool.h"
#include "../../../../core/utils/fragment_arrangment.h"
#include "../../../../core/thread_management/thread_arrange.h"
#include "../../GEMM_utils.h"


namespace decx
{
    /**
    * 线程函数调用，希望可以用到缓存局部性, 32x32x32 (HxWxL)
    * @param pitch_A : The pitch of matrix A (in __m256)
    * @param pitch_B : The pitch of matrix B, the same as that of matrix dst (in __m256)
    * @param proc_hA : The height of matrix A (depend on thread number)
    * @param proc_wB : The width of matrix B, in __m256 (depend on thread number)
    * @param __linear : _A->width == _B->height, in __m256
    */
    _THREAD_CALL_
    static void _GEMM_loopL_fp32_fixed_first(const float* A, const float* B, float* dst,
        decx::_C_MM_ *_MMC_props);


    _THREAD_CALL_
    static void _GEMM_loopL_fp32_fixed(const float* A, const float* B, float* dst, 
        decx::_C_MM_* _MMC_props);


    _THREAD_CALL_
    static void _GEMM_loopL_fp32_flexWH_first(const float* A, const float* B, float* dst,
        decx::_C_MM_* _MMC_props);


    _THREAD_CALL_
    static void _GEMM_loopL_fp32_flexWH(const float* A, const float* B, float* dst, 
        decx::_C_MM_* _MMC_props);

    /**
    * @param proc_dims : ~.x -> proc_WB (in __m256 x2); ~.y -> proc_HA; ~.z -> prc_linear (in __m256)
    * @param pitchA : pitch of matrix A, in float
    * @param pitchB : pitch of matrix B, in float
    */
    _THREAD_CALL_
    static void _GEMM_loopL_fp32_flexWHL_first(const float* A, const float* B, float* dst, const uint _linear, 
        decx::_C_MM_* _MMC_props);


    _THREAD_CALL_
    static void _GEMM_loopL_fp32_flexWHL(const float* A, const float* B, float* dst, const uint _linear,
        decx::_C_MM_* _MMC_props);

    /**
    * @param _linear : proc_linear, in __m256
    * @param pitchA : pitch of matrix A, in float
    * @param pitchB : pitch of matrix B, in float
    */
    _THREAD_CALL_
    static void _GEMM_loopL_fp32_flexL_first(const float* A, const float* B, float* dst, const uint _linear,
        decx::_C_MM_* _MMC_props);


    _THREAD_CALL_
    static void _GEMM_loopL_fp32_flexL(const float* A, const float* B, float* dst, const uint _linear,
        decx::_C_MM_* _MMC_props);

}


#define calc_linear_fp32(dex){    \
    reg_B0 = _mm256_load_ps(B + dex_B);        \
    reg_B1 = _mm256_load_ps(B + dex_B + 8);        \
    _accu0 = _mm256_fmadd_ps(_mm256_permutevar8x32_ps(reg_A, _mm256_set1_epi32(dex)), reg_B0, _accu0);     \
    _accu1 = _mm256_fmadd_ps(_mm256_permutevar8x32_ps(reg_A, _mm256_set1_epi32(dex)), reg_B1, _accu1);     \
    dex_B += 16;     \
}


#define _BLOCKED_GEMM_FP32_HA_ 16
#define _BLOCKED_GEMM_FP32_WB_ 4
#define _BLOCKED_GEMM_FP32_LINEAR_ 8


_THREAD_CALL_
static void decx::_GEMM_loopL_fp32_fixed_first(const float* __restrict    A, 
                                               const float* __restrict    B, 
                                               float* __restrict          dst,
                                               decx::_C_MM_ *         _MMC_props)
{
    register __m256 reg_A, reg_B0, reg_B1, 
        _accu0 = _mm256_set1_ps(0),
        _accu1 = _mm256_set1_ps(0);
    size_t dex_A = 0, tmp_dex_A = 0, dex_B = 0, dex_dst = 0;

    for (int i = 0; i < _BLOCKED_GEMM_FP32_HA_; ++i) {
        dex_B = 0;
        for (int j = 0; j < _BLOCKED_GEMM_FP32_WB_; ++j) {
            tmp_dex_A = dex_A;
#pragma unroll _BLOCKED_GEMM_FP32_LINEAR_
            for (int k = 0; k < _BLOCKED_GEMM_FP32_LINEAR_; ++k) {
                reg_A = _mm256_load_ps(A + tmp_dex_A);
                
                calc_linear_fp32(0);        calc_linear_fp32(1);
                calc_linear_fp32(2);        calc_linear_fp32(3);
                calc_linear_fp32(4);        calc_linear_fp32(5);
                calc_linear_fp32(6);        calc_linear_fp32(7);

                tmp_dex_A += 8;
            }
            _mm256_store_ps(dst + dex_dst, _accu0);
            _mm256_store_ps(dst + dex_dst + 8, _accu1);
            _accu0 = _mm256_set1_ps(0);
            _accu1 = _mm256_set1_ps(0);
            dex_dst += 16;
            dex_B += (size_t)_MMC_props->_pitchB - _BLOCKED_GEMM_FP32_LINEAR_ * 16 * 8;
        }
        dex_dst += (_MMC_props->_pitchdst - 16 * _BLOCKED_GEMM_FP32_WB_);
        dex_A += _MMC_props->_pitchA;
    }
}



_THREAD_CALL_
static void decx::_GEMM_loopL_fp32_fixed(const float* __restrict   A, 
                                         const float* __restrict   B, 
                                         float* __restrict         dst,
                                         decx::_C_MM_*         _MMC_props)
{
    register __m256 reg_A, reg_B0, reg_B1, _accu0, _accu1;
    size_t dex_A = 0, tmp_dex_A = 0, dex_B = 0, dex_dst = 0;

    for (int i = 0; i < _BLOCKED_GEMM_FP32_HA_; ++i) {
        dex_B = 0;
        for (int j = 0; j < _BLOCKED_GEMM_FP32_WB_; ++j) {
            tmp_dex_A = dex_A;
            _accu0 = _mm256_load_ps(dst + dex_dst);
            _accu1 = _mm256_load_ps(dst + dex_dst + 8);
#pragma unroll _BLOCKED_GEMM_FP32_LINEAR_
            for (int k = 0; k < _BLOCKED_GEMM_FP32_LINEAR_; ++k) {
                reg_A = _mm256_load_ps(A + tmp_dex_A);

                calc_linear_fp32(0);        calc_linear_fp32(1);
                calc_linear_fp32(2);        calc_linear_fp32(3);
                calc_linear_fp32(4);        calc_linear_fp32(5);
                calc_linear_fp32(6);        calc_linear_fp32(7);
                tmp_dex_A += 8;
            }
            _mm256_store_ps(dst + dex_dst, _accu0);
            _mm256_store_ps(dst + dex_dst + 8, _accu1);
            dex_dst += 16;
            dex_B += (size_t)_MMC_props->_pitchB - _BLOCKED_GEMM_FP32_LINEAR_ * 16 * 8;
        }
        dex_dst += (_MMC_props->_pitchdst - 16 * _BLOCKED_GEMM_FP32_WB_);
        dex_A += _MMC_props->_pitchA;
    }
}


// flexible width and height, fixed linear

_THREAD_CALL_
static void decx::_GEMM_loopL_fp32_flexWH_first(const float* __restrict    A, 
                                                const float* __restrict    B, 
                                                float* __restrict          dst,
                                                decx::_C_MM_ *         _MMC_props)
{
    register __m256 reg_A, reg_B0, reg_B1, 
        _accu0 = _mm256_set1_ps(0),
        _accu1 = _mm256_set1_ps(0);
    size_t dex_A = 0, tmp_dex_A = 0, dex_B = 0, dex_dst = 0;

    for (int i = 0; i < _MMC_props->_proc_dims.y; ++i) {
        dex_B = 0;
        for (int j = 0; j < _MMC_props->_proc_dims.x; ++j) {
            tmp_dex_A = dex_A;
#pragma unroll _BLOCKED_GEMM_FP32_LINEAR_
            for (int k = 0; k < _BLOCKED_GEMM_FP32_LINEAR_; ++k) {
                reg_A = _mm256_load_ps(A + tmp_dex_A);

                calc_linear_fp32(0);        calc_linear_fp32(1);
                calc_linear_fp32(2);        calc_linear_fp32(3);
                calc_linear_fp32(4);        calc_linear_fp32(5);
                calc_linear_fp32(6);        calc_linear_fp32(7);

                tmp_dex_A += 8;
            }
            _mm256_store_ps(dst + dex_dst, _accu0);
            _mm256_store_ps(dst + dex_dst + 8, _accu1);
            _accu0 = _mm256_set1_ps(0);
            _accu1 = _mm256_set1_ps(0);
            dex_dst += 16;
            dex_B += (size_t)_MMC_props->_pitchB - _BLOCKED_GEMM_FP32_LINEAR_ * 16 * 8;
        }
        dex_dst += (_MMC_props->_pitchdst - 16 * _MMC_props->_proc_dims.x);
        dex_A += _MMC_props->_pitchA;
    }
}


_THREAD_CALL_
static void decx::_GEMM_loopL_fp32_flexWH(const float* __restrict    A, 
                                          const float* __restrict    B, 
                                          float* __restrict          dst,
                                          decx::_C_MM_ *         _MMC_props)
{
    register __m256 reg_A, reg_B0, reg_B1, _accu0, _accu1;
    size_t dex_A = 0, tmp_dex_A = 0, dex_B = 0, dex_dst = 0;

    for (int i = 0; i < _MMC_props->_proc_dims.y; ++i) {
        dex_B = 0;
        for (int j = 0; j < _MMC_props->_proc_dims.x; ++j) {
            tmp_dex_A = dex_A;
            _accu0 = _mm256_load_ps(dst + dex_dst);
            _accu1 = _mm256_load_ps(dst + dex_dst + 8);
#pragma unroll _BLOCKED_GEMM_FP32_LINEAR_
            for (int k = 0; k < _BLOCKED_GEMM_FP32_LINEAR_; ++k) {
                reg_A = _mm256_load_ps(A + tmp_dex_A);

                calc_linear_fp32(0);        calc_linear_fp32(1);
                calc_linear_fp32(2);        calc_linear_fp32(3);
                calc_linear_fp32(4);        calc_linear_fp32(5);
                calc_linear_fp32(6);        calc_linear_fp32(7);
                tmp_dex_A += 8;
            }
            _mm256_store_ps(dst + dex_dst, _accu0);
            _mm256_store_ps(dst + dex_dst + 8, _accu1);
            dex_dst += 16;
            dex_B += (size_t)_MMC_props->_pitchB - _BLOCKED_GEMM_FP32_LINEAR_ * 16 * 8;
        }
        dex_dst += (_MMC_props->_pitchdst - 16 * _MMC_props->_proc_dims.x);
        dex_A += _MMC_props->_pitchA;
    }
}


// all the metricies are flexible

_THREAD_CALL_
static void decx::_GEMM_loopL_fp32_flexWHL_first(const float* __restrict    A, 
                                                 const float* __restrict    B, 
                                                 float* __restrict    dst,
                                                 const uint           _linear,
                                                 decx::_C_MM_*      _MMC_props)
{
    register __m256 reg_A, reg_B0, reg_B1, 
        _accu0 = _mm256_set1_ps(0),
        _accu1 = _mm256_set1_ps(0);
    size_t dex_A = 0, tmp_dex_A = 0, dex_B = 0, dex_dst = 0;

    for (int i = 0; i < _MMC_props->_proc_dims.y; ++i) {
        dex_B = 0;
        for (int j = 0; j < _MMC_props->_proc_dims.x; ++j) {
            tmp_dex_A = dex_A;

            for (int k = 0; k < _linear; ++k) {
                reg_A = _mm256_load_ps(A + tmp_dex_A);

                calc_linear_fp32(0);        calc_linear_fp32(1);
                calc_linear_fp32(2);        calc_linear_fp32(3);
                calc_linear_fp32(4);        calc_linear_fp32(5);
                calc_linear_fp32(6);        calc_linear_fp32(7);

                tmp_dex_A += 8;
            }
            _mm256_store_ps(dst + dex_dst, _accu0);
            _mm256_store_ps(dst + dex_dst + 8, _accu1);
            _accu0 = _mm256_set1_ps(0);
            _accu1 = _mm256_set1_ps(0);
            dex_dst += 16;
            dex_B += (size_t)_MMC_props->_pitchB - _linear * 16 * 8;
        }
        dex_dst += (_MMC_props->_pitchdst - 16 * _MMC_props->_proc_dims.x);
        dex_A += _MMC_props->_pitchA;
    }
}



_THREAD_CALL_
static void decx::_GEMM_loopL_fp32_flexWHL(const float* __restrict    A, 
                                           const float* __restrict    B, 
                                           float* __restrict    dst,
                                           const uint           _linear,
                                           decx::_C_MM_* _MMC_props)
{
    register __m256 reg_A, reg_B0, reg_B1, _accu0, _accu1;
    size_t dex_A = 0, tmp_dex_A = 0, dex_B = 0, dex_dst = 0;

    for (int i = 0; i < _MMC_props->_proc_dims.y; ++i) {
        dex_B = 0;
        for (int j = 0; j < _MMC_props->_proc_dims.x; ++j) {
            tmp_dex_A = dex_A;
            _accu0 = _mm256_load_ps(dst + dex_dst);
            _accu1 = _mm256_load_ps(dst + dex_dst + 8);

            for (int k = 0; k < _linear; ++k) {
                reg_A = _mm256_load_ps(A + tmp_dex_A);

                calc_linear_fp32(0);        calc_linear_fp32(1);
                calc_linear_fp32(2);        calc_linear_fp32(3);
                calc_linear_fp32(4);        calc_linear_fp32(5);
                calc_linear_fp32(6);        calc_linear_fp32(7);
                tmp_dex_A += 8;
            }
            /*_accu0 = _mm256_set1_ps(37);
            _accu1 = _mm256_set1_ps(37);*/
            _mm256_store_ps(dst + dex_dst, _accu0);
            _mm256_store_ps(dst + dex_dst + 8, _accu1);
            dex_dst += 16;
            dex_B += (size_t)_MMC_props->_pitchB - _linear * 16 * 8;
        }
        dex_dst += (_MMC_props->_pitchdst - 16 * _MMC_props->_proc_dims.x);
        dex_A += _MMC_props->_pitchA;
    }
}



// flexible linear, fixed width and height


_THREAD_CALL_
static void decx::_GEMM_loopL_fp32_flexL_first(const float* __restrict    A,
                                               const float* __restrict    B, 
                                               float* __restrict          dst, 
                                               const uint                 _linear,
                                               decx::_C_MM_*          _MMC_props)
{
    register __m256 reg_A, reg_B0, reg_B1, 
        _accu0 = _mm256_set1_ps(0),
        _accu1 = _mm256_set1_ps(0);
    size_t dex_A = 0, tmp_dex_A = 0, dex_B = 0, dex_dst = 0;

    for (int i = 0; i < _BLOCKED_GEMM_FP32_HA_; ++i) {
        dex_B = 0;

#pragma unroll _BLOCKED_GEMM_FP32_WB_
        for (int j = 0; j < _BLOCKED_GEMM_FP32_WB_; ++j) {
            tmp_dex_A = dex_A;

            for (int k = 0; k < _linear; ++k) {
                reg_A = _mm256_load_ps(A + tmp_dex_A);

                calc_linear_fp32(0);        calc_linear_fp32(1);
                calc_linear_fp32(2);        calc_linear_fp32(3);
                calc_linear_fp32(4);        calc_linear_fp32(5);
                calc_linear_fp32(6);        calc_linear_fp32(7);

                tmp_dex_A += 8;
            }
            _mm256_store_ps(dst + dex_dst, _accu0);
            _mm256_store_ps(dst + dex_dst + 8, _accu1);
            _accu0 = _mm256_set1_ps(0);
            _accu1 = _mm256_set1_ps(0);
            dex_dst += 16;
            dex_B += (size_t)_MMC_props->_pitchB - _linear * 16 * 8;
        }
        dex_dst += (_MMC_props->_pitchdst - 16 * _BLOCKED_GEMM_FP32_WB_);
        dex_A += _MMC_props->_pitchA;
    }
}



_THREAD_CALL_
static void decx::_GEMM_loopL_fp32_flexL(const float* __restrict   A, 
                                         const float* __restrict   B, 
                                         float* __restrict         dst, 
                                         const uint                _linear,
                                         decx::_C_MM_*         _MMC_props)
{   
    register __m256 reg_A, reg_B0, reg_B1, _accu0, _accu1;
    size_t dex_A = 0, tmp_dex_A = 0, dex_B = 0, dex_dst = 0;

    for (int i = 0; i < _BLOCKED_GEMM_FP32_HA_; ++i) {
        dex_B = 0;

#pragma unroll _BLOCKED_GEMM_FP32_WB_
        for (int j = 0; j < _BLOCKED_GEMM_FP32_WB_; ++j) {
            tmp_dex_A = dex_A;
            _accu0 = _mm256_load_ps(dst + dex_dst);
            _accu1 = _mm256_load_ps(dst + dex_dst + 8);

            for (int k = 0; k < _linear; ++k) {
                reg_A = _mm256_load_ps(A + tmp_dex_A);

                calc_linear_fp32(0);        calc_linear_fp32(1);
                calc_linear_fp32(2);        calc_linear_fp32(3);
                calc_linear_fp32(4);        calc_linear_fp32(5);
                calc_linear_fp32(6);        calc_linear_fp32(7);
                tmp_dex_A += 8;
            }
            _mm256_store_ps(dst + dex_dst, _accu0);
            _mm256_store_ps(dst + dex_dst + 8, _accu1);
            dex_dst += 16;
            dex_B += (size_t)_MMC_props->_pitchB - _linear * 16 * 8;
        }
        dex_dst += (_MMC_props->_pitchdst - 16 * _BLOCKED_GEMM_FP32_WB_);
        dex_A += _MMC_props->_pitchA;
    }
}

#endif