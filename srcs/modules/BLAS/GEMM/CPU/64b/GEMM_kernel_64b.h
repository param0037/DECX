/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _GEMM_KERNEL_64B_H_
#define _GEMM_KERNEL_64B_H_

#include "../fp64/GEMM_block_kernel_fp64.h"
#include "../cpl32/GEMM_block_kernel_cpl32.h"


namespace decx
{
    namespace gemm {
        namespace CPUK {
            template <bool _is_cpl>
            _THREAD_CALL_
            static void GEMM_AB_fp64_Loop_fixed(const double* __restrict A, const double* __restrict B, double* __restrict dst, 
                decx::_C_MM_* __restrict _MMC);

            template <bool _is_cpl>
            _THREAD_CALL_
            static void GEMM_AB_fp64_Loop_flex(const double* __restrict A, const double* __restrict B, double* __restrict dst, 
                decx::_C_MM_* __restrict _MMC);


            /**
            * All the dimention of process area can be equally divided by blocked_GEMM
            * @param pitch_A : The pitch of matrix A (in __m256)
            * @param pitch_B : The pitch of matrix B, the same as that of matrix dst (in __m256)
            * @param proc_hA : The height of matrix A (depend on thread number)
            * @param proc_wB : The width of matrix B, in __m256 (depend on thread number)
            * @param __linear : _A->width == _B->height, in __m256
            */
            template <bool _is_cpl> _THREAD_FUNCTION_
            void GEMM_AB_fp64_AllFixed(const double* A, const double* B, double* dst, const uint pitchA, const uint pitchB, const uint pitchdst,
                const uint _linear, const uint2 proc_dims);

            /**
            * The linear region of process area can be equally divided by blocked_GEMM, the width and height can not be divided equally
            * @param pitch_A : The pitch of matrix A (in __m256)
            * @param pitch_B : The pitch of matrix B, the same as that of matrix dst (in __m256)
            * @param proc_hA : The height of matrix A (depend on thread number)
            * @param proc_wB : The width of matrix B, in __m256 (depend on thread number)
            * @param __linear : _A->width == _B->height, in __m256
            */
            template <bool _is_cpl> _THREAD_FUNCTION_
            void GEMM_AB_fp64_flexWH(const double* A, const double* B, double* dst, const uint pitchA, const uint pitchB, const uint pitchdst,
                const uint _linear, const uint2 proc_dims);
        }
    }
}


template <bool _is_cpl>
_THREAD_CALL_
static void decx::gemm::CPUK::GEMM_AB_fp64_Loop_fixed(const double* A, const double* B, double* dst, decx::_C_MM_* __restrict _MMC)
{
    size_t dex_B = 0, dex_A = 0;
    const uint _LoopL = _MMC->_linear / 4 / _BLOCKED_GEMM_FP64_LINEAR_;
    const uint _LoopL_left = (_MMC->_linear / 4) % _BLOCKED_GEMM_FP64_LINEAR_;

    // __linear...
    if (_is_cpl) { decx::_GEMM_loopL_cpl32_fixed_first(A + dex_A, B + dex_B, dst, _MMC); }
    else { decx::_GEMM_loopL_fp64_fixed_first(A + dex_A, B + dex_B, dst, _MMC); }

    dex_A += _BLOCKED_GEMM_FP64_LINEAR_ * 4;
    dex_B += _BLOCKED_GEMM_FP64_LINEAR_ * 8 * 4;

    for (int k = 1; k < _LoopL; ++k) {
        if (_is_cpl) { decx::_GEMM_loopL_cpl32_fixed(A + dex_A, B + dex_B, dst, _MMC); }
        else { decx::_GEMM_loopL_fp64_fixed(A + dex_A, B + dex_B, dst, _MMC); }

        dex_A += _BLOCKED_GEMM_FP64_LINEAR_ * 4;
        dex_B += _BLOCKED_GEMM_FP64_LINEAR_ * 8 * 4;
    }
    if (_LoopL_left) {
        if (_is_cpl) { decx::_GEMM_loopL_cpl32_fixed(A + dex_A, B + dex_B, dst, _MMC); }
        else { decx::_GEMM_loopL_fp64_flexL(A + dex_A, B + dex_B, dst, _LoopL_left, _MMC); }
    }
}


template <bool _is_cpl>
_THREAD_CALL_
static void decx::gemm::CPUK::GEMM_AB_fp64_Loop_flex(const double* A, const double* B, double* dst, decx::_C_MM_* __restrict _MMC)
{
    size_t dex_B = 0, dex_A = 0;
    const uint _LoopL = _MMC->_linear / 4 / _BLOCKED_GEMM_FP64_LINEAR_;
    const uint _LoopL_left = (_MMC->_linear / 4) % _BLOCKED_GEMM_FP64_LINEAR_;

    // __linear...
    if (_is_cpl) { decx::_GEMM_loopL_cpl32_flexWH_first(A + dex_A, B + dex_B, dst, _MMC); }
    else { decx::_GEMM_loopL_fp64_flexWH_first(A + dex_A, B + dex_B, dst, _MMC); }

    dex_A += _BLOCKED_GEMM_FP64_LINEAR_ * 4;
    dex_B += _BLOCKED_GEMM_FP64_LINEAR_ * 8 * 4;

    for (int k = 1; k < _LoopL; ++k) {
        if (_is_cpl) { decx::_GEMM_loopL_cpl32_flexWH(A + dex_A, B + dex_B, dst, _MMC); }
        else { decx::_GEMM_loopL_fp64_flexWH(A + dex_A, B + dex_B, dst, _MMC); }
        dex_A += _BLOCKED_GEMM_FP64_LINEAR_ * 4;
        dex_B += _BLOCKED_GEMM_FP64_LINEAR_ * 8 * 4;
    }
    if (_LoopL_left) {
        if (_is_cpl) { decx::_GEMM_loopL_cpl32_flexWHL(A + dex_A, B + dex_B, dst, _LoopL_left, _MMC); }
        else { decx::_GEMM_loopL_fp64_flexWHL(A + dex_A, B + dex_B, dst, _LoopL_left, _MMC); }
    }
}



#endif