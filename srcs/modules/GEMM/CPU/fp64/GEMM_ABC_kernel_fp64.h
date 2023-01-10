/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _GEMM_ABC_KERNEL_FP64_H_
#define _GEMM_ABC_KERNEL_FP64_H_


#include "GEMM_ABC_block_kernel_fp64.h"
#include "../../GEMM_utils.h"


namespace decx
{
    namespace gemm {
        namespace CPUK {
            _THREAD_CALL_
            static void GEMM_ABC_fp64_Loop_fixed(const double* __restrict A, const double* __restrict B, const double* C, double* __restrict dst,
                decx::_C_MM_* __restrict _MMC);


            _THREAD_CALL_
            static void GEMM_ABC_fp64_Loop_flex(const double* __restrict A, const double* __restrict B, const double* C, double* __restrict dst,
                decx::_C_MM_* __restrict _MMC);


            /**
            * All the dimention of process area can be equally divided by blocked_GEMM
            * @param pitch_A : The pitch of matrix A (in __m256)
            * @param pitch_B : The pitch of matrix B, the same as that of matrix dst (in __m256)
            * @param proc_hA : The height of matrix A (depend on thread number)
            * @param proc_wB : The width of matrix B, in __m256 (depend on thread number)
            * @param __linear : _A->width == _B->height, in __m256
            */
            _THREAD_FUNCTION_
            void GEMM_ABC_fp64_AllFixed(const double* A, const double* B, const double* C, double* dst, const uint pitchA, const uint pitchB, const uint pitchdst,
                const uint pitchC, const uint _linear, const uint2 proc_dims);

            /**
            * The linear region of process area can be equally divided by blocked_GEMM, the width and height can not be divided equally
            * @param pitch_A : The pitch of matrix A (in __m256)
            * @param pitch_B : The pitch of matrix B, the same as that of matrix dst (in __m256)
            * @param proc_hA : The height of matrix A (depend on thread number)
            * @param proc_wB : The width of matrix B, in __m256 (depend on thread number)
            * @param __linear : _A->width == _B->height, in __m256
            */
            _THREAD_FUNCTION_
            void GEMM_ABC_fp64_flexWH(const double* A, const double* B, const double* C, double* dst, const uint pitchA, const uint pitchB, const uint pitchdst,
                const uint pitchC, const uint _linear, const uint2 proc_dims);
        }
    }
}



_THREAD_CALL_
static void decx::gemm::CPUK::GEMM_ABC_fp64_Loop_fixed(const double* A, const double* B, const double* C, double* dst, decx::_C_MM_* __restrict _MMC)
{
    size_t dex_B = 0, dex_A = 0;
    const uint _LoopL = _MMC->_linear / 4 / _BLOCKED_GEMM_FP64_LINEAR_;
    const uint _LoopL_left = (_MMC->_linear / 4) % _BLOCKED_GEMM_FP64_LINEAR_;
    // __linear...
    decx::gemm::CPUK::_GEMM_ABC_loopL_fp64_fixed_first(A + dex_A, B + dex_B, C, dst, _MMC);
    dex_A += _BLOCKED_GEMM_FP64_LINEAR_ * 4;
    dex_B += _BLOCKED_GEMM_FP64_LINEAR_ * 8 * 4;

    for (int k = 1; k < _LoopL; ++k) {
        decx::_GEMM_loopL_fp64_fixed(A + dex_A, B + dex_B, dst, _MMC);
        dex_A += _BLOCKED_GEMM_FP64_LINEAR_ * 4;
        dex_B += _BLOCKED_GEMM_FP64_LINEAR_ * 8 * 4;
    }
    if (_LoopL_left)
        decx::_GEMM_loopL_fp64_flexL(A + dex_A, B + dex_B, dst, _LoopL_left, _MMC);
}


_THREAD_CALL_
static void decx::gemm::CPUK::GEMM_ABC_fp64_Loop_flex(const double* A, const double* B, const double* C, double* dst, decx::_C_MM_* __restrict _MMC)
{
    size_t dex_B = 0, dex_A = 0;
    const uint _LoopL = _MMC->_linear / 4 / _BLOCKED_GEMM_FP64_LINEAR_;
    const uint _LoopL_left = (_MMC->_linear / 4) % _BLOCKED_GEMM_FP64_LINEAR_;
    // __linear...
    decx::gemm::CPUK::_GEMM_ABC_loopL_fp64_flexWH_first(A + dex_A, B + dex_B, C, dst, _MMC);
    dex_A += _BLOCKED_GEMM_FP64_LINEAR_ * 4;
    dex_B += _BLOCKED_GEMM_FP64_LINEAR_ * 8 * 4;

    for (int k = 1; k < _LoopL; ++k) {
        decx::_GEMM_loopL_fp64_flexWH(A + dex_A, B + dex_B, dst, _MMC);
        dex_A += _BLOCKED_GEMM_FP64_LINEAR_ * 4;
        dex_B += _BLOCKED_GEMM_FP64_LINEAR_ * 8 * 4;
    }
    if (_LoopL_left)
        decx::_GEMM_loopL_fp64_flexWHL(A + dex_A, B + dex_B, dst, _LoopL_left, _MMC);
}




#endif