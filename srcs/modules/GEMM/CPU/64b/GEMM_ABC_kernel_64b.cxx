/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "GEMM_ABC_kernel_64b.h"


template <bool _is_cpl> _THREAD_FUNCTION_
void decx::gemm::CPUK::GEMM_ABC_fp64_AllFixed(const double* __restrict       A,      
                                         const double* __restrict       B,   
                                         const double* __restrict       C,   
                                         double* __restrict             dst, 
                                         const uint                    pitchA,         // The pitch of matrix A (in double)
                                         const uint                    pitchB,         // The pitch of TMP matrix B, the same as that of matrix dst (in double)
                                         const uint                    pitchC,         // The pitch of TMP matrix B, the same as that of matrix dst (in double)
                                         const uint                    pitchdst,       // The pitch of matrix dst (in double)
                                         const uint                    _linear,
                                         const uint2                   proc_dims)      // _A->width == _B->height, in __m256
{
    size_t dex_A = 0, dex_B = 0, dex_dst = 0, dex_C = 0;

    const uint _LoopHA = proc_dims.y / _BLOCKED_GEMM_FP64_HA_;
    const uint _LoopWB = proc_dims.x / _BLOCKED_GEMM_FP64_WB_;
    decx::_C_MM_ _MMC(pitchA, pitchB, pitchdst, _linear, proc_dims, pitchC);
    
    for (int i = 0; i < _LoopHA; ++i) {
        dex_B = 0;
        for (int j = 0; j < _LoopWB; ++j) {
            decx::gemm::CPUK::GEMM_ABC_fp64_Loop_fixed<_is_cpl>(A + dex_A, B + dex_B, C + dex_C, dst + dex_dst, &_MMC);
            dex_dst += _BLOCKED_GEMM_FP64_WB_ * 8;
            dex_C += _BLOCKED_GEMM_FP64_WB_ * 8;
        }
        dex_dst += ((size_t)pitchdst * _BLOCKED_GEMM_FP64_HA_ - (8 * proc_dims.x));
        dex_C += ((size_t)pitchC * _BLOCKED_GEMM_FP64_HA_ - (8 * proc_dims.x));
        dex_A += (size_t)pitchA * _BLOCKED_GEMM_FP64_HA_;
    }
}



template <bool _is_cpl> _THREAD_FUNCTION_
void decx::gemm::CPUK::GEMM_ABC_fp64_flexWH(const double* __restrict           A,
                                       const double* __restrict           B,
                                       const double* __restrict           C,
                                       double* __restrict                 dst, 
                                       const uint                        pitchA,      // The pitch of matrix A (in double)
                                       const uint                        pitchB,      // The pitch of TMP matrix B, the same as that of matrix dst (in double)
                                       const uint                        pitchC,      // The pitch of TMP matrix C, the same as that of matrix dst (in double)
                                       const uint                        pitchdst,    // The pitch of matrix dst (in double)
                                       const uint                        _linear,
                                       const uint2                       proc_dims)     // _A->width == _B->height, in __m256
{
    size_t dex_A = 0, dex_B = 0, dex_dst = 0, dex_C = 0;

    const uint _LoopHA = proc_dims.y / _BLOCKED_GEMM_FP64_HA_;
    const uint _LoopWB = proc_dims.x / _BLOCKED_GEMM_FP64_WB_;
    const uint _LoopHA_left = proc_dims.y % _BLOCKED_GEMM_FP64_HA_;
    const uint _LoopWB_left = proc_dims.x % _BLOCKED_GEMM_FP64_WB_;
    decx::_C_MM_ _MMC(pitchA, pitchB, pitchdst, _linear, make_uint2(_BLOCKED_GEMM_FP64_WB_, _BLOCKED_GEMM_FP64_HA_), pitchC);
    
    for (int i = 0; i < _LoopHA; ++i) {
        dex_B = 0;
        for (int j = 0; j < _LoopWB; ++j) {
            decx::gemm::CPUK::GEMM_ABC_fp64_Loop_fixed<_is_cpl>(A + dex_A, B + dex_B, C + dex_C, dst + dex_dst, &_MMC);
            dex_dst += _BLOCKED_GEMM_FP64_WB_ * 8;
            dex_C += _BLOCKED_GEMM_FP64_WB_ * 8;
        }
        dex_dst += ((size_t)pitchdst * _BLOCKED_GEMM_FP64_HA_ - (8 * _BLOCKED_GEMM_FP64_WB_ * _LoopWB));
        dex_C += ((size_t)pitchC * _BLOCKED_GEMM_FP64_HA_ - (8 * _BLOCKED_GEMM_FP64_WB_ * _LoopWB));
        dex_A += (size_t)pitchA * _BLOCKED_GEMM_FP64_HA_;

        if (_LoopWB_left) {
            _MMC._proc_dims.x = proc_dims.x % _BLOCKED_GEMM_FP64_WB_;
            decx::gemm::CPUK::GEMM_ABC_fp64_Loop_flex<_is_cpl>(A + dex_A, B + dex_B, C + dex_C, dst + dex_dst, &_MMC);
            dex_dst += 8 * _LoopWB_left;
            dex_C += 8 * _LoopWB_left;
        }
    }
    if (_LoopHA_left) {
        dex_B = 0;
        for (int j = 0; j < _LoopWB; ++j) {
            _MMC._proc_dims = make_uint2(_BLOCKED_GEMM_FP64_WB_, proc_dims.y % _BLOCKED_GEMM_FP64_HA_);
            decx::gemm::CPUK::GEMM_ABC_fp64_Loop_flex<_is_cpl>(A + dex_A, B + dex_B, C + dex_C, dst + dex_dst, &_MMC);
            dex_dst += _BLOCKED_GEMM_FP64_WB_ * 8;
            dex_C += _BLOCKED_GEMM_FP64_WB_ * 8;
        }
        dex_dst += ((size_t)pitchdst * _LoopHA_left - (8 * _BLOCKED_GEMM_FP64_WB_ * _LoopWB));
        dex_C += ((size_t)pitchC * _LoopHA_left - (8 * _BLOCKED_GEMM_FP64_WB_ * _LoopWB));
        dex_A += (size_t)pitchA * _LoopHA_left;

        if (_LoopWB_left)
        {
            _MMC._proc_dims.x = proc_dims.x % _BLOCKED_GEMM_FP64_WB_;
            decx::gemm::CPUK::GEMM_ABC_fp64_Loop_flex<_is_cpl>(A + dex_A, B + dex_B, C + dex_C, dst + dex_dst, &_MMC);
        }
    }
}



template void decx::gemm::CPUK::GEMM_ABC_fp64_AllFixed<true>(const double* A, const double* B, const double* C, double* dst, const uint pitchA, const uint pitchB, const uint pitchdst,
    const uint pitchC, const uint _linear, const uint2 proc_dims);


template void decx::gemm::CPUK::GEMM_ABC_fp64_AllFixed<false>(const double* A, const double* B, const double* C, double* dst, const uint pitchA, const uint pitchB, const uint pitchdst,
    const uint pitchC, const uint _linear, const uint2 proc_dims);


template void decx::gemm::CPUK::GEMM_ABC_fp64_flexWH<true>(const double* A, const double* B, const double* C, double* dst, const uint pitchA, const uint pitchB, const uint pitchdst,
    const uint pitchC, const uint _linear, const uint2 proc_dims);


template void decx::gemm::CPUK::GEMM_ABC_fp64_flexWH<false>(const double* A, const double* B, const double* C, double* dst, const uint pitchA, const uint pitchB, const uint pitchdst,
    const uint pitchC, const uint _linear, const uint2 proc_dims);