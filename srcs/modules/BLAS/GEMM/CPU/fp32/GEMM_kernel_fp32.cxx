/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "GEMM_kernel_fp32.h"



_THREAD_FUNCTION_
void decx::gemm::CPUK::GEMM_AB_fp32_AllFixed(const float* __restrict           A,      
                                        const float* __restrict           B,      
                                        float* __restrict                 dst, 
                                        const uint                        pitchA,         // The pitch of matrix A (in float)
                                        const uint                        pitchB,         // The pitch of TMP matrix B, the same as that of matrix dst (in float)
                                        const uint                        pitchdst,       // The pitch of matrix dst (in float)
                                        const uint                        _linear,
                                        const uint2                       proc_dims)      // _A->width == _B->height, in __m256
{
    size_t dex_A = 0, dex_B = 0, dex_dst = 0;

    const uint _LoopHA = proc_dims.y / _BLOCKED_GEMM_FP32_HA_;
    const uint _LoopWB = proc_dims.x / _BLOCKED_GEMM_FP32_WB_;
    decx::_C_MM_ _MMC(pitchA, pitchB, pitchdst, _linear, proc_dims);
    
    for (int i = 0; i < _LoopHA; ++i) {
        dex_B = 0;
        for (int j = 0; j < _LoopWB; ++j) {
            decx::gemm::CPUK::GEMM_AB_fp32_Loop_fixed(A + dex_A, B + dex_B, dst + dex_dst, &_MMC);
            dex_dst += _BLOCKED_GEMM_FP32_WB_ * 16;
        }
        dex_dst += ((size_t)pitchdst * _BLOCKED_GEMM_FP32_HA_ - (16 * proc_dims.x));
        dex_A += (size_t)pitchA * _BLOCKED_GEMM_FP32_HA_;
    }
}


_THREAD_FUNCTION_
void decx::gemm::CPUK::GEMM_AB_fp32_flexWH(const float* __restrict       A,
                                      const float* __restrict       B,      
                                      float* __restrict             dst, 
                                      const uint                    pitchA,      // The pitch of matrix A (in float)
                                      const uint                    pitchB,      // The pitch of TMP matrix B, the same as that of matrix dst (in float)
                                      const uint                    pitchdst,    // The pitch of matrix dst (in float)
                                      const uint                    _linear,
                                      const uint2                   proc_dims)     // _A->width == _B->height, in __m256
{
    size_t dex_A = 0, dex_B = 0, dex_dst = 0;

    const uint _LoopHA = proc_dims.y / _BLOCKED_GEMM_FP32_HA_;
    const uint _LoopWB = proc_dims.x / _BLOCKED_GEMM_FP32_WB_;
    const uint _LoopHA_left = proc_dims.y % _BLOCKED_GEMM_FP32_HA_;
    const uint _LoopWB_left = proc_dims.x % _BLOCKED_GEMM_FP32_WB_;
    decx::_C_MM_ _MMC(pitchA, pitchB, pitchdst, _linear, make_uint2(_BLOCKED_GEMM_FP32_WB_, _BLOCKED_GEMM_FP32_HA_));
    
    for (int i = 0; i < _LoopHA; ++i) {
        dex_B = 0;
        for (int j = 0; j < _LoopWB; ++j) {
            decx::gemm::CPUK::GEMM_AB_fp32_Loop_fixed(A + dex_A, B + dex_B, dst + dex_dst, &_MMC);
            dex_dst += _BLOCKED_GEMM_FP32_WB_ * 16;
        }
        dex_dst += ((size_t)pitchdst * _BLOCKED_GEMM_FP32_HA_ - (16 * _BLOCKED_GEMM_FP32_WB_ * _LoopWB));
        dex_A += (size_t)pitchA * _BLOCKED_GEMM_FP32_HA_;

        if (_LoopWB_left) {
            _MMC._proc_dims.x = proc_dims.x % _BLOCKED_GEMM_FP32_WB_;
            decx::gemm::CPUK::GEMM_AB_fp32_Loop_flex(A + dex_A, B + dex_B, dst + dex_dst, &_MMC);
            dex_dst += 16 * _LoopWB_left;
        }
    }
    if (_LoopHA_left) {
        dex_B = 0;
        for (int j = 0; j < _LoopWB; ++j) {
            _MMC._proc_dims = make_uint2(_BLOCKED_GEMM_FP32_WB_, proc_dims.y % _BLOCKED_GEMM_FP32_HA_);
            decx::gemm::CPUK::GEMM_AB_fp32_Loop_flex(A + dex_A, B + dex_B, dst + dex_dst, &_MMC);
            dex_dst += _BLOCKED_GEMM_FP32_WB_ * 16;
        }
        dex_dst += ((size_t)pitchdst * _LoopHA_left - (16 * _BLOCKED_GEMM_FP32_WB_ * _LoopWB));
        dex_A += (size_t)pitchA * _LoopHA_left;

        if (_LoopWB_left)
        {
            _MMC._proc_dims.x = proc_dims.x % _BLOCKED_GEMM_FP32_WB_;
            decx::gemm::CPUK::GEMM_AB_fp32_Loop_flex(A + dex_A, B + dex_B, dst + dex_dst, &_MMC);
        }
    }
}
