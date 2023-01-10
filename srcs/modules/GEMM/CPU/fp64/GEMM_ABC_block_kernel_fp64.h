/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _GEMM_ABC_BLOCK_KERNEL_FP64_H_
#define _GEMM_ABC_BLOCK_KERNEL_FP64_H_


#include "../../../core/basic.h"
#include "../../../core/thread_management/thread_pool.h"
#include "../../../core/utils/fragment_arrangment.h"
#include "../../../core/thread_management/thread_arrange.h"
#include "GEMM_block_kernel_fp64.h"



namespace decx
{
    namespace gemm {
        namespace CPUK {
            /**
            * 线程函数调用，希望可以用到缓存局部性, 32x32x32 (HxWxL)
            * @param pitch_A : The pitch of matrix A (in __m256)
            * @param pitch_B : The pitch of matrix B, the same as that of matrix dst (in __m256)
            * @param proc_hA : The height of matrix A (depend on thread number)
            * @param proc_wB : The width of matrix B, in __m256 (depend on thread number)
            * @param __linear : _A->width == _B->height, in __m256
            */
            _THREAD_CALL_
            static void _GEMM_ABC_loopL_fp64_fixed_first(const double* A, const double* B, const double *C, double* dst, 
                const decx::_C_MM_* _MMC_props);



            _THREAD_CALL_
            static void _GEMM_ABC_loopL_fp64_flexWH_first(const double* A, const double* B, const double *C, double* dst, 
                const decx::_C_MM_* _MMC_props);


            /**
            * @param _linear : proc_linear, in __m256
            * @param pitchA : pitch of matrix A, in double
            * @param pitchB : pitch of matrix B, in double
            */
            _THREAD_CALL_
            static void _GEMM_ABC_loopL_fp64_flexL_first(const double* A, const double* B, const double *C, double* dst, 
                const decx::_C_MM_* _MMC_props);
        }
    }
}


_THREAD_CALL_
static void decx::gemm::CPUK::_GEMM_ABC_loopL_fp64_fixed_first(const double* __restrict    A, 
                                                   const double* __restrict    B,
                                                   const double* __restrict    C,
                                                   double* __restrict          dst,
                                                   const decx::_C_MM_*    _MMC_props)
{
    register __m256d reg_A, reg_B0, reg_B1, _accu0, _accu1, tmp;
    size_t dex_A = 0, tmp_dex_A = 0, dex_B = 0, dex_dst = 0, dex_C = 0;

    for (int i = 0; i < _BLOCKED_GEMM_FP64_HA_; ++i) {
        dex_B = 0;
        for (int j = 0; j < _BLOCKED_GEMM_FP64_WB_; ++j) {
            tmp_dex_A = dex_A;
            _accu0 = _mm256_load_pd(C + dex_C);
            _accu1 = _mm256_load_pd(C + dex_C + 4);
#pragma unroll _BLOCKED_GEMM_FP64_LINEAR_
            for (int k = 0; k < _BLOCKED_GEMM_FP64_LINEAR_; ++k) {
                reg_A = _mm256_load_pd(A + tmp_dex_A);
                
                calc_linear_fp64_0;        calc_linear_fp64_1;
                calc_linear_fp64_2;        calc_linear_fp64_3;

                tmp_dex_A += 4;
            }
            _mm256_store_pd(dst + dex_dst, _accu0);
            _mm256_store_pd(dst + dex_dst + 4, _accu1);
            dex_dst += 8;
            dex_C += 8;
            dex_B += (size_t)_MMC_props->_pitchB - _BLOCKED_GEMM_FP64_LINEAR_ * 8 * 4;
        }
        dex_dst += (_MMC_props->_pitchdst - 8 * _BLOCKED_GEMM_FP64_WB_);
        dex_C += (_MMC_props->_pitchC - 8 * _BLOCKED_GEMM_FP64_WB_);
        dex_A += _MMC_props->_pitchA;
    }
}





_THREAD_CALL_
static void decx::gemm::CPUK::_GEMM_ABC_loopL_fp64_flexWH_first(const double* __restrict    A, 
                                                    const double* __restrict    B,
                                                    const double* __restrict    C,
                                                    double* __restrict          dst,
                                                    const decx::_C_MM_*         _MMC_props)
{
    register __m256d reg_A, reg_B0, reg_B1, _accu0, _accu1, tmp;
    size_t dex_A = 0, tmp_dex_A = 0, dex_B = 0, dex_dst = 0, dex_C = 0;

    for (int i = 0; i < _MMC_props->_proc_dims.y; ++i) {
        dex_B = 0;
        for (int j = 0; j < _MMC_props->_proc_dims.x; ++j) {
            tmp_dex_A = dex_A;
            _accu0 = _mm256_load_pd(C + dex_dst);
            _accu1 = _mm256_load_pd(C + dex_dst + 4);
#pragma unroll _BLOCKED_GEMM_FP64_LINEAR_
            for (int k = 0; k < _BLOCKED_GEMM_FP64_LINEAR_; ++k) {
                reg_A = _mm256_load_pd(A + tmp_dex_A);
                
                calc_linear_fp64_0;        calc_linear_fp64_1;
                calc_linear_fp64_2;        calc_linear_fp64_3;

                tmp_dex_A += 4;
            }
            _mm256_store_pd(dst + dex_dst, _accu0);
            _mm256_store_pd(dst + dex_dst + 4, _accu1);
            dex_dst += 8;
            dex_C += 8;
            dex_B += (size_t)_MMC_props->_pitchB - _BLOCKED_GEMM_FP64_LINEAR_ * 8 * 4;
        }
        dex_dst += (_MMC_props->_pitchdst - 8 * _MMC_props->_proc_dims.x);
        dex_C += (_MMC_props->_pitchC - 8 * _MMC_props->_proc_dims.x);
        dex_A += _MMC_props->_pitchA;
    }
}





_THREAD_CALL_
static void decx::gemm::CPUK::_GEMM_ABC_loopL_fp64_flexL_first(const double* __restrict    A,
                                                   const double* __restrict    B, 
                                                   const double* __restrict    C, 
                                                   double* __restrict          dst, 
                                                   const decx::_C_MM_*         _MMC_props)
{
    register __m256d reg_A, reg_B0, reg_B1, _accu0, _accu1, tmp;
    size_t dex_A = 0, tmp_dex_A = 0, dex_B = 0, dex_dst = 0, dex_C = 0;

    for (int i = 0; i < _BLOCKED_GEMM_FP64_HA_; ++i) {
        dex_B = 0;

#pragma unroll _BLOCKED_GEMM_FP64_WB_
        for (int j = 0; j < _BLOCKED_GEMM_FP64_WB_; ++j) {
            tmp_dex_A = dex_A;
            _accu0 = _mm256_load_pd(C + dex_C);
            _accu1 = _mm256_load_pd(C + dex_C + 4);
            for (int k = 0; k < _MMC_props->_linear; ++k) {
                reg_A = _mm256_load_pd(A + tmp_dex_A);
                
                calc_linear_fp64_0;        calc_linear_fp64_1;
                calc_linear_fp64_2;        calc_linear_fp64_3;

                tmp_dex_A += 4;
            }
            _mm256_store_pd(dst + dex_dst, _accu0);
            _mm256_store_pd(dst + dex_dst + 4, _accu1);
            dex_dst += 4;
            dex_B += (size_t)_MMC_props->_pitchB - _MMC_props->_linear * 8 * 4;
        }
        dex_dst += (_MMC_props->_pitchdst - 8 * _BLOCKED_GEMM_FP64_WB_);
        dex_C += (_MMC_props->_pitchC - 8 * _BLOCKED_GEMM_FP64_WB_);
        dex_A += _MMC_props->_pitchA;
    }
}

#endif