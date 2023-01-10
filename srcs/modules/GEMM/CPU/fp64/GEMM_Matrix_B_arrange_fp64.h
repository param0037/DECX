/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _GEMM_MATRIX_B_ARRANGE_FP64_H_
#define _GEMM_MATRIX_B_ARRANGE_FP64_H_

#include "../../../core/basic.h"
#include "../../../core/thread_management/thread_pool.h"
#include "../../../core/thread_management/thread_arrange.h"
#include "../../../core/utils/fragment_arrangment.h"


namespace decx
{
    _THREAD_FUNCTION_
    /**
    * @param eff_L_len : the effective lenght of linear region -> B->height
    * @param lane_num : the lane number of matrix B -> B->pitch / 16
    * @param WsrcB : the pitch of matrix B (in float)
    * @param WtmpB : the pitch of chache B (in float)
    */
    static void _sort_ST_MatB_fp64(double* __restrict   srcB, 
                            double* __restrict   dstB,
                            const uint          eff_L_len,
                            const uint          lane_num,
                            const uint          WsrcB,
                            const uint          WtmpB);


    _THREAD_FUNCTION_
    /**
    * @param eff_L_len : the effective lenght of linear region -> B->height
    * @param lane_num : the lane number of matrix B -> B->pitch / 16
    * @param WsrcB : the pitch of matrix B (in float)
    * @param WtmpB : the pitch of chache B (in float)
    */
    static void _sort_ST_MatB_fp64_L8(double* __restrict   srcB, 
                               double* __restrict   dstB,
                               const uint          eff_L_len,
                               const uint          lane_num,
                               const uint          WsrcB,
                               const uint          WtmpB);


    /**
    * @param eff_L_len : the effective lenght of linear region -> B->height
    * @param lane_num : the lane number of matrix B -> B->pitch / 16
    * @param WsrcB : the pitch of matrix B (in float)
    * @param WtmpB : the pitch of chache B (in float)
    */
    void arrange_MatB_fp64_caller(double*                      srcB, 
                                  double*                      dstB, 
                                  const uint                  WsrcB,
                                  const uint                  WtmpB,
                                  const uint                  _eff_L_len,
                                  const bool                  is_L8,
                                  decx::utils::_thr_2D*       t2D,
                                  decx::utils::frag_manager*  f_mgr);
}



_THREAD_FUNCTION_
static void decx::_sort_ST_MatB_fp64(double* __restrict     srcB, 
                              double* __restrict     dstB,
                              const uint             eff_L_len,
                              const uint             lane_num,
                              const uint             WsrcB,
                              const uint             WtmpB)
{
    size_t dex_src = 0, dex_dst = 0, tmp_dex_src = 0, tmp_dex_dst = 0;

    for (int i = 0; i < lane_num; ++i)
    {
        tmp_dex_src = dex_src;
        tmp_dex_dst = dex_dst;
        for (int _L = 0; _L < eff_L_len; ++_L)
        {
            _mm256_storeu_pd(dstB + tmp_dex_dst, _mm256_loadu_pd(srcB + tmp_dex_src));
            _mm256_storeu_pd(dstB + tmp_dex_dst + 4, _mm256_loadu_pd(srcB + tmp_dex_src + 4));
            tmp_dex_dst += 8;
            tmp_dex_src += WsrcB;
        }
        dex_src += 8;
        dex_dst += WtmpB;
    }
}


_THREAD_FUNCTION_
static void decx::_sort_ST_MatB_fp64_L8(double* __restrict     srcB, 
                                 double* __restrict     dstB,
                                 const uint            eff_L_len,
                                 const uint            lane_num,
                                 const uint            WsrcB,
                                 const uint            WtmpB)
{
    size_t dex_src = 0, dex_dst = 0, tmp_dex_src = 0, tmp_dex_dst = 0;

    for (int i = 0; i < lane_num; ++i)
    {
        tmp_dex_src = dex_src;
        tmp_dex_dst = dex_dst;
        for (int _L = 0; _L < eff_L_len; ++_L)
        {
            _mm256_storeu_pd(dstB + tmp_dex_dst, _mm256_loadu_pd(srcB + tmp_dex_src));
            _mm256_storeu_pd(dstB + tmp_dex_dst + 4, _mm256_loadu_pd(srcB + tmp_dex_src + 4));
            tmp_dex_dst += 8;
            tmp_dex_src += WsrcB;
        }
        dex_src += 8;
        dex_dst += WtmpB;
    }
    tmp_dex_src = dex_src;
    tmp_dex_dst = dex_dst;
    for (int _L = 0; _L < eff_L_len; ++_L)
    {
        _mm256_storeu_pd(dstB + tmp_dex_dst, _mm256_loadu_pd(srcB + tmp_dex_src));
        tmp_dex_dst += 8;
        tmp_dex_src += WsrcB;
    }
}


#endif