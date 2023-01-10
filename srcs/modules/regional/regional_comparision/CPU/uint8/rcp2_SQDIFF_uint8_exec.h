/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _RCP_SQDIFF_UINT8_EXEC_H_
#define _RCP_SQDIFF_UINT8_EXEC_H_



#include "../rcp_SQDIFF_T_loop_core.h"
#include "../rcp2_sliding_window_macros.h"
#include "../../../../core/utils/fragment_arrangment.h"


#define _BLOCKED_RCP2_UINT8_H_ 8
#define _BLOCKED_RCP2_UINT8_W_ 8


namespace decx
{
    namespace rcp {
        namespace CPUK {
            static _THREAD_CALL_ void _rcp2_SQDIFF_rN_rect_fixed_uint8_ST(double* __restrict src, uint8_t* kernel, float* __restrict dst,
                const uint2 ker_dims, const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);



            static _THREAD_CALL_ void _rcp2_SQDIFF_rN_rect_flex_uint8_ST(double* __restrict src, uint8_t* kernel, float* __restrict dst,
                const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);

            // ------------------------------------------------------- norm -------------------------------------------------------

            /*
            * @param _sqrt_k_sum : SUM<i, j>(kernel(i, j) ^ 2)
            */
            static _THREAD_CALL_ void _rcp2_SQDIFF_norm_rN_rect_fixed_uint8_ST(double* __restrict src, uint8_t* kernel, float* __restrict dst,
                const float _sqrt_k_sum, const uint2 ker_dims, const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);



            static _THREAD_CALL_ void _rcp2_SQDIFF_norm_rN_rect_flex_uint8_ST(double* __restrict src, uint8_t* kernel, float* __restrict dst,
                const float _sqrt_k_sum, const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);
        }
    }
}



namespace decx
{
    namespace rcp {
        namespace CPUK {
            /*
            * This function is suitable only when 3 __m256 are needed to cover all the data on one row
            * which indicates that half_kerdim.x -> [5, 8]
            *
            * @param proc_dim : .x -> in _m256, the width of proccess area of signle thread (on dst matrix)
            *                   .y -> the height of proccess area of signle thread (on dst matrix)
            * @param ker_dim : .x -> the width of kernel (in float); .y -> the height of kernel
            * @param reg_WL : the leftover on width. ( = (half_ker_dims.x * 2 + 8) - 2 * 8)
            * @param Wsrc : the pitch of src matrix, in __m256
            */
            _THREAD_FUNCTION_
            void _rcp2_SQDIFF_uint8_ST(double* src, uint8_t* kernel, float* dst, const uint2 proc_dim, const uint2 ker_dims,
                const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);


            _THREAD_FUNCTION_
            void _rcp2_SQDIFF_norm_uint8_ST(double* src, uint8_t* kernel, float* dst, const float _sqrt_k_sum, const uint2 proc_dim, const uint2 ker_dims,
                const uint Wsrc, const uint Wdst, const ushort reg_WL, const uint _loop);
        }
    }
}



namespace decx
{
    namespace rcp{
        void _rcp2_SQDIFF_rN_uint8_caller(double* src, uint8_t* kernel, float* dst, const uint2 proc_dim, const uint2 ker_dims,
            const uint Wsrc, const uint Wdst, const ushort reg_WL,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
            const uint _loop);


        void _rcp2_SQDIFF_NORM_rN_uint8_caller(double* src, uint8_t* kernel, float* dst, const float _sqrt_k_sum, const uint2 proc_dim, const uint2 ker_dims,
            const uint Wsrc, const uint Wdst, const ushort reg_WL,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
            const uint _loop);
    }
}



namespace decx
{
    namespace rcp{
        template <bool _norm>
        static void _rcp2_SQDIFF_uint8_organiser(double* src, uint8_t* kernel, float* dst, const float _sqrt_k_sum,
                                  const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc,
                                  const uint Wdst, decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);
    }
}


template <bool _norm>
void decx::rcp::_rcp2_SQDIFF_uint8_organiser(double*                       src,
                                            uint8_t*                             kernel, 
                                            float*                             dst, 
                                            const float                        _sqrt_k_sum,
                                            const uint2                        proc_dim, 
                                            const uint2                        ker_dims,
                                            const uint                         Wsrc,
                                            const uint                         Wdst,
                                            decx::utils::_thr_1D*              t1D,
                                            decx::utils::frag_manager*         f_mgr)
{
    const uint _loop = (ker_dims.x - 1) / 16;
    ushort reg_WL = (ushort)(ker_dims.x - _loop * 16);
    if (_norm) {
        decx::rcp::_rcp2_SQDIFF_NORM_rN_uint8_caller(src, kernel, dst, _sqrt_k_sum, proc_dim, ker_dims, Wsrc, Wdst, reg_WL, t1D, f_mgr, _loop);
    }
    else {
        decx::rcp::_rcp2_SQDIFF_rN_uint8_caller(src, kernel, dst, proc_dim, ker_dims, Wsrc, Wdst, reg_WL, t1D, f_mgr, _loop);
    }
}


#endif