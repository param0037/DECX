/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _RCP2_CCOEFF_UINT8_EXEC_H_
#define _RCP2_CCOEFF_UINT8_EXEC_H_


#include "../rcp_CCOEFF_T_loop_core.h"
#include "../rcp2_sliding_window_macros.h"
#include "../../../../../core/utils/fragment_arrangment.h"



namespace decx
{
    namespace rcp {
        namespace CPUK {
            static _THREAD_CALL_ void _rcp2_CCOEFF_rN_rect_fixed_uint8_ST(const double* src, float* kernel, const float* I_src, float* dst, 
                const uint2 ker_dims, const uint Wsrc, const uint W_Isrc, const uint Wdst, const ushort reg_WL, const uint _loop);



            static _THREAD_CALL_ void _rcp2_CCOEFF_rN_rect_flex_uint8_ST(const double* src, float* kernel, const float* I_src, float* dst, 
                const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint W_Isrc, const uint Wdst, const ushort reg_WL, const uint _loop);


            // ------------------------------------------------------- norm -------------------------------------------------------

            /*
            * @param _sqrt_k_sum : SUM<i, j>(kernel(i, j) ^ 2)
            */
            static _THREAD_CALL_ void _rcp2_CCOEFF_norm_rN_rect_fixed_uint8_ST(const double* src, float* kernel, const float* I_src, float* dst, 
                const float _sqrt_k_sum, const uint2 ker_dims, const uint Wsrc, const uint W_Isrc, const uint Wdst, const ushort reg_WL, const uint _loop);



            static _THREAD_CALL_ void _rcp2_CCOEFF_norm_rN_rect_flex_uint8_ST(const double* src, float* kernel, const float* I_src, float* dst, 
                const float _sqrt_k_sum, const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint W_Isrc, const uint Wdst, const ushort reg_WL, const uint _loop);
        }
    }
}




_THREAD_CALL_
void decx::rcp::CPUK::_rcp2_CCOEFF_rN_rect_fixed_uint8_ST(const double* __restrict   src, 
                                        float* __restrict   kernel, 
                                        const float* __restrict   I_src,
                                        float* __restrict   dst,
                                        const uint2         ker_dims,
                                        const uint          Wsrc,
                                        const uint          W_Isrc,
                                        const uint          Wdst,
                                        const ushort        reg_WL,
                                        const uint          _loop)
{
    __m256 res_vec8;                                            
    size_t dex_src = 0, dex_dst = 0, dex_Isrc = 0;                            
        
    for (int i = 0; i < _BLOCKED_RCP2_UINT8_H_; ++i) {
        for (int j = 0; j < _BLOCKED_RCP2_UINT8_W_; ++j) {
                    
            res_vec8 = decx::rcp::CPUK::_rcp2_CCOEFF_uint8_loop_in_kernel(
                src + dex_src, kernel, I_src + dex_Isrc, ker_dims, reg_WL, Wsrc, W_Isrc, _loop);
            _mm256_store_ps(dst + dex_dst, res_vec8);           
            ++dex_src;                                       
            dex_dst += 8;                                       
            dex_Isrc += 8;                                       
        }                                                       
        dex_dst += (Wdst << 3) - (_BLOCKED_RCP2_FP32_W_ << 3);  
        dex_src += Wsrc - _BLOCKED_RCP2_FP32_W_;  
        dex_Isrc += (W_Isrc << 3) - (_BLOCKED_RCP2_FP32_W_ << 3);  
    }                                                           
}



_THREAD_CALL_
void decx::rcp::CPUK::_rcp2_CCOEFF_rN_rect_flex_uint8_ST(const double* __restrict   src, 
                                        float* __restrict   kernel, 
                                        const float* __restrict   I_src,
                                        float* __restrict   dst,
                                       const uint2         proc_dim,
                                       const uint2         ker_dims,
                                       const uint          Wsrc,
                                       const uint          W_Isrc,
                                       const uint          Wdst,
                                       const ushort        reg_WL,
                                       const uint          _loop)
{
    __m256 res_vec8;                                    
        size_t dex_src = 0, dex_dst = 0, dex_Isrc = 0;                    
        
    for (int i = 0; i < proc_dim.y; ++i) {
        for (int j = 0; j < proc_dim.x; ++j) {
            res_vec8 = decx::rcp::CPUK::_rcp2_CCOEFF_uint8_loop_in_kernel(
                src + dex_src, kernel, I_src + dex_Isrc, ker_dims, reg_WL, Wsrc, W_Isrc, _loop);
            _mm256_store_ps(dst + dex_dst, res_vec8);   
            ++dex_src;                               
            dex_dst += 8;                               
            dex_Isrc += 8;                               
        }                                               
        dex_dst += (Wdst << 3) - (proc_dim.x << 3);     
        dex_src += Wsrc - proc_dim.x;     
        dex_Isrc += (W_Isrc << 3) - (proc_dim.x << 3);     
    }                                                   
}




_THREAD_CALL_
void decx::rcp::CPUK::_rcp2_CCOEFF_norm_rN_rect_fixed_uint8_ST(const double* __restrict   src, 
                                        float* __restrict   kernel, 
                                        const float* __restrict   I_src,
                                        float* __restrict   dst,
                                        const float         _k_sq_sum,
                                        const uint2         ker_dims,
                                        const uint          Wsrc,
                                        const uint          W_Isrc,
                                        const uint          Wdst,
                                        const ushort        reg_WL,
                                        const uint          _loop)
{
    __m256 res_vec8;                                            
    size_t dex_src = 0, dex_dst = 0, dex_Isrc = 0;                            
        
    for (int i = 0; i < _BLOCKED_RCP2_UINT8_H_; ++i) {
        for (int j = 0; j < _BLOCKED_RCP2_UINT8_W_; ++j) {
                    
            res_vec8 = decx::rcp::CPUK::_rcp2_CCOEFF_NORM_uint8_loop_in_kernel(
                src + dex_src, kernel, I_src + dex_Isrc, _k_sq_sum, ker_dims, reg_WL, Wsrc, W_Isrc, _loop);
            _mm256_store_ps(dst + dex_dst, res_vec8);           
            ++dex_src;                                       
            dex_dst += 8;                                       
            dex_Isrc += 8;                                       
        }                                                       
        dex_dst += (Wdst << 3) - (_BLOCKED_RCP2_FP32_W_ << 3);  
        dex_src += Wsrc - _BLOCKED_RCP2_FP32_W_;  
        dex_Isrc += (W_Isrc << 3) - (_BLOCKED_RCP2_FP32_W_ << 3);  
    }                                                           
}



_THREAD_CALL_
void decx::rcp::CPUK::_rcp2_CCOEFF_norm_rN_rect_flex_uint8_ST(const double* __restrict   src, 
                                        float* __restrict   kernel, 
                                        const float* __restrict   I_src,
                                        float* __restrict   dst,
                                        const float         _k_sq_sum,
                                       const uint2         proc_dim,
                                       const uint2         ker_dims,
                                       const uint          Wsrc,
                                       const uint          W_Isrc,
                                       const uint          Wdst,
                                       const ushort        reg_WL,
                                       const uint          _loop)
{
    __m256 res_vec8;                                    
        size_t dex_src = 0, dex_dst = 0, dex_Isrc = 0;                    
        
    for (int i = 0; i < proc_dim.y; ++i) {
        for (int j = 0; j < proc_dim.x; ++j) {
            res_vec8 = decx::rcp::CPUK::_rcp2_CCOEFF_NORM_uint8_loop_in_kernel(
                src + dex_src, kernel, I_src + dex_Isrc, _k_sq_sum, ker_dims, reg_WL, Wsrc, W_Isrc, _loop);
            _mm256_store_ps(dst + dex_dst, res_vec8);   
            ++dex_src;                               
            dex_dst += 8;                               
            dex_Isrc += 8;                               
        }                                               
        dex_dst += (Wdst << 3) - (proc_dim.x << 3);     
        dex_src += Wsrc - proc_dim.x;     
        dex_Isrc += (W_Isrc << 3) - (proc_dim.x << 3);     
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
            void _rcp2_CCOEFF_uint8_ST(double* src, float* kernel, float* I_src, float* dst, const uint2 proc_dim, const uint2 ker_dims,
                const uint Wsrc, const uint W_Isrc, const uint Wdst, const ushort reg_WL, const uint _loop);


            _THREAD_FUNCTION_
            void _rcp2_CCOEFF_norm_uint8_ST(double* src, float* kernel, float* I_src, float* dst, const float _sqrt_k_sum, const uint2 proc_dim, const uint2 ker_dims,
                const uint Wsrc, const uint W_Isrc, const uint Wdst, const ushort reg_WL, const uint _loop);
        }
    }
}



namespace decx
{
    namespace rcp{
        void _rcp2_CCOEFF_rN_uint8_caller(double* src, float* kernel, float* I_src, float* dst, const uint2 proc_dim, const uint2 ker_dims,
            const uint Wsrc, const uint W_Isrc, const uint Wdst, const ushort reg_WL,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
            const uint _loop);



        void _rcp2_CCOEFF_norm_rN_uint8_caller(double* src, float* kernel, float* I_src, float* dst, const float _sqrt_k_sum, const uint2 proc_dim, const uint2 ker_dims,
            const uint Wsrc, const uint W_Isrc, const uint Wdst, const ushort reg_WL,
            decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr,
            const uint _loop);
    }
}



namespace decx
{
    namespace rcp{
        template <bool _norm>
        static void _rcp2_CCOEFF_uint8_organiser(double* src, float* kernel, float* I_src, float* dst, const float _sqrt_k_sum,
                                  const uint2 proc_dim, const uint2 ker_dims, const uint Wsrc, const uint W_I_src,
                                  const uint Wdst, decx::utils::_thr_1D* t1D, decx::utils::frag_manager* f_mgr);
    }
}


template <bool _norm>
void decx::rcp::_rcp2_CCOEFF_uint8_organiser(double*                       src,
                                    float*                             kernel, 
                                    float*                             I_src, 
                                    float*                             dst, 
                                    const float                        _sqrt_k_sum,
                                    const uint2                        proc_dim, 
                                    const uint2                        ker_dims,
                                    const uint                         Wsrc,
                                    const uint                         W_Isrc,
                                    const uint                         Wdst,
                                    decx::utils::_thr_1D*              t1D,
                                    decx::utils::frag_manager*         f_mgr)
{
    const uint _loop = (ker_dims.x - 1) / 24;
    ushort reg_WL = (ushort)(ker_dims.x - _loop * 24);
    if (_norm) {
        decx::rcp::_rcp2_CCOEFF_norm_rN_uint8_caller(src, kernel, I_src, dst,
            _sqrt_k_sum, proc_dim, ker_dims, Wsrc, W_Isrc, Wdst, reg_WL, t1D, f_mgr, _loop);
    }
    else {
        decx::rcp::_rcp2_CCOEFF_rN_uint8_caller(src, kernel, I_src, dst,
            proc_dim, ker_dims, Wsrc, W_Isrc, Wdst, reg_WL, t1D, f_mgr, _loop);
    }
}




#endif