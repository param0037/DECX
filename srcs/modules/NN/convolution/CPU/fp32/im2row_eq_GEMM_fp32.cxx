/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "im2row_eq_GEMM_fp32.h"



namespace decx {
    namespace conv_I2R {
        namespace CPUK {
            _THREAD_CALL_ static void
                _im2row_eq_GEMM_fp32_fixed(const float* I2C_buf, const float* kernel, float* dst, const uint WI2C,
                    const uint dst_dpitch, const uint kernel_TN);


            _THREAD_CALL_ static void
                _im2row_eq_GEMM_fp32_flex(const float* I2C_buf, const float* kernel, float* dst, const uint WI2C,
                    const uint proc_WH, const uint dst_dpitch, const uint kernel_TN);
        }
    }
}


_THREAD_CALL_ static void
decx::conv_I2R::CPUK::_im2row_eq_GEMM_fp32_fixed(const float* __restrict       I2C_buf,
                                             const float* __restrict       kernel, 
                                             float*                        dst, 
                                             const uint                    WI2C,
                                             const uint                    dst_dpitch,     // dpitch of dst (in float)
                                             const uint                    kernel_TN)      // # of tensor in kernel
{
    size_t dex_src = 0, dex_kernel = 0, dex_dst = 0;

    __m256 recv_src, recv_ker, _accu;

    for (int i = 0; i < _im2row_eqMM_frag_size_; ++i) {
        dex_dst = i * dst_dpitch;
        for (int j = 0; j < kernel_TN; ++j) {
            _accu = _mm256_set1_ps(0);
            dex_src = i * WI2C;
            dex_kernel = j * WI2C;
            
            for (int k = 0; k < WI2C / 8; ++k) {
                recv_src = _mm256_load_ps(I2C_buf + dex_src);
                recv_ker = _mm256_load_ps(kernel + dex_kernel);

                _accu = _mm256_fmadd_ps(recv_src, recv_ker, _accu);
                dex_src += 8;
                dex_kernel += 8;
            }
            dst[dex_dst] = decx::utils::simd::_mm256_h_sum(_accu);
            ++dex_dst;
        }
    }
}



_THREAD_CALL_ void
decx::conv_I2R::CPUK::_im2row_eq_GEMM_fp32_flex(const float* __restrict       I2C_buf, 
                                            const float* __restrict       kernel, 
                                            float*                        dst, 
                                            const uint                    WI2C,
                                            const uint                    proc_WH,        // how many rows of I2C_buf to be processed
                                            const uint                    dst_dpitch,     // dpitch of dst (in float)
                                            const uint                    kernel_TN)      // # of tensor in kernel
{
    size_t dex_src = 0, dex_kernel = 0, dex_dst = 0;

    __m256 recv_src, recv_ker, _accu;

    for (int i = 0; i < proc_WH; ++i) {
        dex_dst = i * dst_dpitch;
        for (int j = 0; j < kernel_TN; ++j) {
            _accu = _mm256_set1_ps(0);
            dex_src = i * WI2C;
            dex_kernel = j * WI2C;
            
            for (int k = 0; k < WI2C / 8; ++k) {
                recv_src = _mm256_load_ps(I2C_buf + dex_src);
                recv_ker = _mm256_load_ps(kernel + dex_kernel);

                _accu = _mm256_fmadd_ps(recv_src, recv_ker, _accu);
                dex_src += 8;
                dex_kernel += 8;
            }
            dst[dex_dst] = decx::utils::simd::_mm256_h_sum(_accu);
            ++dex_dst;
        }
    }
}

#if 1

_THREAD_FUNCTION_ void
decx::conv_I2R::CPUK::_im2row_eq_GEMM_fp32_ST(const float* __restrict       I2C_buf, 
                                          const float* __restrict       kernel, 
                                          float*                        dst, 
                                          const uint                    WI2C,
                                          const size_t                  proc_WH,        // how many rows of I2C_buf to be processed
                                          const uint                    dst_dpitch,     // dpitch of dst (in float)
                                          const uint                    kernel_TN)      // # of tensor in kernel
{
    size_t dex_src = 0, dex_kernel = 0, dex_dst = 0;

    __m256 recv_src, recv_ker, _accu;

    for (int i = 0; i < proc_WH; ++i) {
        dex_dst = i * dst_dpitch;
        for (int j = 0; j < kernel_TN; ++j) {
            _accu = _mm256_set1_ps(0);
            dex_src = i * WI2C;
            dex_kernel = j * WI2C;
            
            for (int k = 0; k < WI2C / 8; ++k) {
                recv_src = _mm256_load_ps(I2C_buf + dex_src);
                recv_ker = _mm256_load_ps(kernel + dex_kernel);

                _accu = _mm256_fmadd_ps(recv_src, recv_ker, _accu);
                dex_src += 8;
                dex_kernel += 8;
            }
            dst[dex_dst] = decx::utils::simd::_mm256_h_sum(_accu);
            ++dex_dst;
        }
    }
}

#else
_THREAD_FUNCTION_ void
decx::conv_I2R::CPUK::_im2row_eq_GEMM_fp32_ST(const float* __restrict       I2C_buf, 
                                          const float* __restrict       kernel, 
                                          float*                        dst, 
                                          const uint                    WI2C,
                                          const size_t                  proc_WH,        // how many rows of I2C_buf to be processed
                                          const uint                    dst_dpitch,     // dpitch of dst (in float)
                                          const uint                    kernel_TN)      // # of tensor in kernel
{
    size_t dex_src = 0, dex_dst = 0;

    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen_from_fragLen(&f_mgr, proc_WH, _im2row_eqMM_frag_size_);

    for (int i = 0; i < f_mgr.frag_num - 1; ++i) {
        _im2row_eq_GEMM_fp32_fixed(I2C_buf + dex_src, kernel, dst + dex_dst,
            WI2C, dst_dpitch, kernel_TN);

        dex_src += _im2row_eqMM_frag_size_ * WI2C;
        dex_dst += _im2row_eqMM_frag_size_ * dst_dpitch;
    }
    const uint _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    _im2row_eq_GEMM_fp32_flex(I2C_buf + dex_src, kernel, dst + dex_dst,
        WI2C, _L, dst_dpitch, kernel_TN);
}

#endif


//_THREAD_FUNCTION_ void
//decx::conv::CPUK::_im2row_eq_GEMM_fp32_ST(const float* __restrict       I2C_buf, 
//                                          const float* __restrict       kernel, 
//                                          float*                        dst, 
//                                          const uint                    WI2C,
//                                          const size_t                  proc_WH,        // how many rows of I2C_buf to be processed
//                                          const uint                    dst_dpitch,     // dpitch of dst (in float)
//                                          const uint                    kernel_TN)      // # of tensor in kernel
//{
//    size_t dex_src = 0, dex_kernel = 0, dex_dst = 0;
//
//    __m256 recv_src, recv_ker, _accu[4];
//
//    for (int i = 0; i < proc_WH; ++i) {
//        dex_dst = i * dst_dpitch;
//        for (int j = 0; j < kernel_TN; j += 4) {
//            _accu[0] = _mm256_set1_ps(0);           _accu[1] = _mm256_set1_ps(0);
//            _accu[2] = _mm256_set1_ps(0);           _accu[3] = _mm256_set1_ps(0);
//            dex_src = i * WI2C;
//            dex_kernel = j * WI2C;
//            
//            for (int k = 0; k < WI2C / 8; ++k) {
//                recv_src = _mm256_load_ps(I2C_buf + dex_src);
//
//                recv_ker = _mm256_load_ps(kernel + dex_kernel);
//                _accu[0] = _mm256_fmadd_ps(recv_src, recv_ker, _accu[0]);
//
//                recv_ker = _mm256_load_ps(kernel + dex_kernel + WI2C);
//                _accu[1] = _mm256_fmadd_ps(recv_src, recv_ker, _accu[1]);
//
//                recv_ker = _mm256_load_ps(kernel + dex_kernel + WI2C * 2);
//                _accu[2] = _mm256_fmadd_ps(recv_src, recv_ker, _accu[2]);
//
//                recv_ker = _mm256_load_ps(kernel + dex_kernel + WI2C * 3);
//                _accu[3] = _mm256_fmadd_ps(recv_src, recv_ker, _accu[3]);
//
//                dex_src += 8;
//                dex_kernel += 8;
//            }
//            //dst[dex_dst] = decx::utils::simd::_mm256_h_sum(_accu);
//            _mm_store_ps(dst + dex_dst, _mm_setr_ps(decx::utils::simd::_mm256_h_sum(_accu[0]),
//                decx::utils::simd::_mm256_h_sum(_accu[1]),
//                decx::utils::simd::_mm256_h_sum(_accu[2]),
//                decx::utils::simd::_mm256_h_sum(_accu[3])));
//            dex_dst += 4;
//        }
//
//
//    }
//}



_THREAD_FUNCTION_ void
decx::conv_I2R::_im2row_eq_GEMM_caller_fp32(const float* __restrict       I2C_buf,
                                            const float* __restrict       kernel, 
                                            float*                        dst, 
                                            const uint                    WI2C,
                                            const size_t                  proc_WH,        // how many rows of I2C_buf to be processed
                                            const uint                    dst_dpitch,     // dpitch of dst (in float)
                                            const uint                    kernel_TN,
                                            decx::utils::_thread_arrange_1D* t1D)         // # of tensor in kernel
{
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, proc_WH, t1D->total_thread);

    const float* loc_I2C_buf = I2C_buf;
    float* loc_dst = dst;

    const size_t frag_src = (size_t)f_mgr.frag_len * (size_t)WI2C,
        frag_dst = (size_t)f_mgr.frag_len * (size_t)dst_dpitch;

    for (int i = 0; i < f_mgr.frag_num - 1; ++i) {
        t1D->_async_thread[i] = decx::cpu::register_task_default(
            decx::conv_I2R::CPUK::_im2row_eq_GEMM_fp32_ST, loc_I2C_buf, kernel, loc_dst, WI2C,
            f_mgr.frag_len, dst_dpitch, kernel_TN);

        loc_I2C_buf += frag_src;
        loc_dst += frag_dst;
    }

    const uint _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;

    t1D->_async_thread[f_mgr.frag_num - 1] = decx::cpu::register_task_default(
        decx::conv_I2R::CPUK::_im2row_eq_GEMM_fp32_ST, loc_I2C_buf, kernel, loc_dst, WI2C,
        _L, dst_dpitch, kernel_TN);

    t1D->__sync_all_threads();
}