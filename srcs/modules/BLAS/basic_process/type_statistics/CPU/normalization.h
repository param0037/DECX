/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _NORMALIZATION_H_
#define _NORMALIZATION_H_


#include "../../../../core/basic.h"
#include "../../../../classes/All_classes.h"
#include "cmp_exec.h"

#ifdef _DECX_BLAS_CPU_
#include "../../../../core/thread_management/thread_pool.h"
namespace decx
{
    namespace bp
    {
        namespace CPUK {
            _THREAD_FUNCTION_ void normalize_scale_v8_fp32(const float* src, float* dst, const double2 min_max, const double2 range, const uint64_t proc_len);

            typedef void (norm_scale_kernel_fp32) (const float*, float*, const double2, const double2, const uint64_t);
        }

        template <typename T_op, typename T_data, uint8_t _align>
        static void norm_caller(T_op* _op, const T_data* src, T_data* dst, const double2 min_max, const double2 range, const uint64_t proc_len);
    }
}


template <typename T_op, typename T_data, uint8_t _align>
static void decx::bp::norm_caller(T_op* _op, const T_data* src, T_data* dst, const double2 min_max, const double2 range,
    const uint64_t proc_len)
{
    decx::utils::_thread_arrange_1D t1D(decx::cpu::_get_permitted_concurrency());
    decx::utils::frag_manager f_mgr;
    decx::utils::frag_manager_gen(&f_mgr, proc_len / _align, t1D.total_thread);
    
    for (uint32_t i = 0; i < t1D.total_thread - 1; ++i) {
        t1D._async_thread[i] =
            decx::cpu::register_task_default( _op,
                src + (i * _align) * f_mgr.frag_len,
                dst + (i * _align) * f_mgr.frag_len, 
                min_max, range, f_mgr.frag_len);
    }
    const uint64_t _L = f_mgr.is_left ? f_mgr.frag_left_over : f_mgr.frag_len;
    t1D._async_thread[t1D.total_thread - 1] =
        decx::cpu::register_task_default( _op,
            src + (f_mgr.frag_len * _align * (t1D.total_thread - 1)),
            dst + (f_mgr.frag_len * _align * (t1D.total_thread - 1)), 
            min_max, range, _L);

    t1D.__sync_all_threads();
}


#endif


namespace decx
{
    namespace cpu {
        _DECX_API_ void scale_raw_API(decx::_Vector* src, decx::_Vector* dst, const double2 range, de::DH* handle);

        _DECX_API_ void scale_raw_API(decx::_Matrix* src, decx::_Matrix* dst, const double2 range, de::DH* handle);
    }
}


namespace de
{
    namespace cpu {
        _DECX_API_ de::DH Scale(de::Vector& src, de::Vector& dst, de::Point2D_d range);


        _DECX_API_ de::DH Scale(de::Matrix& src, de::Matrix& dst, de::Point2D_d range);
    }
}


#endif