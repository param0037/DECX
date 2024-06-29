/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
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