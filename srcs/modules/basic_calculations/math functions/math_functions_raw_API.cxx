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


#include "math_functions_raw_API.h"


template <bool _print>
void decx::cpu::Log10_Raw_API(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle)
{
    decx::utils::_thread_arrange_1D* t1D = NULL;
    decx::utils::frag_manager f_mgr;

    switch (type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());
        
        decx::utils::frag_manager_gen(&f_mgr, len / 8, t1D->total_thread);

        decx::calc::operators_caller<decx::calc::_fp32_single_ops, float, 8>(decx::calc::CPUK::log10_fvec8_ST,
            (float*)src, (float*)dst, t1D, &f_mgr);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, len / 4, t1D->total_thread);

        decx::calc::operators_caller<decx::calc::_fp64_single_ops, double, 4>(decx::calc::CPUK::log10_dvec4_ST,
            (double*)src, (double*)dst, t1D, &f_mgr);
        break;

    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            UNSUPPORTED_TYPE);
        break;
    }
}

template _DECX_API_ void decx::cpu::Log10_Raw_API<true>(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);
template _DECX_API_ void decx::cpu::Log10_Raw_API<false>(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);


template <bool _print>
void decx::cpu::Log2_Raw_API(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle)
{
    decx::utils::_thread_arrange_1D* t1D = NULL;
    decx::utils::frag_manager f_mgr;

    switch (type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, len / 8, t1D->total_thread);

        decx::calc::operators_caller<decx::calc::_fp32_single_ops, float, 8>(decx::calc::CPUK::log2_fvec8_ST,
            (float*)src, (float*)dst, t1D, &f_mgr);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, len / 4, t1D->total_thread);

        decx::calc::operators_caller<decx::calc::_fp64_single_ops, double, 4>(decx::calc::CPUK::log2_dvec4_ST,
            (double*)src, (double*)dst, t1D, &f_mgr);
        break;

    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            UNSUPPORTED_TYPE);
        break;
    }
}

template _DECX_API_ void decx::cpu::Log2_Raw_API<true>(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);
template _DECX_API_ void decx::cpu::Log2_Raw_API<false>(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);



template <bool _print>
void decx::cpu::Exp_Raw_API(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle)
{
    decx::utils::_thread_arrange_1D* t1D = NULL;
    decx::utils::frag_manager f_mgr;

    switch (type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, len / 8, t1D->total_thread);

        decx::calc::operators_caller<decx::calc::_fp32_single_ops, float, 8>(decx::calc::CPUK::exp_fvec8_ST,
            (float*)src, (float*)dst, t1D, &f_mgr);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, len / 4, t1D->total_thread);

        decx::calc::operators_caller<decx::calc::_fp64_single_ops, double, 4>(decx::calc::CPUK::exp_dvec4_ST,
            (double*)src, (double*)dst, t1D, &f_mgr);
        break;

    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            UNSUPPORTED_TYPE);
        break;
    }
}

template _DECX_API_ void decx::cpu::Exp_Raw_API<true>(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);
template _DECX_API_ void decx::cpu::Exp_Raw_API<false>(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);


template <bool _print>
void decx::cpu::Sin_Raw_API(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle)
{
    decx::utils::_thread_arrange_1D* t1D = NULL;
    decx::utils::frag_manager f_mgr;

    switch (type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, len / 8, t1D->total_thread);

        decx::calc::operators_caller<decx::calc::_fp32_single_ops, float, 8>(decx::calc::CPUK::sin_fvec8_ST,
            (float*)src, (float*)dst, t1D, &f_mgr);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, len / 4, t1D->total_thread);

        decx::calc::operators_caller<decx::calc::_fp64_single_ops, double, 4>(decx::calc::CPUK::sin_dvec4_ST,
            (double*)src, (double*)dst, t1D, &f_mgr);
        break;

    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            UNSUPPORTED_TYPE);
        break;
    }
}

template _DECX_API_ void decx::cpu::Sin_Raw_API<true>(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);
template _DECX_API_ void decx::cpu::Sin_Raw_API<false>(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);


template <bool _print>
void decx::cpu::Cos_Raw_API(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle)
{
    decx::utils::_thread_arrange_1D* t1D = NULL;
    decx::utils::frag_manager f_mgr;

    switch (type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, len / 8, t1D->total_thread);

        decx::calc::operators_caller<decx::calc::_fp32_single_ops, float, 8>(decx::calc::CPUK::cos_fvec8_ST,
            (float*)src, (float*)dst, t1D, &f_mgr);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, len / 4, t1D->total_thread);

        decx::calc::operators_caller<decx::calc::_fp64_single_ops, double, 4>(decx::calc::CPUK::cos_dvec4_ST,
            (double*)src, (double*)dst, t1D, &f_mgr);
        break;

    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            UNSUPPORTED_TYPE);
        break;
    }
}

template _DECX_API_ void decx::cpu::Cos_Raw_API<true>(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);
template _DECX_API_ void decx::cpu::Cos_Raw_API<false>(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);


template <bool _print>
void decx::cpu::Tan_Raw_API(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle)
{
    decx::utils::_thread_arrange_1D* t1D = NULL;
    decx::utils::frag_manager f_mgr;

    switch (type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, len / 8, t1D->total_thread);

        decx::calc::operators_caller<decx::calc::_fp32_single_ops, float, 8>(decx::calc::CPUK::tan_fvec8_ST,
            (float*)src, (float*)dst, t1D, &f_mgr);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, len / 4, t1D->total_thread);

        decx::calc::operators_caller<decx::calc::_fp64_single_ops, double, 4>(decx::calc::CPUK::tan_dvec4_ST,
            (double*)src, (double*)dst, t1D, &f_mgr);
        break;

    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            UNSUPPORTED_TYPE);
        break;
    }
}

template _DECX_API_ void decx::cpu::Tan_Raw_API<true>(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);
template _DECX_API_ void decx::cpu::Tan_Raw_API<false>(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);


template <bool _print>
void decx::cpu::Asin_Raw_API(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle)
{
    decx::utils::_thread_arrange_1D* t1D = NULL;
    decx::utils::frag_manager f_mgr;

    switch (type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, len / 8, t1D->total_thread);

        decx::calc::operators_caller<decx::calc::_fp32_single_ops, float, 8>(decx::calc::CPUK::asin_fvec8_ST,
            (float*)src, (float*)dst, t1D, &f_mgr);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, len / 4, t1D->total_thread);

        decx::calc::operators_caller<decx::calc::_fp64_single_ops, double, 4>(decx::calc::CPUK::asin_dvec4_ST,
            (double*)src, (double*)dst, t1D, &f_mgr);
        break;

    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            UNSUPPORTED_TYPE);
        break;
    }
}

template _DECX_API_ void decx::cpu::Asin_Raw_API<true>(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);
template _DECX_API_ void decx::cpu::Asin_Raw_API<false>(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);



template <bool _print>
void decx::cpu::Acos_Raw_API(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle)
{
    decx::utils::_thread_arrange_1D* t1D = NULL;
    decx::utils::frag_manager f_mgr;

    switch (type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, len / 8, t1D->total_thread);

        decx::calc::operators_caller<decx::calc::_fp32_single_ops, float, 8>(decx::calc::CPUK::acos_fvec8_ST,
            (float*)src, (float*)dst, t1D, &f_mgr);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, len / 4, t1D->total_thread);

        decx::calc::operators_caller<decx::calc::_fp64_single_ops, double, 4>(decx::calc::CPUK::acos_dvec4_ST,
            (double*)src, (double*)dst, t1D, &f_mgr);
        break;

    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            UNSUPPORTED_TYPE);
        break;
    }
}

template _DECX_API_ void decx::cpu::Acos_Raw_API<true>(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);
template _DECX_API_ void decx::cpu::Acos_Raw_API<false>(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);


template <bool _print>
void decx::cpu::Atan_Raw_API(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle)
{
    decx::utils::_thread_arrange_1D* t1D = NULL;
    decx::utils::frag_manager f_mgr;

    switch (type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, len / 8, t1D->total_thread);

        decx::calc::operators_caller<decx::calc::_fp32_single_ops, float, 8>(decx::calc::CPUK::atan_fvec8_ST,
            (float*)src, (float*)dst, t1D, &f_mgr);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, len / 4, t1D->total_thread);

        decx::calc::operators_caller<decx::calc::_fp64_single_ops, double, 4>(decx::calc::CPUK::atan_dvec4_ST,
            (double*)src, (double*)dst, t1D, &f_mgr);
        break;

    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            UNSUPPORTED_TYPE);
        break;
    }
}

template _DECX_API_ void decx::cpu::Atan_Raw_API<true>(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);
template _DECX_API_ void decx::cpu::Atan_Raw_API<false>(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);


template <bool _print>
void decx::cpu::Sqrt_Raw_API(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle)
{
    decx::utils::_thread_arrange_1D* t1D = NULL;
    decx::utils::frag_manager f_mgr;

    switch (type)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, len / 8, t1D->total_thread);

        decx::calc::operators_caller<decx::calc::_fp32_single_ops, float, 8>(decx::calc::CPUK::sqrt_fvec8_ST,
            (float*)src, (float*)dst, t1D, &f_mgr);
        break;

    case de::_DATA_TYPES_FLAGS_::_FP64_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, len / 4, t1D->total_thread);

        decx::calc::operators_caller<decx::calc::_fp64_single_ops, double, 4>(decx::calc::CPUK::sqrt_dvec4_ST,
            (double*)src, (double*)dst, t1D, &f_mgr);
        break;

    default:
        decx::err::handle_error_info_modify(handle, decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE,
            UNSUPPORTED_TYPE);
        break;
    }
}

template _DECX_API_ void decx::cpu::Sqrt_Raw_API<true>(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);
template _DECX_API_ void decx::cpu::Sqrt_Raw_API<false>(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);