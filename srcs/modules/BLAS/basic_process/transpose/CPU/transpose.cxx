/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "transpose.h"

namespace decx
{
    namespace blas {
        static void Transpose_4b(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle);
        static void Transpose_1b(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle);
        static void Transpose_8b(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle);
        static void Transpose_16b(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle);
    }
}


static void decx::blas::Transpose_4b(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle)
{
    if (decx::blas::g_cpu_transpose_4b_config._res_ptr == NULL) {
        decx::blas::g_cpu_transpose_4b_config.RegisterResource(new decx::blas::_cpu_transpose_config,
            5, decx::blas::_cpu_transpose_config::release);
    }

    decx::blas::g_cpu_transpose_4b_config.lock();

    const uint32_t _conc = decx::cpu::_get_permitted_concurrency();

    auto* _planner = decx::blas::g_cpu_transpose_4b_config.get_resource_raw_ptr < decx::blas::_cpu_transpose_config>();
    if (_planner->changed(4, _conc, make_uint2(src->Width(), src->Height()))) {
        _planner->config(4, decx::cpu::_get_permitted_concurrency(), make_uint2(src->Width(), src->Height()), handle);
        Check_Runtime_Error(handle);
    }

    decx::utils::_thread_arrange_1D t1D(_conc);
    _planner->transpose_4b_caller((float*)src->Mat.ptr, (float*)dst->Mat.ptr, src->Pitch(), dst->Pitch(), &t1D);

    decx::blas::g_cpu_transpose_4b_config.unlock();
}



static void decx::blas::Transpose_8b(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle)
{
    if (decx::blas::g_cpu_transpose_8b_config._res_ptr == NULL) {
        decx::blas::g_cpu_transpose_8b_config.RegisterResource(new decx::blas::_cpu_transpose_config,
            5, decx::blas::_cpu_transpose_config::release);
    }

    decx::blas::g_cpu_transpose_8b_config.lock();

    const uint32_t _conc = decx::cpu::_get_permitted_concurrency();

    auto* _planner = decx::blas::g_cpu_transpose_8b_config.get_resource_raw_ptr<decx::blas::_cpu_transpose_config>();
    if (_planner->changed(8, _conc, make_uint2(src->Width(), src->Height()))) {
        _planner->config(8, decx::cpu::_get_permitted_concurrency(), make_uint2(src->Width(), src->Height()), handle);
        Check_Runtime_Error(handle);
    }

    decx::utils::_thread_arrange_1D t1D(_conc);
    _planner->transpose_8b_caller((double*)src->Mat.ptr, (double*)dst->Mat.ptr, src->Pitch(), dst->Pitch(), &t1D);

    decx::blas::g_cpu_transpose_8b_config.unlock();
}


static void decx::blas::Transpose_1b(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle)
{
    if (decx::blas::g_cpu_transpose_1b_config._res_ptr == NULL) {
        decx::blas::g_cpu_transpose_1b_config.RegisterResource(new decx::blas::_cpu_transpose_config,
            5, decx::blas::_cpu_transpose_config::release);
    }
    
    decx::blas::g_cpu_transpose_1b_config.lock();

    const uint32_t _conc = decx::cpu::_get_permitted_concurrency();

    auto* _planner = decx::blas::g_cpu_transpose_1b_config.get_resource_raw_ptr < decx::blas::_cpu_transpose_config>();
    if (_planner->changed(1, _conc, make_uint2(src->Width(), src->Height()))) {
        _planner->config(1, _conc, make_uint2(src->Width(), src->Height()), handle);
        Check_Runtime_Error(handle);
    }

    decx::utils::_thread_arrange_1D t1D(_conc);
    _planner->transpose_1b_caller((uint64_t*)src->Mat.ptr, (uint64_t*)dst->Mat.ptr, 
        src->Pitch() / 8, dst->Pitch() / 8, &t1D);

    decx::blas::g_cpu_transpose_1b_config.unlock();
}


static void decx::blas::Transpose_16b(decx::_Matrix* src, decx::_Matrix* dst, de::DH* handle)
{
    if (decx::blas::g_cpu_transpose_16b_config._res_ptr == NULL) {
        decx::blas::g_cpu_transpose_16b_config.RegisterResource(new decx::blas::_cpu_transpose_config,
            5, decx::blas::_cpu_transpose_config::release);
    }

    decx::blas::g_cpu_transpose_16b_config.lock();

    const uint32_t _conc = decx::cpu::_get_permitted_concurrency();

    auto* _planner = decx::blas::g_cpu_transpose_16b_config.get_resource_raw_ptr<decx::blas::_cpu_transpose_config>();
    if (_planner->changed(16, _conc, make_uint2(src->Width(), src->Height()))) {
        _planner->config(16, decx::cpu::_get_permitted_concurrency(), make_uint2(src->Width(), src->Height()), handle);
        Check_Runtime_Error(handle);
    }

    decx::utils::_thread_arrange_1D t1D(_conc);
    _planner->transpose_16b_caller((double*)src->Mat.ptr, (double*)dst->Mat.ptr, src->Pitch(), dst->Pitch(), &t1D);

    decx::blas::g_cpu_transpose_16b_config.unlock();
}


_DECX_API_ void de::blas::cpu::Transpose(de::Matrix& src, de::Matrix& dst)
{
    de::ResetLastError();

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    switch (_src->get_layout()._single_element_size)
    {
    case 1:
        decx::blas::Transpose_1b(_src, _dst, de::GetLastError());
        break;
    case 4:
        decx::blas::Transpose_4b(_src, _dst, de::GetLastError());
        break;
    case 8:
        decx::blas::Transpose_8b(_src, _dst, de::GetLastError());
        break;
    case 16:
        decx::blas::Transpose_16b(_src, _dst, de::GetLastError());
        break;
    default:
        break;
    }
}