/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#include "../clip_range.h"
#include "../../operators_frame_exec.h"
#include "../../../core/configs/config.h"
#include "Matrix_operators.h"


_DECX_API_ de::DH de::cpu::Clip(de::Matrix& src, de::Matrix& dst, const de::Point2D_d range)
{
    de::DH handle;
    if (!decx::cpu::_is_CPU_init()) {
        Print_Error_Message(4, CPU_NOT_INIT);
        decx::err::CPU_Not_init(&handle);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    decx::utils::_thread_arrange_1D* t1D = NULL;
    decx::utils::frag_manager f_mgr;

    switch (_src->Type())
    {
    case decx::_DATA_TYPES_FLAGS_::_FP32_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, _src->Pitch() * _src->Height() / 8, t1D->total_thread);

        decx::calc::operators_caller_c<decx::calc::_fp32_clip_ops, float, 8, float2>(decx::calc::CPUK::_clip_range_fp32,
            (float*)_src->Mat.ptr, make_float2(range.x, range.y), (float*)_dst->Mat.ptr, t1D, &f_mgr);
        break;

    case decx::_DATA_TYPES_FLAGS_::_FP64_:
        // configure the process domain for each thread
        t1D = new decx::utils::_thread_arrange_1D(decx::cpu::_get_permitted_concurrency());

        decx::utils::frag_manager_gen(&f_mgr, _src->Pitch() * _src->Height() / 4, t1D->total_thread);

        decx::calc::operators_caller_c<decx::calc::_fp64_clip_ops, double, 4, double2>(decx::calc::CPUK::_clip_range_fp64,
            (double*)_src->Mat.ptr, make_double2(range.x, range.y), (double*)_dst->Mat.ptr, t1D, &f_mgr);
        break;

    default:
        break;
    }

    if (t1D != NULL) {
        delete t1D;
    }

    decx::err::Success(&handle);
    return handle;
}