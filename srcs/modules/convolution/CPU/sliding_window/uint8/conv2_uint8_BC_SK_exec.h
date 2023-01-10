/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _CONV2_UINT8_BC_SK_EXEC_H_
#define _CONV2_UINT8_BC_SK_EXEC_H_

#include "conv2_uint8_exec.h"
#include "../../../../core/thread_management/thread_pool.h"
#include "../../../../core/thread_management/thread_arrange.h"
#include "../../../../core/utils/array_ptr_info.h"
#include "../../../../core/utils/fragment_arrangment.h"
#include "../../../../classes/classes_util.h"


#if !defined(_fmgr)
#define _fmgr decx::utils::frag_manager
#endif


namespace decx
{
    namespace conv {
        namespace CPUK {
            _THREAD_CALL_
                void _conv2_rN_uint8_ST_unconfigured(double* src, uint8_t* kernel, float* dst, const uint2 proc_dim, const uint2 ker_dims,
                    const uint Wsrc, const uint Wdst, const ushort reg_WL, const _fmgr* f_mgrH, const _fmgr* f_mgrW, const uint _loop);
        }
    }
}



namespace decx
{
    namespace conv {
        namespace CPUK {
            _THREAD_CALL_ void _conv2_rN_BC_SK_uint8_ST_top(float* __restrict src, float* __restrict tmp_src, float* __restrict kernel,
                float* __restrict dst, const uint2 proc_dim, const decx::_C2_MK32* conv2_mk_props);


            _THREAD_CALL_ void _conv2_rN_BC_SK_uint8_ST_mid(float* __restrict src, float* __restrict tmp_src, float* __restrict kernel,
                float* __restrict dst, const uint2 proc_dim, const decx::_C2_MK32* conv2_mk_props);


            _THREAD_CALL_ void _conv2_rN_BC_SK_uint8_ST_bottom(float* __restrict src, float* __restrict tmp_src, float* __restrict kernel,
                float* __restrict dst, const uint2 proc_dim, const decx::_C2_MK32* conv2_mk_props);
        }
    }
}


#endif