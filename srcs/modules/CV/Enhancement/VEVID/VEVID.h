/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _VEVID_H_
#define _VEVID_H_

#include "../../../classes/Matrix.h"
#include "../../../core/thread_management/thread_pool.h"
#include "../../../core/thread_management/thread_arrange.h"
#include "../../../core/utils/intrinsics_ops.h"
#include "../../../core/utils/fragment_arrangment.h"

#include "../../../classes/Matrix.h"


namespace decx{
namespace vis{
    namespace CPUK
    {
        _THREAD_FUNCTION_ void _VEVID_u8_kernel(const double* __restrict src, float* __restrict dst, 
            const uint32_t pitchsrc_v8, const uint32_t pitchdst_v1, const uint32_t proc_H, const float _phase_gain,
            const float _original_gain);
    }

    void VEVID_u8_caller(const double* src, float* dst, const uint32_t pitchsrc_v8, const uint32_t pitchdst_v1, const uint32_t proc_H, const float _phase_gain,
        const float _original_gain);
}
}


namespace de
{
    namespace vis {
        namespace cpu {
            _DECX_API_ de::DH VEVID_gray(de::Matrix& src, de::Matrix& dst, const float _phase_gain,
                const float _original_gain);
        }
    }
}


#endif