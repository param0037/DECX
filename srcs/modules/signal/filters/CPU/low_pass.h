/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _LOW_PASS_H_
#define _LOW_PASS_H_

#include "../../../core/basic.h"
#include "../../../core/thread_management/thread_pool.h"
#include "../../../core/thread_management/thread_arrange.h"
#include "../../../core/utils/intrinsics_ops.h"
#include "../../../classes/Vector.h"
#include "../../../classes/Matrix.h"
#include "../../../core/utils/fragment_arrangment.h"


namespace decx {
    namespace signal {
        namespace CPUK {
            _THREAD_FUNCTION_ void
            ideal_LP1D_cpl32_ST(const double* src, double* dst, const size_t cutoff_freq, const size_t _proc_len, 
                const size_t real_bound, const size_t global_dex_offset);

            /*
            * @param _proc_dims : ~.x -> width of the processed area; ~.y -> height of the processed area;
            * @param real_bound : ~.x -> x-axis of the processed area; ~.y -> y-axis of the processed area;
            * @param pitch : pitch of the processed area, in float4 (vec2 of datatype of de::CPf);
            * @param global_dex_offset : ~.x -> offset on x-axis; ~.y -> offset on y-axis;
            */
            _THREAD_FUNCTION_ void
            ideal_LP2D_cpl32_ST(const double* src, double* dst, const uint2 _proc_dims, const uint2 real_bound, 
                const uint2 cutoff_freq, const uint pitch, const uint2 global_dex_offset);
        }
    }
}



namespace de {
    namespace signal {
        namespace cpu {
            _DECX_API_ de::DH LowPass1D_Ideal(de::Vector& src, de::Vector& dst, const size_t cutoff_frequency);


            _DECX_API_ de::DH LowPass2D_Ideal(de::Matrix& src, de::Matrix& dst, const de::Point2D cutoff_frequency);
        }
    }
}


#endif