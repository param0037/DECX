/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _LOW_PASS_CUH_
#define _LOW_PASS_CUH_

#include "../../../core/basic.h"
#include "../../../classes/GPU_Vector.h"
#include "../../../classes/GPU_Matrix.h"
#include "../../../core/configs/config.h"


namespace decx {
    namespace signal {
        namespace GPUK {
            __global__ void
            cu_ideal_LP1D_cpl32(const float4* src, float4* dst, const size_t _proc_len, const size_t real_bound, const size_t cutoff_freq);


            /*
            * @param _proc_dims : ~.x -> width of the processed area; ~.y -> height of the processed area;
            * @param real_bound : ~.x -> x-axis of the processed area; ~.y -> y-axis of the processed area;
            * @param pitch : pitch of the processed area, in float4 (vec2 of datatype of de::CPf);
            */
            __global__ void
            cu_ideal_LP2D_cpl32(const float4* src, float4* dst, const uint2 _proc_dims, const uint2 real_bound, const uint2 cutoff_freq, const uint pitch);
        }
    }
}


namespace de {
    namespace signal {
        namespace cuda {
            _DECX_API_ de::DH LowPass1D_Ideal(de::GPU_Vector& src, de::GPU_Vector& dst, const size_t cutoff_frequency);


            _DECX_API_ de::DH LowPass2D_Ideal(de::GPU_Matrix& src, de::GPU_Matrix& dst, const de::Point2D cutoff_frequency);
        }
    }
}


#endif