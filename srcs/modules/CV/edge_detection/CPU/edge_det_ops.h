/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _EDGE_DET_OPS_H_
#define _EDGE_DET_OPS_H_


#include "../../../core/thread_management/thread_pool.h"
#include "../../../core/utils/intrinsics_ops.h"


namespace decx {
    namespace vis 
    {
        namespace CPUK {
            typedef void(*_canny_operator_ptr)(const uint8_t*, float*, float*, const uint32_t, const uint32_t, const uint,
                const uint2);

            /**
            * @param proc_dims : ~.x -> width of the processed area, in vec8
            * @param Wsrc : pitch of source matrix, in vec8
            * @param Wdst : pitch of destinated matrix, in vec8
            */
            _THREAD_FUNCTION_ void
                Sobel_XY_uint8(const uint8_t* src, float* G, float* dir, const uint32_t WG, const uint32_t WD, const uint Wdst,
                    const uint2 proc_dims);
        }

        namespace CPUK {
            /**
            * @param proc_dims : ~.x -> width of the processed area, in vec8
            * @param Wsrc : pitch of source matrix, in vec8
            * @param Wdst : pitch of destinated matrix, in vec8
            */
            _THREAD_FUNCTION_ void
                Scharr_XY_uint8(const uint8_t* src, float* G, float* dir, const uint32_t WG, const uint32_t WD, const uint Wdst,
                    const uint2 proc_dims);
        }



        namespace CPUK {
            /**
            * This function is per-element
            * @param _thres : ~.x -> lower threshold; ~.y -> higher threshold
            */
            _THREAD_FUNCTION_ void
                _Edge_Detector_Post_processing(const float* G_info_map, const float* dir_info_map,
                    float* _cache, uint64_t* dst,
                    const uint WG, const uint WD, const uint Wdst, const uint2 proc_dims, const float2 _thres);
        }
    }
}


#endif