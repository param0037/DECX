/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _EDGE_DET_OPS_H_
#define _EDGE_DET_OPS_H_


#include "../../../core/thread_management/thread_pool.h"
#include "../../../core/utils/intrinsics_ops.h"


namespace decx {
    namespace vis 
    {
        struct __align__(8) _gradient_info {
            float gradient, direction_rad;
        };


        namespace CPUK {
            typedef void (*Canny_operator_kernel)(const uint8_t*, float*, float*, const uint, const uint, const uint2);

            /**
            * @param proc_dims : ~.x -> width of the processed area, in vec8
            * @param Wsrc : pitch of source matrix, in vec8
            * @param Wdst : pitch of destinated matrix, in vec8
            */
            _THREAD_FUNCTION_ void
                Sobel_XY_uint8_T(const uint8_t* src, float* G, float* dir, const uint Wsrc, const uint Wdst,
                    const uint2 proc_dims);


            /**
            * @param proc_dims : ~.x -> width of the processed area, in vec8
            * @param Wsrc : pitch of source matrix, in vec8
            * @param Wdst : pitch of destinated matrix, in vec8
            */
            _THREAD_FUNCTION_ void
                Sobel_XY_uint8(const uint8_t* src, float* G, float* dir, const uint Wsrc, const uint Wdst,
                    const uint2 proc_dims);

            /**
            * @param proc_dims : ~.x -> width of the processed area, in vec8
            * @param Wsrc : pitch of source matrix, in vec8
            * @param Wdst : pitch of destinated matrix, in vec8
            */
            _THREAD_FUNCTION_ void
                Sobel_XY_uint8_B(const uint8_t* src, float* G, float* dir, const uint Wsrc, const uint Wdst,
                    const uint2 proc_dims);

            /**
            * @param proc_dims : ~.x -> width of the processed area, in vec8
            * @param Wsrc : pitch of source matrix, in vec8
            * @param Wdst : pitch of destinated matrix, in vec8
            */
            _THREAD_FUNCTION_ void
                Sobel_XY_uint8_TB(const uint8_t* src, float* G, float* dir, const uint Wsrc, const uint Wdst,
                    const uint2 proc_dims);
        }

        namespace CPUK {
            /**
            * @param proc_dims : ~.x -> width of the processed area, in vec8
            * @param Wsrc : pitch of source matrix, in vec8
            * @param Wdst : pitch of destinated matrix, in vec8
            */
            _THREAD_FUNCTION_ void
                Scharr_XY_uint8_T(const uint8_t* src, float* G, float* dir, const uint Wsrc, const uint Wdst,
                    const uint2 proc_dims);


            /**
            * @param proc_dims : ~.x -> width of the processed area, in vec8
            * @param Wsrc : pitch of source matrix, in vec8
            * @param Wdst : pitch of destinated matrix, in vec8
            */
            _THREAD_FUNCTION_ void
                Scharr_XY_uint8(const uint8_t* src, float* G, float* dir, const uint Wsrc, const uint Wdst,
                    const uint2 proc_dims);

            /**
            * @param proc_dims : ~.x -> width of the processed area, in vec8
            * @param Wsrc : pitch of source matrix, in vec8
            * @param Wdst : pitch of destinated matrix, in vec8
            */
            _THREAD_FUNCTION_ void
                Scharr_XY_uint8_B(const uint8_t* src, float* G, float* dir, const uint Wsrc, const uint Wdst,
                    const uint2 proc_dims);

            /**
            * @param proc_dims : ~.x -> width of the processed area, in vec8
            * @param Wsrc : pitch of source matrix, in vec8
            * @param Wdst : pitch of destinated matrix, in vec8
            */
            _THREAD_FUNCTION_ void
                Scharr_XY_uint8_TB(const uint8_t* src, float* G, float* dir, const uint Wsrc, const uint Wdst,
                    const uint2 proc_dims);
        }

        namespace CPUK {
            /**
            * This function is per-element
            * @param _thres : ~.x -> lower threshold; ~.y -> higher threshold
            */
            _THREAD_FUNCTION_ void
            Edge_Detector_Post_processing(const float* G_info_map, const float* dir_info_map, uint8_t* dst, const uint Wsrc, const uint Wdst,
                const uint2 proc_dims, const float2 _thres);
        }
    }
}


#endif
