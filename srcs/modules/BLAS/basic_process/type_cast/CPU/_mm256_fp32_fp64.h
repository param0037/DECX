/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _MM256_FP32_FP64_H_
#define _MM256_FP32_FP64_H_

#include "../../../../core/thread_management/thread_pool.h"
#include "../../../../core/thread_management/thread_arrange.h"
#include "../../../../core/utils/fragment_arrangment.h"
#include "../../../../core/utils/intrinsics_ops.h"
#include "../../../../core/configs/config.h"
#include "../type_cast_methods.h"


namespace decx
{
    namespace type_cast {
        namespace CPUK {
            /**
            * @param src : fp32 input pointer
            * @param dst : fp64 output pointer
            * @param proc_num : The number of elements to be processed (grouped in 4 per group)(vec4)
            */
            _THREAD_FUNCTION_ void
            _v256_cvtps_pd1D(const float* src, double* dst, const size_t proc_num);



            /**
            * @param src : fp64 input pointer
            * @param dst : fp32 output pointer
            * @param proc_num : The number of elements to be processed (grouped in 4 per group)(vec4)
            */
            _THREAD_FUNCTION_ void
            _v256_cvtpd_ps1D(const double* src, float* dst, const size_t proc_num);


            /**
            * @param src : fp32 input pointer
            * @param dst : fp64 output pointer
            * @param proc_dims : ~.x -> width of processed area, in vec4; ~.y -> height of processed area
            * @param Wsrc : width of source matrix, in element
            * @param Wdst : eidth of destinated matrix, in element
            */
            _THREAD_FUNCTION_ void
            _v256_cvtps_pd2D(const float* src, double* dst, const uint2 proc_dims, const uint Wsrc, const uint Wdst);



            /**
            * @param src : fp32 input pointer
            * @param dst : fp64 output pointer
            * @param proc_dims : ~.x -> width of processed area, in vec4; ~.y -> height of processed area
            * @param Wsrc : width of source matrix, in element
            * @param Wdst : eidth of destinated matrix, in element
            */
            _THREAD_FUNCTION_ void
            _v256_cvtpd_ps2D(const double* src, float* dst, const uint2 proc_dims, const uint Wsrc, const uint Wdst);
        }

        /**
        * @param proc_num : The number of elements to be processed (in element)
        */
        void _cvtfp32_fp64_caller1D(const float* src, double* dst, const size_t proc_num);


        /**
        * @param proc_num : The number of elements to be processed (in element)
        */
        void _cvtfp64_fp32_caller1D(const double* src, float* dst, const size_t proc_num);


        /**
        * @param proc_num : The number of elements to be processed (in element)
        */
        void _cvtfp32_fp64_caller2D(const float* src, double* dst, const ulong2 proc_dims, const uint Wsrc, const uint Wdst);


        /**
        * @param proc_num : The number of elements to be processed (in element)
        */
        void _cvtfp64_fp32_caller2D(const double* src, float* dst, const ulong2 proc_dims, const uint Wsrc, const uint Wdst);
    }
}


#endif