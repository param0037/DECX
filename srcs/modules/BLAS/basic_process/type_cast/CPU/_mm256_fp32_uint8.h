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


#ifndef _MM256_FP32_UINT8_H_
#define _MM256_FP32_UINT8_H_



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
            * @param proc_dims : dimensions of processed area : ~.x -> width (in vec8); ~.y -> height
            * @param Wsrc : pitch of source matrix (in vec4 (in float))
            * @param Wdst : pitch of destinated matrix (in element)
            * @param proc_num : The number of elements to be processed (grouped in 4 per group)(vec4)
            */
            _THREAD_FUNCTION_ void
            _v256_cvtui8_f32_1D(const float* src, float* dst, const size_t proc_len);



            /**
            * This function will convert int32 to unsigned int8, and clamp all negative inputs to zero. 
            * if src > 256 (overflow) -> dst := 0
            * else dst -> truncate8(src)
            * @param src : fp64 input pointer
            * @param dst : fp32 output pointer
            * @param proc_dims : dimensions of processed area : ~.x -> width (in vec8); ~.y -> height
            * @param Wsrc : pitch of source matrix (in element)
            * @param Wdst : pitch of destinated matrix (in element)
            */
            _THREAD_FUNCTION_ void
            _v256_cvtf32_ui8_cyclic1D(const float* src, int* dst, const size_t proc_len);



            /**
            * This function will convert int32 to unsigned int8, and clamp all negative inputs to zero. 
            * if src > 256 (overflow) -> dst := 255
            * else dst -> truncate8(src)
            * @param src : fp64 input pointer
            * @param dst : fp32 output pointer
            * @param proc_dims : dimensions of processed area : ~.x -> width (in vec8); ~.y -> height
            * @param Wsrc : pitch of source matrix (in element)
            * @param Wdst : pitch of destinated matrix (in element)
            */
            _THREAD_FUNCTION_ void
            _v256_cvtf32_ui8_saturated1D(const float* src, int* dst, const size_t proc_len);

            typedef void (*_cvt_f32_u8_kernel1D) (const float*, int*, const size_t);
        }

        /**
        * @param src : fp64 input pointer
        * @param dst : fp32 output pointer
        * @param proc_dims : dimensions of processed area : ~.x -> width (in vec8); ~.y -> height
        * @param Wsrc : pitch of source matrix (in vec4 (in float))
        * @param Wdst : pitch of destinated matrix (in element)
        */
        void _cvtui8_f32_caller1D(const float* src, float* dst, const size_t proc_len);


        /**
        * @param src : fp64 input pointer
        * @param dst : fp32 output pointer
        * @param proc_dims : dimensions of processed area : ~.x -> width (in vec8); ~.y -> height
        * @param Wsrc : pitch of source matrix (in element)
        * @param Wdst : pitch of destinated matrix (in vec4 (in float))
        */
        template <bool _print>
        void _cvtf32_ui8_caller1D(const float* src, int* dst, const size_t proc_len, const int flag, de::DH* handle);
    }


    // ------------------------------------------- 2D --------------------------------------------------

    namespace type_cast {
        namespace CPUK {
            /**
            * @param src : fp32 input pointer
            * @param dst : fp64 output pointer
            * @param proc_dims : dimensions of processed area : ~.x -> width (in vec8); ~.y -> height
            * @param Wsrc : pitch of source matrix (in vec4 (in float))
            * @param Wdst : pitch of destinated matrix (in element)
            * @param proc_num : The number of elements to be processed (grouped in 4 per group)(vec4)
            */
            _THREAD_FUNCTION_ void
            _v256_cvtui8_f32_2D(const float* src, float* dst, const uint2 proc_dims, const uint Wsrc, const uint Wdst);



            /**
            * This function will convert int32 to unsigned int8, and clamp all negative inputs to zero. 
            * if src > 256 (overflow) -> dst := 0
            * else dst -> truncate8(src)
            * @param src : fp64 input pointer
            * @param dst : fp32 output pointer
            * @param proc_dims : dimensions of processed area : ~.x -> width (in vec8); ~.y -> height
            * @param Wsrc : pitch of source matrix (in element)
            * @param Wdst : pitch of destinated matrix (in element)
            */
            _THREAD_FUNCTION_ void
            _v256_cvtf32_ui8_cyclic2D(const float* src, int* dst, const uint2 proc_dims, const uint Wsrc, const uint Wdst);



            /**
            * This function will convert int32 to unsigned int8, and clamp all negative inputs to zero. 
            * if src > 256 (overflow) -> dst := 255
            * else dst -> truncate8(src)
            * @param src : fp64 input pointer
            * @param dst : fp32 output pointer
            * @param proc_dims : dimensions of processed area : ~.x -> width (in vec8); ~.y -> height
            * @param Wsrc : pitch of source matrix (in element)
            * @param Wdst : pitch of destinated matrix (in element)
            */
            _THREAD_FUNCTION_ void
            _v256_cvtf32_ui8_saturated2D(const float* src, int* dst, const uint2 proc_dims, const uint Wsrc, const uint Wdst);

            typedef void (*_cvt_f32_u8_kernel2D) (const float*, int*, const uint2, const uint, const uint);
        }

        /**
        * @param src : fp64 input pointer
        * @param dst : fp32 output pointer
        * @param proc_dims : dimensions of processed area : ~.x -> width (in vec8); ~.y -> height
        * @param Wsrc : pitch of source matrix (in vec4 (in float))
        * @param Wdst : pitch of destinated matrix (in element)
        */
        void _cvtui8_f32_caller2D(const float* src, float* dst, const ulong2 proc_dims, const uint Wsrc, const uint Wdst);


        /**
        * @param src : fp64 input pointer
        * @param dst : fp32 output pointer
        * @param proc_dims : dimensions of processed area : ~.x -> width (in vec8); ~.y -> height
        * @param Wsrc : pitch of source matrix (in element)
        * @param Wdst : pitch of destinated matrix (in vec4 (in float))
        */
        template <bool _print>
        void _cvtf32_ui8_caller2D(const float* src, int* dst, const ulong2 proc_dims, 
            const uint Wsrc, const uint Wdst, const int flag, de::DH* handle);
    }
}


#endif