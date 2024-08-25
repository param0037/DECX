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

#ifndef _TYPECAST_EXEC_X86_64_H_
#define _TYPECAST_EXEC_X86_64_H_

#include "../../../../modules/core/thread_management/thread_arrange.h"
#include "../../../SIMD/intrinsics_ops.h"
#include "../../../../modules/core/configs/config.h"
#include "../type_cast_methods.h"
#include "../../../element_wise/common/cpu_element_wise_planner.h"


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
        _THREAD_FUNCTION_ void _v256_cvtui8_f32_1D(const uint8_t* src, float* dst, const uint64_t proc_len);
        _THREAD_FUNCTION_ void _v256_cvtui8_i32_1D(const uint8_t* src, int32_t* dst, const uint64_t proc_len);
        _THREAD_FUNCTION_ void _v256_cvtps_i32(const float* src, int32_t* dst, const uint64_t proc_num);
        _THREAD_FUNCTION_ void _v256_cvtps_pd1D(const float* src, double* dst, const uint64_t proc_num);

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
        _THREAD_FUNCTION_ void _v256_cvtf32_ui8_cyclic1D(const float* src, uint8_t* dst, const uint64_t proc_len);
        _THREAD_FUNCTION_ void _v256_cvti32_ui8_cyclic1D(const int32_t* src, uint8_t* dst, const uint64_t proc_len);
        _THREAD_FUNCTION_ void _v256_cvti32_ps(const int32_t* src, float* dst, const uint64_t proc_num);
        _THREAD_FUNCTION_ void _v256_cvtpd_ps1D(const double* src, float* dst, const uint64_t proc_num);

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
        _THREAD_FUNCTION_ void _v256_cvtf32_ui8_saturated1D(const float* src, uint8_t* dst, const uint64_t proc_len);
        _THREAD_FUNCTION_ void _v256_cvti32_ui8_saturated1D(const int32_t* src, uint8_t* dst, const uint64_t proc_len);


        _THREAD_FUNCTION_ void _v256_cvti32_ui8_truncate1D(const int32_t* src, uint8_t* dst, const uint64_t proc_len);
        _THREAD_FUNCTION_ void _v256_cvti32_ui8_truncate_clamp_zero1D(const int32_t* src, uint8_t* dst, const uint64_t proc_len);

        template <typename _type_in, typename _type_out>
        using _cvt_kernel1D = void(const _type_in*, _type_out*, const uint64_t);

        template <typename _type_in, typename _type_out>
        using _cvt_kernel2D = void(const _type_in*, _type_out*, const uint2, const uint32_t, const uint32_t);
    }

    /**
    * @param src : fp64 input pointer
    * @param dst : fp32 output pointer
    * @param proc_dims : dimensions of processed area : ~.x -> width (in vec8); ~.y -> height
    * @param Wsrc : pitch of source matrix (in vec4 (in float))
    * @param Wdst : pitch of destinated matrix (in element)
    */
    // void _cvtui8_f32_caller1D(const uint8_t* src, float* dst, const uint64_t proc_len);
    // void _cvtui8_i32_caller1D(const float* src, float* dst, const uint64_t proc_len);
    // void _cvtfp32_i32_caller(const float* src, float* dst, const uint64_t proc_num);
    // void _cvti32_fp32_caller(const float* src, float* dst, const uint64_t proc_num);
    // void _cvtfp32_fp64_caller1D(const float* src, double* dst, const uint64_t proc_num);
    // void _cvtfp64_fp32_caller1D(const double* src, float* dst, const uint64_t proc_num);

    /**
    * @param src : fp64 input pointer
    * @param dst : fp32 output pointer
    * @param proc_dims : dimensions of processed area : ~.x -> width (in vec8); ~.y -> height
    * @param Wsrc : pitch of source matrix (in element)
    * @param Wdst : pitch of destinated matrix (in vec4 (in float))
    */
    // template <bool _print>
    decx::type_cast::CPUK::_cvt_kernel1D<float, uint8_t>* _cvtf32_ui8_selector1D(const int32_t flag);
    // template<bool _print>
    decx::type_cast::CPUK::_cvt_kernel1D<int32_t, uint8_t>* _cvti32_ui8_selector1D(const int32_t flag);
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
        _THREAD_FUNCTION_ void _v256_cvtui8_f32_2D(const uint8_t* src, float* dst, const uint2 proc_dims, const uint32_t Wsrc, const uint32_t Wdst);
        _THREAD_FUNCTION_ void _v256_cvtui8_i32_2D(const uint8_t* src, int32_t* dst, const uint2 proc_dims, const uint32_t Wsrc, const uint32_t Wdst);
        _THREAD_FUNCTION_ void _v256_cvtps_pd2D(const float* src, double* dst, const uint2 proc_dims, const uint32_t Wsrc, const uint32_t Wdst);

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
        _THREAD_FUNCTION_ void _v256_cvtf32_ui8_cyclic2D(const float* src, uint8_t* dst, const uint2 proc_dims, const uint32_t Wsrc, const uint32_t Wdst);
        _THREAD_FUNCTION_ void _v256_cvti32_ui8_cyclic2D(const int32_t* src, uint8_t* dst, const uint2 proc_dims, const uint32_t Wsrc, const uint32_t Wdst);
        _THREAD_FUNCTION_ void _v256_cvtpd_ps2D(const double* src, float* dst, const uint2 proc_dims, const uint32_t Wsrc, const uint32_t Wdst);

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
        _THREAD_FUNCTION_ void _v256_cvtf32_ui8_saturated2D(const float* src, uint8_t* dst, const uint2 proc_dims, const uint32_t Wsrc, const uint32_t Wdst);
        _THREAD_FUNCTION_ void _v256_cvti32_ui8_saturated2D(const int32_t* src, uint8_t* dst, const uint2 proc_dims, const uint32_t Wsrc, const uint32_t Wdst);


        _THREAD_FUNCTION_ void _v256_cvti32_ui8_truncate2D(const int32_t* src, uint8_t* dst, const uint2 proc_dims, const uint Wsrc, const uint Wdst);
        _THREAD_FUNCTION_ void _v256_cvti32_ui8_truncate_clamp_zero2D(const int32_t* src, uint8_t* dst, const uint2 proc_dims, const uint Wsrc, const uint Wdst);


        typedef void (*_cvt_f32_u8_kernel2D) (const float*, int*, const uint2, const uint32_t, const uint32_t);
    }

    /**
    * @param src : fp64 input pointer
    * @param dst : fp32 output pointer
    * @param proc_dims : dimensions of processed area : ~.x -> width (in vec8); ~.y -> height
    * @param Wsrc : pitch of source matrix (in vec4 (in float))
    * @param Wdst : pitch of destinated matrix (in element)
    */
    // void _cvtui8_f32_caller2D(const float* src, float* dst, const ulong2 proc_dims, const uint32_t Wsrc, const uint32_t Wdst);
    // void _cvtui8_i32_caller2D(const float* src, float* dst, const ulong2 proc_dims, const uint32_t Wsrc, const uint32_t Wdst);
    // void _cvtfp32_fp64_caller2D(const float* src, double* dst, const ulong2 proc_dims, const uint32_t Wsrc, const uint32_t Wdst);
    // void _cvtfp64_fp32_caller2D(const double* src, float* dst, const ulong2 proc_dims, const uint32_t Wsrc, const uint32_t Wdst);

    /**
    * @param src : fp64 input pointer
    * @param dst : fp32 output pointer
    * @param proc_dims : dimensions of processed area : ~.x -> width (in vec8); ~.y -> height
    * @param Wsrc : pitch of source matrix (in element)
    * @param Wdst : pitch of destinated matrix (in vec4 (in float))
    */
   
    void _cvtf32_ui8_caller2D(const float* src, int* dst, const ulong2 proc_dims, const uint Wsrc, const uint Wdst, const int flag, de::DH* handle);
    void _cvti32_ui8_caller2D(const float* src, int* dst, const ulong2 proc_dims, const uint Wsrc, const uint Wdst, const int flag, de::DH* handle);
}

namespace type_cast{
    template <typename _type_in, typename _type_out>
    static void typecast1D_general_caller(decx::type_cast::CPUK::_cvt_kernel1D<_type_in, _type_out> *_kernel_ptr, 
        decx::cpu_ElementWise1D_planner* _planner, const _type_in* src, _type_out* dst, const uint64_t proc_len,
        decx::utils::_thr_1D* t1D);
}
}


template <typename _type_in, typename _type_out>
static void decx::type_cast::typecast1D_general_caller(
    decx::type_cast::CPUK::_cvt_kernel1D<_type_in, _type_out>* _kernel_ptr, 
    decx::cpu_ElementWise1D_planner* _planner,  
    const _type_in* src,            _type_out* dst,                             
    const uint64_t proc_len,        decx::utils::_thr_1D* t1D)
{
    _planner->plan(decx::cpu::_get_permitted_concurrency(), proc_len, sizeof(_type_in), sizeof(_type_out));

    _planner->caller(_kernel_ptr, src, dst, t1D);
}


#endif
