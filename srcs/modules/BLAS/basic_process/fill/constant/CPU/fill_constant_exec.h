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

#ifndef _FILL_CONSTANT_EXEC_H_
#define _FILL_CONSTANT_EXEC_H_


#include "../../../../../core/thread_management/thread_pool.h"
#include "../../../../../core/thread_management/thread_arrange.h"
#include "../../../../../core/utils/fragment_arrangment.h"
#include "../../../../../core/utils/intrinsics_ops.h"

// 1D
namespace decx
{
    namespace bp {
        namespace CPUK 
        {
            template <bool _is_left>
            /**
            * @param src :
            * @param val : values
            * @param fill_len : length of the area to be filled, in vec8 (for float, int32)
            */
            _THREAD_FUNCTION_ void
            fill_v256_b32_1D_end(float* __restrict src, const __m256 val, const size_t fill_len, const __m256 _end_val = _mm256_set1_ps(0));


            /**
            * @param src :
            * @param val : values
            * @param fill_len : length of the area to be filled, in vec8 (for double, de::CPf)
            */
            template <bool _is_left>
            _THREAD_FUNCTION_ void
            fill_v256_b64_1D_end(double* src, const __m256d val, const size_t fill_len, const __m256d _end_val = _mm256_set1_pd(0));
        }

        /*
        * @param fill_len : length of the area to be filled, in element
        */
        void fill1D_v256_b32_caller_MT(float* src, const float val, const size_t fill_len, 
            decx::utils::_thread_arrange_1D* t1D);


        void fill1D_v256_b32_caller_ST(float* src, const float val, const size_t fill_len);


        /*
        * @param fill_len : length of the area to be filled, in element
        */
        void fill1D_v256_b64_caller_MT(double* src, const double val, const size_t fill_len,
            decx::utils::_thread_arrange_1D* t1D);


        void fill1D_v256_b64_caller_ST(double* src, const double val, const size_t fill_len);
    }
}


// 2D
namespace decx
{
    namespace bp {
        namespace CPUK {
            /**
            * @param src :
            * @param val : values
            * @param fill_len : length of the area to be filled, in vec8 (for float, int32)
            * @param Wsrc : width of the matrix, in element
            */
            template <bool _is_left>
            _THREAD_FUNCTION_ void
            fill_v256_b32_2D_LF(float* __restrict src, const __m256 val, const uint2 proc_dims, const uint Wsrc, 
                const __m256 _end_val = _mm256_set1_ps(0));


            /**
            * @param src :
            * @param val : values
            * @param fill_len : length of the area to be filled, in vec8 (for double, de::CPf)
            * @param Wsrc : width of the matrix, in element
            */
            template <bool _is_left>
            _THREAD_FUNCTION_ void
            fill_v256_b64_2D_LF(double* src, const __m256d val, const uint2 proc_dims, const uint Wsrc, 
                const __m256d _end_val = _mm256_set1_pd(0));
        }

        /**
        * @param fill_len : length of the area to be filled, in element
        */
        void fill2D_v256_b32_caller_MT(float* src, const float val, const uint2 proc_dims, const uint Wsrc,
            decx::utils::_thread_arrange_1D* t1D);


        void fill2D_v256_b32_caller_ST(float* src, const float val, const uint2 proc_dims, const uint Wsrc);


        /**
        * @param fill_len : length of the area to be filled, in element
        */
        void fill2D_v256_b64_caller_MT(double* src, const double val, const uint2 proc_dims, const uint Wsrc,
            decx::utils::_thread_arrange_1D* t1D);


        void fill2D_v256_b64_caller_ST(double* src, const double val, const uint2 proc_dims, const uint Wsrc);
    }
}


// 3D
//namespace decx
//{
//    namespace bp {
//        namespace CPUK {
//            /**
//            * @param src :
//            * @param val : values
//            * @param fill_len : length of the area to be filled, in vec8 (for float, int32)
//            * @param Wsrc : width of the matrix, in element
//            */
//            _THREAD_FUNCTION_ void
//                fill_v256_b32_3D(float* src, const __m256 val, const uint2 proc_dims, const uint Wsrc);
//
//
//            _THREAD_FUNCTION_ void
//                fill_v256_b32_2D_LF(float* src, const __m256 val, const __m256 _end_val, const uint2 proc_dims, const uint Wsrc);
//
//
//            /**
//            * @param src :
//            * @param val : values
//            * @param fill_len : length of the area to be filled, in vec8 (for double, de::CPf)
//            * @param Wsrc : width of the matrix, in element
//            */
//            _THREAD_FUNCTION_ void
//                fill_v256_b64_2D(double* src, const __m256d val, const uint2 proc_dims, const uint Wsrc);
//
//
//            _THREAD_FUNCTION_ void
//                fill_v256_b64_2D_LF(double* src, const __m256d val, const __m256d _end_val, const uint2 proc_dims, const uint Wsrc);
//        }
//
//        /**
//        * @param fill_len : length of the area to be filled, in element
//        */
//        void fill2D_v256_b32_caller_MT(float* src, const float val, const uint2 proc_dims, const uint Wsrc,
//            decx::utils::_thread_arrange_1D* t1D);
//
//
//        void fill2D_v256_b32_caller_ST(float* src, const float val, const uint2 proc_dims, const uint Wsrc);
//
//
//        /**
//        * @param fill_len : length of the area to be filled, in element
//        */
//        void fill2D_v256_b64_caller_MT(double* src, const double val, const uint2 proc_dims, const uint Wsrc,
//            decx::utils::_thread_arrange_1D* t1D);
//
//
//        void fill2D_v256_b64_caller_ST(double* src, const double val, const uint2 proc_dims, const uint Wsrc);
//    }
//}


#endif