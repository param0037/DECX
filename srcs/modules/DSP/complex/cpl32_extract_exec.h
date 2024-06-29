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


#ifndef _MODULE_FP32_EXEC_H_
#define _MODULE_FP32_EXEC_H_

#include "../../core/thread_management/thread_pool.h"
#include "../../core/utils/fragment_arrangment.h"
#include "../../core/thread_management/thread_arrange.h"
#include "../../classes/Matrix.h"
#include "../../classes/classes_util.h"


namespace decx
{
    namespace dsp {
        namespace CPUK 
        {
            typedef void (*cpl32_extract_kernel2D) (const double*, float*, const uint2, const uint64_t, const uint64_t);

            /**
            * @param src : Pointer of input complex array
            * @param dst : Pointer of output float32 array
            * @param _proc_dims : x -> width (in vec4); y -> height (in element)
            * @param Wsrc : width of source matrix, in element
            * @param Wdst : width of destinated matrix, in element
            */
            _THREAD_FUNCTION_ void
            _module_fp32_ST2D(const double* src, float* dst, const uint2 _proc_dims, const uint64_t Wsrc, const uint64_t Wdst);


            /**
            * @param src : Pointer of input complex array
            * @param dst : Pointer of output float32 array
            * @param _proc_dims : x -> width (in vec4); y -> height (in element)
            * @param Wsrc : width of source matrix, in element
            * @param Wdst : width of destinated matrix, in element
            */
            _THREAD_FUNCTION_ void
            _angle_fp32_ST2D(const double* src, float* dst, const uint2 _proc_dims, const uint64_t Wsrc, const uint64_t Wdst);
        }

        /**
        * @param src : Pointer of input complex array
        * @param dst : Pointer of output float32 array
        * @param _total_len : The total length of the 1D array, in de::CPf (already aligned to 4)
        */
        void _cpl32_extract_caller(const de::CPf* src, float* dst, const uint2 _proc_dims,
            const uint64_t Wsrc, const uint64_t Wdst, decx::dsp::CPUK::cpl32_extract_kernel2D kernel);
    }
}


#endif