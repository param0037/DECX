/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
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