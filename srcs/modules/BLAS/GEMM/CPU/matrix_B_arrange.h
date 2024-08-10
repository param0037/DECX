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


#ifndef _MATRIX_B_ARRANGE_H_
#define _MATRIX_B_ARRANGE_H_


#include "../../../../common/Classes/Matrix.h"
#include "../GEMM_utils.h"
#include "../../../core/thread_management/thread_arrange.h"
#include "../../../core/thread_management/thread_pool.h"


namespace decx
{
    namespace blas {
        void matrix_B_arrange_fp32(const float* src, float* dst, const uint32_t pitchsrc_v1,
            const uint32_t pitchdst_v8, const decx::utils::frag_manager* _fmgr_WH, decx::utils::_thr_2D* t2D);

        template <bool _cplxf>
        void matrix_B_arrange_64b(const double* src, double* dst, const uint32_t pitchsrc_v1,
            const uint32_t pitchdst_v8, const decx::utils::frag_manager* _fmgr_WH, decx::utils::_thr_2D* t2D);


        void matrix_B_arrange_cplxd(const de::CPd* src, de::CPd* dst, const uint32_t pitchsrc_v1,
            const uint32_t pitchdst_v8, const decx::utils::frag_manager* _fmgr_WH, decx::utils::_thr_2D* t2D);
    }
}

#endif
