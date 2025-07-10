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


#ifndef _GEMM_CALLERS_H_
#define _GEMM_CALLERS_H_

#include "../GEMM_utils.h"
#include <BLAS/GEMM/CPU/cpu_GEMM_config.h>

namespace decx
{
namespace blas {
    template <bool _ABC>
    void GEMM_fp32(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst, decx::_Matrix* C = NULL);


    template <bool _ABC, bool _cplxf>
    void GEMM_64b(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst, decx::_Matrix* C = NULL);


    template <bool _ABC>
    void GEMM_cplxd(decx::_Matrix* A, decx::_Matrix* B, decx::_Matrix* dst, decx::_Matrix* C = NULL);
}
}


namespace decx
{
    namespace blas {
        extern decx::ResourceHandle g_cpu_GEMM_fp32_planner;
        extern decx::ResourceHandle g_cpu_GEMM_64b_planner;
        extern decx::ResourceHandle g_cpu_GEMM_cplxd_planner;
    }
}



#endif