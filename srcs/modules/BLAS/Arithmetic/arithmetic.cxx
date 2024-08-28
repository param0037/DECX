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

#include "arithmetic.h"
#include "../../../common/element_wise/common/cpu_element_wise_planner.h"
#include "../../../common/element_wise/CPU/arithmetic_kernels.h"



_DECX_API_ void de::blas::cpu::
Arithmetic(de::InputVector A, de::InputVector B, de::OutputVector dst, const int32_t arith_flag)
{
    de::ResetLastError();

    const decx::_Vector* _A = dynamic_cast<const decx::_Vector*>(&A);
    const decx::_Vector* _B = dynamic_cast<const decx::_Vector*>(&B);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    decx::blas::vec_bin_arithmetic_caller(_A, _B, _dst, arith_flag, de::GetLastError());
}



_DECX_API_ void de::blas::cpu::
Arithmetic(de::InputMatrix A, de::InputMatrix B, de::OutputMatrix dst, const int32_t arith_flag)
{
    de::ResetLastError();

    const decx::_Matrix* _A = dynamic_cast<const decx::_Matrix*>(&A);
    const decx::_Matrix* _B = dynamic_cast<const decx::_Matrix*>(&B);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    decx::blas::mat_bin_arithmetic_caller(_A, _B, _dst, arith_flag, de::GetLastError());
}