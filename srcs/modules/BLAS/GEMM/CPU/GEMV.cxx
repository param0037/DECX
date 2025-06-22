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

#include <BLAS/MVM/CPU/MVM_planner.h>
#include <Classes/Matrix.h>
#include <Classes/Vector.h>


namespace de
{
namespace blas{
namespace cpu{

    _DECX_API_ void GEMV(de::Matrix& A, de::Vector& B, de::Vector& dst);

}
}
}


_DECX_API_ void de::blas::cpu::GEMV(de::Matrix& A, de::Vector& B, de::Vector& dst)
{
    decx::_Matrix* _A = dynamic_cast<decx::_Matrix*>(&A);
    decx::_Vector* _B = dynamic_cast<decx::_Vector*>(&B);
    decx::_Vector* _dst = dynamic_cast<decx::_Vector*>(&dst);

    decx::blas::cpu_MVM_planner<float> planner;
    planner.Config(make_uint2(_A->Width(), _A->Height()));

    _dst->re_construct(de::_FP32_, _A->Height());

    planner.Run((float*)_A->Mat, (float*)_B->Vec, (float*)_dst->Vec, _A->Pitch());
    return;
}