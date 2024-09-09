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

#include "../../../../common/Basic_process/gather/CPU/gather_kernels.h"
#include "resample.h"


_DECX_API_ void 
de::dsp::cpu::Resample(de::InputMatrix src, de::InputMatrix map, de::OutputMatrix dst)
{
    de::ResetLastError();

    const decx::_Matrix* _src = dynamic_cast<const decx::_Matrix*>(&src);
    const decx::_Matrix* _map = dynamic_cast<const decx::_Matrix*>(&map);
    decx::_Matrix* _dst = dynamic_cast<decx::_Matrix*>(&dst);

    decx::CPUK::gather2D_fp32_exec_bilinear((float*)_src->Mat.ptr, (float2*)_map->Mat.ptr, (float*)_dst->Mat.ptr, 
        make_uint2(_dst->Pitch() / 8, _dst->Height()), _src->Pitch(), _map->Pitch(), _dst->Pitch());
}