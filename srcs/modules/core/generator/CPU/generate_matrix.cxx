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

#include <Element_wise/Generator/CPU/fill_kernels.h>
#include "cpu_Generator.h"
#include <Classes/Matrix.h>
#include <Classes/Number.h>

namespace decx
{
    template <typename _data_type>
    static void cpu_generate2D(decx::_Matrix* mat, const _data_type val = 0);


    static void cpu_random2D(decx::_Matrix* mat, const double min, const double max, const int32_t seed = 0);
}


template <typename _data_type>
static void decx::cpu_generate2D(decx::_Matrix* mat, const _data_type val)
{
    decx::cpu_Generator2D planner;
    const uint32_t conc = decx::cpu::_get_permitted_concurrency();
    planner.plan(conc, make_uint2(mat->Width(), mat->Height()), sizeof(_data_type), sizeof(_data_type));
    
    decx::utils::_thread_arrange_1D t1D(conc);
    planner.fill_caller(decx::CPUK::fill2D_constant_fp32, (_data_type*)mat->Mat.ptr, mat->Pitch(), &t1D, val);
}


static void decx::cpu_random2D(decx::_Matrix* mat, const double min, const double max, const int32_t seed)
{
    if (seed){
        srand(seed);
    }

    decx::cpu_Generator2D planner;
    const uint32_t conc = decx::cpu::_get_permitted_concurrency();
    const int32_t element_size = mat->get_layout()._single_element_size;
    planner.plan(conc, make_uint2(mat->Width(), mat->Height()), element_size, element_size);
    
    decx::utils::_thread_arrange_1D t1D(conc);
    switch (mat->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        planner.fill_caller(decx::CPUK::fill2D_rand_fp32, (float*)mat->Mat.ptr, mat->Pitch(), &t1D, (float)min, (float)max);
        break;

    case de::_DATA_TYPES_FLAGS_::_INT32_:
        planner.fill_caller(decx::CPUK::fill2D_rand_int32, (int32_t*)mat->Mat.ptr, mat->Pitch(), &t1D, (int32_t)min, (int32_t)max);
        break;
    
    default:
        break;
    }
    
}


_DECX_API_ void de::cpu::Generate(de::OutputMatrix mat, const de::_DATA_TYPES_FLAGS_ type_out, de::Number val)
{
    de::GetLastError();

    decx::_Matrix* _mat = dynamic_cast<decx::_Matrix*>(&mat);

    switch (type_out)
    {
    case de::_DATA_TYPES_FLAGS_::_FP32_:
        decx::cpu_generate2D<float>(_mat, val.get_data_ref<float>());
        break;
    
    default:
        break;
    }
}

_DECX_API_ void de::cpu::Random(de::OutputMatrix mat, const de::_DATA_TYPES_FLAGS_ type_out, 
        const uint32_t width, const uint32_t height, const int32_t seed, const de::Point2D_d range)
{
    de::GetLastError();

    decx::_Matrix* _mat = dynamic_cast<decx::_Matrix*>(&mat);

    _mat->re_construct(type_out, width, height);
    decx::cpu_random2D(_mat, range.x, range.y, seed);
}
