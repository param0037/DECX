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


#ifndef _MATRIX_TRANSFORM_H_
#define _MATRIX_TRANSFORM_H_

#include "../../../../../core/basic.h"
#include "../../../../../core/thread_management/thread_pool.h"
#include "../../../../../core/thread_management/thread_arrange.h"
#include "../../../../../core/utils/fragment_arrangment.h"
#include "../../../../../classes/Vector.h"
#include "../../../../../classes/Matrix.h"
#include "../../../vector4.h"


namespace decx
{
    namespace tf {
        namespace CPUK {
            typedef void (*_vec_mul_mat4_kernel1D_ptr)(const float*, float*, const decx::_Mat4x4f, const size_t);

            _THREAD_FUNCTION_ void
                _vec4_mul_mat4x4_fp32_1D(const float* src, float* dst, const decx::_Mat4x4f _tf_mat,
                    const size_t _proc_len);


            _THREAD_FUNCTION_ void
                _vec3_mul_mat4x3_fp32_1D(const float* src, float* dst, const decx::_Mat4x4f _tf_mat,
                    const size_t _proc_len);
        }

        void Vec4_transform(decx::_Vector* src, decx::_Vector* dst, decx::_Matrix* transform_matrix, de::DH* handle);


        void Vec3_transform(decx::_Vector* src, decx::_Vector* dst, decx::_Matrix* transform_matrix, de::DH* handle);
    }
}



namespace de
{
    namespace tf {
        namespace cpu {
            _DECX_API_ de::DH Vec_transform(de::Vector& src, de::Vector& dst, de::Matrix& transform_matrix);
        }
    }
}


#endif