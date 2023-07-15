/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _MATRIX_TRANSFORM_H_
#define _MATRIX_TRANSFORM_H_

#include "../../../../core/basic.h"
#include "../../../../core/thread_management/thread_pool.h"
#include "../../../../core/thread_management/thread_arrange.h"
#include "../../../../core/utils/fragment_arrangment.h"
#include "../../../../classes/Vector.h"
#include "../../../../classes/Matrix.h"
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