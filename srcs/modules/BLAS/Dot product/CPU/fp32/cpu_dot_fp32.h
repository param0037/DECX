/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _CPU_DOT_FP32_H_
#define _CPU_DOT_FP32_H_

#include "../../../../core/thread_management/thread_pool.h"
#include "../../../../classes/Vector.h"
#include "../../../../classes/Matrix.h"
#include "../../../../classes/Tensor.h"
#include "../../../../core/utils/fragment_arrangment.h"
#include "../../../../classes/classes_util.h"
#include "../../../../core/utils/intrinsics_ops.h"


namespace decx
{
    namespace dot {
        namespace CPUK {
            /*
            * @param src : the read-only memory
            * @param len : the proccess length of single thread, in __m256
            * @param res_vec : the result vector in __m256
            */
            _THREAD_FUNCTION_ void
                _dot_vec8_fp32(const float* A, const float* B, const size_t len, float* res_vec);
        }

        /*
        * @param src : the read-only memory
        * @param len : the proccess length of single thread, in __m256
        * @param res_vec : the result vector in __m256
        */
        void _dot_fp32_1D_caller(const float* A, const float* B, const size_t len, float* res_vec);
    }
}



namespace de
{
    namespace cpu
    {
        _DECX_API_ de::DH Dot(de::Vector& A, de::Vector& B, float* res);


        _DECX_API_ de::DH Dot(de::Matrix& A, de::Matrix& B, float* res);


        _DECX_API_ de::DH Dot(de::Tensor& A, de::Tensor& B, float* res);
    }
}



#endif