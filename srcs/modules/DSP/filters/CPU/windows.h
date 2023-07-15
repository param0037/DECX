/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _WINDOWS_H_
#define _WINDOWS_H_


#include "../../../core/basic.h"
#include "../../../classes/Matrix.h"
#include "../../../classes/Vector.h"
#include "../../../core/thread_management/thread_pool.h"
#include "../../../core/thread_management/thread_arrange.h"
#include "../../../core/utils/intrinsics_ops.h"
#include "../../../core/utils/fragment_arrangment.h"


namespace decx {
    namespace signal {
        namespace CPUK 
        {
            /*
            * @param _proc_len : Length of processed area, in __m256 (vec4) (de::CPf x4)
            * @param real_bound : Real length of the array, in de::CPf (its own element)
            * @param global_dex_offset : Offset of global dex, in its own element (de::CPf)
            */
            _THREAD_FUNCTION_ void
            Gaussian_Window1D_cpl32(const double* src, double* dst, const float u, const float sigma, const size_t _proc_len,
                const size_t real_bound, const size_t global_dex_offset);


            _THREAD_FUNCTION_ void
            Gaussian_Window2D_cpl32_no_corrolation(const double* src, double* dst, const float2 u, const float2 sigma, const uint2 proc_dims,
                const uint2 real_bound, const uint global_dex_offset_Y, const uint pitch);


            _THREAD_FUNCTION_ void
            Gaussian_Window2D_cpl32(const double* src, double* dst, const float2 u, const float2 sigma, const float p, const uint2 proc_dims,
                const uint2 real_bound, const uint global_dex_offset_Y, const uint pitch);



            _THREAD_FUNCTION_ void
            Cone_Window2D_cpl32(const double* src, double* dst, const uint2 origin, const float radius, const uint2 proc_dims,
                const uint2 real_bound, const uint global_dex_offset_Y, const uint pitch);



            /**
            * @param _proc_len : Length of processed area, in __m256 (vec4) (de::CPf x4)
            * @param real_bound : Real length of the array, in de::CPf (its own element)
            * @param global_dex_offset : Offset of global dex, in its own element (de::CPf)
            */
            _THREAD_FUNCTION_ void
            Triangular_Window1D_cpl32(const double* src, double* dst, const long long center, const size_t radius, const size_t _proc_len,
                const size_t real_bound, const size_t global_dex_offset);
        }
    }
}



namespace de {
    namespace signal {
        namespace cpu {
            _DECX_API_ de::DH Gaussian_Window1D(de::Vector& src, de::Vector& dst, const float u, const float sigma);


            _DECX_API_ de::DH Triangular_Window1D(de::Vector& src, de::Vector& dst, const long long center, size_t radius);


            _DECX_API_ de::DH Gaussian_Window2D(de::Matrix& src, de::Matrix& dst, const de::Point2D_f u, const de::Point2D_f sigma, const float p);


            _DECX_API_ de::DH Cone_Window2D(de::Matrix& src, de::Matrix& dst, const de::Point2D origin, const float radius);
        }
    }
}


#endif