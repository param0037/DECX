/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _TEMPLATE_CONFIGURATIONS_H_
#define _TEMPLATE_CONFIGURATIONS_H_


#include "../../../../../core/basic.h"
#include "../../../../../core/thread_management/thread_pool.h"


namespace decx
{
    namespace rcp {
        namespace CPUK {
            /*
            * @param src : the read-only memory
            * @param len : the proccess length of single thread, in __m256
            * @param res_vec : the result vector in __m256
            */
            void _template_sq_sum_vec8_fp32(const float* src, const size_t len, float* res_vec);


            /*
            * @param src : the read-only memory
            * @param len : the proccess length of single thread, in __m256d
            * @param res_vec : the result vector in __m256d
            */
            void _template_sq_sum_vec8_uint8(const uint8_t* src, const size_t len, float* res_vec);



            /*
            * @param src : the read-only memory
            * @param len : the proccess length of single thread, in __m256
            * @param res_vec : the result vector in __m256
            */
            void _template_sum_vec8_fp32(const float* src, const size_t len, float* res_vec);


            /*
            * @param src : the read-only memory
            * @param len : the proccess length of single thread, in __m256d
            * @param res_vec : the result vector in __m256d
            */
            void _template_sum_vec4_fp64(const double* src, const size_t len, double* res_vec);



            void _template_normalize_fp32(const float* src, float* dst, const size_t len, const uint2 actual_dims);



            void _template_normalize_fp64(const double* src, double* dst, const size_t len, const uint2 actual_dims);


            /*
            * @param len : the total length of src array at physical memory, in __m256
            * @param Wsrc :
            */
            void _template_normalize_fp32_cpy2D(const float* src, float* dst, const size_t len, const uint pitchsrc, const uint pitchdst, const uint width, const uint height);



            void _template_normalize_uint8_cpy2D(const uint8_t* src, float* dst, const size_t len, const uint pitchsrc, const uint pitchdst, const uint width, const uint height);


            /**
            * @param len : The total length of src array at physical memory, in __m256
            * @param pitchsrc : The pitch of src matrix, in __m256
            * @param pitchdst : The pitch of dst matrix, in __m256
            * @param width : The width of copy, in float
            * @param height : The height of copy, in float
            */
            void _template_normalize_fp32_vec8_cpy2D(const float* src, float* dst, const size_t len, const uint pitchsrc, 
                const uint pitchdst, const uint width, const uint height);



            /**
            * @param len : The total length of src array at physical memory, in __m256
            * @param pitch : The pitch of src matrix, in __m256
            * @param width : The width of copy, in float
            * @param height : The height of copy, in float
            */
            void _template_normalize_fp32_vec8_cpy2D_inplace(float* src, const size_t len, const uint width,
                const uint pitch, const uint height);



            void _template_normalize_fp64_vec4_cpy2D(double* src, const size_t len, const uint2 actual_dims);
        }
    }
}


#endif