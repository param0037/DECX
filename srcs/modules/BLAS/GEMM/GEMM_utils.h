/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _GEMM_UTILS_H_
#define _GEMM_UTILS_H_

#include "../../core/basic.h"


#define GEMM_BlockDim 16

namespace de
{
    enum GEMM_properties
    {
        HALF_GEMM_DIRECT    = 0,
        HALF_GEMM_ACCURATE  = 1
    };
}


#ifdef _DECX_BLAS_CPU_

namespace decx
{
    typedef struct GEMM_AB_configs
    {
        uint _pitchA, _pitchB, _pitchdst, _pitchC, _linear;
        uint2 _proc_dims;

        GEMM_AB_configs() {}


        GEMM_AB_configs(const uint pitchA, 
                        const uint pitchB, 
                        const uint pitchdst, 
                        const uint linear, 
                        const uint2 proc_dims, 
                        const uint pitchC = 0) :
            _pitchA(pitchA),
            _pitchB(pitchB),
            _pitchdst(pitchdst),
            _linear(linear),
            _pitchC(pitchC),
            _proc_dims(proc_dims) {}

    }_C_MM_;
}


namespace decx
{
    namespace gemm {
        namespace CPUK {
            /*
            * @param Wsrc : width of src, in float
            * @param Wdst : width of dst, in float
            * @param cpy_dim : ~.x -> width to copy (in __m256); ~.y -> height to copy (in float)
            */
            void GEMM_fp32_cpy_L8(const float* src, float* dst, const uint Wsrc, const uint Wdst, const uint2 cpy_dim);


            /*
            * @param Wsrc : width of src, in float
            * @param Wdst : width of dst, in float
            * @param cpy_dim : ~.x -> width to copy (in __m256); ~.y -> height to copy (in float)
            */
            void GEMM_fp64_cpy_L8(const double* src, double* dst, const uint Wsrc, const uint Wdst, const uint2 cpy_dim);
        }
    }
}



#endif

#endif