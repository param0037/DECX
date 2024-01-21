/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _DP2D_1WAY_CALLERS_CUH_
#define _DP2D_1WAY_CALLERS_CUH_


#include "DP2D_1way.cuh"



namespace decx
{
    namespace dot
    {
        /**
        * @param dev_A : The device pointer where vector A is stored
        * @param dev_B : The device pointer where vector B is stored
        * @param _actual_len : The actual length of the vector, measured in element
        * @param _kp_configs : The pointer of reduction summation configuration, don't need to be initialized
        * @param S : The pointer of CUDA stream
        *
        * @return The pointer where the result being stored
        */
        template <bool _is_reduce_h>
        const void* cuda_DP2D_1way_fp32_caller_Async(decx::dot::cuda_DP2D_configs<float>* _configs, decx::cuda_stream* S);


        template <bool _is_reduce_h>
        const void* cuda_DP2D_1way_fp16_caller_Async(decx::dot::cuda_DP2D_configs<de::Half>* _configs, const uint32_t _fp16_accu, decx::cuda_stream* S);
    }
}



namespace decx
{
    namespace dot
    {
        template <bool _is_reduce_h>
        void matrix_dot_1way_fp32(decx::_Matrix* A, decx::_Matrix* B, decx::_Vector* res);


        template <bool _is_reduce_h>
        void matrix_dot_1way_fp16(decx::_Matrix* A, decx::_Matrix* B, decx::_Vector* res, const uint32_t _fp16_accu);
    }
}



#endif