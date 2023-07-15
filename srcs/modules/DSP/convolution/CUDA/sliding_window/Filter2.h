/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CUDA_FILTER2_H_
#define _CUDA_FILTER2_H_

#include "../../../../classes/Matrix.h"
#include "../../../../classes/MatrixArray.h"
#include "../../../../classes/GPU_Matrix.h"
#include "../../../../classes/GPU_MatrixArray.h"
#include "../../../../../Async Engine/DecxStream/DecxStream.h"

// APIs for users

namespace de
{
    namespace cuda {
        _DECX_API_ de::DH Filter2D(de::Matrix& src, de::Matrix& kernel, de::Matrix& dst, 
            const int conv_flag, const int accu_flag, const int output_type);


        _DECX_API_ de::DH
            Filter2D_multi_channel(de::MatrixArray& src, de::Matrix& kernel, de::MatrixArray& dst,
                const int conv_flag, const int accu_flag, const int output_type);


        _DECX_API_ de::DH
            Filter2D_multi_channel(de::MatrixArray& src, de::MatrixArray& kernel, de::MatrixArray& dst,
                const int conv_flag, const int accu_flag, const int output_type);
    }
}


// APIs for Async

namespace decx
{
    namespace cuda {
        void Filter2D_Raw_API(decx::_Matrix* _src, decx::_Matrix* _kernel, decx::_Matrix* _dst,
            const int conv_flag, const int accu_flag, const int output_type, de::DH* handle);


        void Filter2D_MC_SK_Raw_API(decx::_MatrixArray* src, decx::_Matrix* kernel, decx::_MatrixArray* dst,
                const int conv_flag, const int accu_flag, const int output_type, de::DH* handle);


        void Filter2D_MC_MK_Raw_API(decx::_MatrixArray* src, decx::_MatrixArray* kernel, decx::_MatrixArray* dst,
                const int conv_flag, const int accu_flag, const int output_type, de::DH* handle);
    }
}

// APIs for users

namespace de
{
    namespace cuda
    {
        _DECX_API_ de::DH Filter2D(de::GPU_Matrix& src, de::GPU_Matrix& kernel, de::GPU_Matrix& dst, 
            const int conv_flag, const int accu_flag, const int output_type);


        _DECX_API_ de::DH
            Filter2D_multi_channel(de::GPU_MatrixArray& src, de::GPU_Matrix& kernel, de::GPU_MatrixArray& dst,
                const int conv_flag, const int accu_flag, const int output_type);


        _DECX_API_ de::DH
            Filter2D_multi_channel(de::GPU_MatrixArray& src, de::GPU_MatrixArray& kernel, de::GPU_MatrixArray& dst,
                const int conv_flag, const int accu_flag, const int output_type);
    }
}

// APIs for Async


namespace decx
{
    namespace cuda {
        void dev_Filter2D_Raw_API(decx::_GPU_Matrix* _src, decx::_GPU_Matrix* _kernel, decx::_GPU_Matrix* _dst,
            const int conv_flag, const int accu_flag, const int output_type, de::DH* handle);


        void dev_Filter2D_MC_SK_Raw_API(decx::_GPU_MatrixArray* src, decx::_GPU_Matrix* kernel, decx::_GPU_MatrixArray* dst,
                const int conv_flag, const int accu_flag, const int output_type, de::DH* handle);


        void dev_Filter2D_MC_MK_Raw_API(decx::_GPU_MatrixArray* src, decx::_GPU_MatrixArray* kernel, decx::_GPU_MatrixArray* dst,
                const int conv_flag, const int accu_flag, const int output_type, de::DH* handle);
    }
}


namespace de
{
    namespace cuda
    {
        _DECX_API_ void Filter2D_Async(de::GPU_Matrix& src, de::GPU_Matrix& kernel, de::GPU_Matrix& dst,
            const int conv_flag, const int accu_flag, const int output_type, de::DecxStream& S);
    }
}


#endif