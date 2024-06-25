/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CUDA_FFTS_CUH_
#define _CUDA_FFTS_CUH_

#include "../../../classes/GPU_Tensor.h"
#include "../../../classes/GPU_Matrix.h"
#include "../../../classes/GPU_Vector.h"
#include "../FFT_commons.h"


namespace de
{
namespace dsp {
namespace cuda
{
    _DECX_API_ void FFT(de::Vector& src, de::Vector& dst);


    _DECX_API_ void FFT(de::GPU_Vector& src, de::GPU_Vector& dst);


    _DECX_API_ void IFFT(de::Vector& src, de::Vector& dst, const de::_DATA_TYPES_FLAGS_ _type_out);


    _DECX_API_ void IFFT(de::GPU_Vector& src, de::GPU_Vector& dst, const de::_DATA_TYPES_FLAGS_ _type_out);


    _DECX_API_ void FFT(de::GPU_Matrix& src, de::GPU_Matrix& dst, const de::_DATA_TYPES_FLAGS_ _type_out);


    _DECX_API_ void IFFT(de::GPU_Matrix& src, de::GPU_Matrix& dst, const de::_DATA_TYPES_FLAGS_ type_out);


    _DECX_API_ void FFT(de::Matrix& src, de::Matrix& dst);


    _DECX_API_ void IFFT(de::Matrix& src, de::Matrix& dst, const de::_DATA_TYPES_FLAGS_ type_out);


    _DECX_API_ void FFT(de::GPU_Tensor& src, de::GPU_Tensor& dst);


    _DECX_API_ void IFFT(de::GPU_Tensor& src, de::GPU_Tensor& dst, const de::_DATA_TYPES_FLAGS_ type_out);
}
}
}



#endif