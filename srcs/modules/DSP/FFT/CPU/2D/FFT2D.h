/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _FFT2D_H_
#define _FFT2D_H_


#include "CPU_FFT2D_planner.h"
#include "../../../../classes/Matrix.h"
#include "../../../../classes/Tensor.h"
#include "../../../../classes/Vector.h"


namespace decx
{
    namespace dsp
    {
        void InitFFT1Resources();

        void FreeFFT1Resources();


        void InitFFT2Resources();

        void FreeFFT2Resources();


        void InitFFT3Resources();

        void FreeFFT3Resources();
    }
}


namespace de
{
namespace dsp {
    namespace cpu 
    {
        _DECX_API_ de::DH FFT(de::Vector& src, de::Vector& dst);


        _DECX_API_ de::DH IFFT(de::Vector& src, de::Vector& dst, const de::_DATA_TYPES_FLAGS_ _output_type);


        _DECX_API_ de::DH FFT(de::Matrix& src, de::Matrix& dst);


        _DECX_API_ de::DH FFT(de::Tensor& src, de::Tensor& dst);


        _DECX_API_ de::DH IFFT(de::Tensor& src, de::Tensor& dst, const de::_DATA_TYPES_FLAGS_ _output_type);


        _DECX_API_ de::DH IFFT(de::Matrix& src, de::Matrix& dst, const de::_DATA_TYPES_FLAGS_ _output_type);
    }
}
}


#endif