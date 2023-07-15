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


#include "../../basic.h"
#include "../../classes/Matrix.h"
#include "../../classes/Vector.h"
#include "fft1D.h"


namespace de
{
	namespace signal {
		namespace cpu {
			_DECX_API_ de::DH Module(de::Matrix& src, de::Matrix& dst);
		}
	}
}


namespace de
{
	namespace signal
	{
		namespace cuda {
			_DECX_API_ de::DH FFT2D(de::Matrix& src, de::Matrix& dst, const int FFT_flag);


			_DECX_API_ de::DH IFFT2D(de::Matrix& src, de::Matrix& dst, const int FFT_flag);


			_DECX_API_ de::DH FFT2D(de::GPU_Matrix& src, de::GPU_Matrix& dst, const int FFT_flag);


			_DECX_API_ de::DH IFFT2D(de::GPU_Matrix& src, de::GPU_Matrix& dst, const int FFT_flag);
		}


		namespace cpu {
			_DECX_API_ de::DH FFT2D(de::Matrix& src, de::Matrix& dst, const int FFT_flag);


			_DECX_API_ de::DH IFFT2D(de::Matrix& src, de::Matrix& dst, const int FFT_flag);
		}
	}
}


#endif