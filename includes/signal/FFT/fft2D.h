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


namespace de
{
	namespace signal
	{
		namespace cuda {
			_DECX_API_ de::DH FFT2D_R2C(de::Matrix& src, de::Matrix& dst);


			_DECX_API_ de::DH FFT2D_C2C(de::Matrix& src, de::Matrix& dst);


			_DECX_API_ de::DH IFFT2D_C2C(de::Matrix& src, de::Matrix& dst);


			_DECX_API_ de::DH IFFT2D_C2R(de::Matrix& src, de::Matrix& dst);
		}


		namespace cpu {
			_DECX_API_ de::DH FFT1D_R2C(de::Vector& src, de::Vector& dst);


			_DECX_API_ de::DH FFT1D_C2C(de::Vector& src, de::Vector& dst);


			_DECX_API_ de::DH IFFT1D_C2C(de::Vector& src, de::Vector& dst);


			_DECX_API_ de::DH IFFT1D_C2R(de::Vector& src, de::Vector& dst);


			_DECX_API_ de::DH FFT2D_R2C(de::Matrix& src, de::Matrix& dst);


			_DECX_API_ de::DH FFT2D_C2C(de::Matrix& src, de::Matrix& dst);


			_DECX_API_ de::DH IFFT2D_C2C(de::Matrix& src, de::Matrix& dst);


			_DECX_API_ de::DH IFFT2D_C2R(de::Matrix& src, de::Matrix& dst);
		}
	}
}


#endif