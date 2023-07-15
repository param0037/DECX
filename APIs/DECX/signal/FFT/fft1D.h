/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _FFT1D_H_
#define _FFT1D_H_

#include "../../basic.h"
#include "../../classes/Vector.h"
#include "../../classes/GPU_Vector.h"
#include "../../classes/GPU_Matrix.h"
#include "../../classes/class_utils.h"


namespace de
{
	namespace signal
	{
		enum FFT_flags {
			FFT_R2C = 0,
			FFT_C2C = 1,
			IFFT_C2C = 2,
			IFFT_C2R = 3
		};

		namespace cuda {
			_DECX_API_ de::DH FFT1D(de::Vector& src, de::Vector& dst, const int FFT_flag);


			_DECX_API_ de::DH IFFT1D(de::Vector& src, de::Vector& dst, const int FFT_flag);
		}

		namespace cpu
		{
			_DECX_API_ de::DH FFT1D(de::Vector& src, de::Vector& dst, const int FFT_flag);


			_DECX_API_ de::DH IFFT1D(de::Vector& src, de::Vector& dst, const int FFT_flag);
		}
	}
}


#endif