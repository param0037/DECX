/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _FILTERS_H_
#define _FILTERS_H_


#include "../../classes/Matrix.h"
#include "../../classes/Vector.h"
#include "../../classes/GPU_Vector.h"
#include "../../classes/GPU_Matrix.h"


namespace de {
	namespace dsp {
		namespace cuda {
			_DECX_API_ de::DH LowPass1D_Ideal(de::GPU_Vector& src, de::GPU_Vector& dst, const size_t cutoff_frequency);


			_DECX_API_ de::DH LowPass2D_Ideal(de::GPU_Matrix& src, de::GPU_Matrix& dst, const de::Point2D cutoff_frequency);


			_DECX_API_ de::DH Gaussian_Window1D(de::GPU_Vector& src, de::GPU_Vector& dst, const float u, const float sigma);


			_DECX_API_ de::DH Gaussian_Window2D(de::GPU_Matrix& src, de::GPU_Matrix& dst, const de::Point2D_f u, const de::Point2D_f sigma, const float p);


			_DECX_API_ de::DH Triangular_Window1D(de::GPU_Vector& src, de::GPU_Vector& dst, const long long origin, const size_t radius);


			_DECX_API_ de::DH Cone_Window2D(de::GPU_Matrix& src, de::GPU_Matrix& dst, const de::Point2D origin, const float radius);
		}

		namespace cpu {
			_DECX_API_ de::DH LowPass1D_Ideal(de::Vector& src, de::Vector& dst, const size_t cutoff_frequency);


			_DECX_API_ de::DH LowPass2D_Ideal(de::Matrix& src, de::Matrix& dst, const de::Point2D cutoff_frequency);


			_DECX_API_ de::DH Gaussian_Window1D(de::Vector& src, de::Vector& dst, const float u, const float sigma);


			_DECX_API_ de::DH Triangular_Window1D(de::Vector& src, de::Vector& dst, const long long center, size_t radius);


			_DECX_API_ de::DH Gaussian_Window2D(de::Matrix& src, de::Matrix& dst, const de::Point2D_f u, const de::Point2D_f sigma, const float p);


			_DECX_API_ de::DH Cone_Window2D(de::Matrix& src, de::Matrix& dst, const de::Point2D origin, const float radius);


			_DECX_API_ de::DH ButterWorth_LP2D(de::Matrix& src, de::Matrix& dst, const float cutoff_freq, const int order);


			_DECX_API_ void Resample(de::InputMatrix src, de::InputMatrix map, de::OutputMatrix dst);
		}
	}
}


#endif