/**
*	---------------------------------------------------------------------
*	Author : Wayne Anderson
*   Date   : 2021.04.16
*	---------------------------------------------------------------------
*	This is a part of the open source program named "DECX", copyright c Wayne,
*	2021.04.16
*/

#ifndef _CLASS_UTILS_H_
#define _CLASS_UTILS_H_

#include "../basic.h"
#include "type_info.h"

#ifdef __cplusplus
namespace de
{
	struct Half
	{
		unsigned short val;
	};


	typedef struct __align__(8) complex_f
	{
		float real, image;

		complex_f(const float Freal, const float Fimage) {
			real = Freal;
			image = Fimage;
		}

		complex_f() {}
	}CPf;


	typedef struct __align__(16) complex_d
	{
		double real, image;

		complex_d(const double Freal, const double Fimage) {
			this->real = Freal;
			this->image = Fimage;
		}

		complex_d() { this->real = 0; this->image = 0; }
	}CPd;


	struct __align__(8) Point2D
	{
		int32_t x, y;

		Point2D(const int32_t _x, const int32_t _y) { x = _x; y = _y; }

		Point2D(const uint32_t _x, const uint32_t _y) { x = (int32_t)_x; y = (int32_t)_y; }

		Point2D() {}
	};


	struct __align__(16) Point3D
	{
		int32_t x, y, z;

		Point3D(const int32_t _x, const int32_t _y, const int32_t _z) { x = _x; y = _y; z = _z; }

		Point3D(const uint32_t _x, const uint32_t _y, const uint32_t _z) { x = (int32_t)_x; y = (int32_t)_y; z = (int32_t)_z; }

		Point3D() {}
	};


	struct __align__(8) Point2D_f
	{
		float x, y;
		Point2D_f(const float _x, const float _y) { x = _x; y = _y; }
		Point2D_f() {}
	};


    struct __align__(16) Point2D_d
	{
		double x, y;
		Point2D_d(const double _x, const double _y) { x = _x; y = _y; }
		Point2D_d() {}
	};


    struct __align__(4) uchar4
	{
		uint8_t x, y, z, w;
	};

	namespace vis
	{
		struct __align__(4) Pixel
		{
			uint8_t r, g, b, alpha;
		};
	}
}


namespace de
{
	struct __align__(16) Vector3f {
		float x, y, z;
	};

	struct __align__(8) Vector2f {
		float x, y;
	};

	struct __align__(16) Vector4f {
		float x, y, z, w;
	};
}


namespace de
{
	enum Fp16_Accuracy_Levels
	{
		/**
		* Usually does the loads in fp16 but all calculations in fp32.
		* Usually the slowest method, this method cares about overflow (from 16-bit to 32-bit).
		*/
		Fp16_Accurate_L1 = 0,

		/**
		* Usually does the loads in fp16 but all calculations in fp32.
		* Usually not that fast than the method Dot_Fp16_Accurate_L1, this method doesn't care about overflow.
		*/
		Fp16_Accurate_L2 = 1,

		/**
		* Usually no accurate dispatching, do the loads and calculations all in fp16.
		* Usually the fastest method.
		*/
		Fp16_Accurate_L3 = 2,
	};
}
#endif


#ifdef _C_CONTEXT_
typedef struct __align__(8) DECX_Point2D_t
{
	int32_t x, y;
}DECX_Point2D;


typedef struct __align__(8) DECX_Point2D_f_t
{
	float x, y;
}DECX_Point2D_f;


typedef struct __align__(16) DECX_Point2D_d_t
{
	double x, y;
}DECX_Point2D_d;


typedef struct __align__(8) DECX_Complex_f_t
{
	float _real, _image;
}DECX_CPf;
#endif

#endif