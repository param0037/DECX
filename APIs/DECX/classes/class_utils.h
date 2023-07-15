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


	__align__(8) struct Point2D
	{
		int x, y;
		Point2D(const int _x, const int _y) { x = _x; y = _y; }
		Point2D() {}
	};


	__align__(16) struct Point3D
	{
		int x, y, z;
		Point3D(const int _x, const int _y, const int _z) { x = _x; y = _y; z = _z; }
		Point3D() {}
	};


	__align__(8) struct Point2D_f
	{
		float x, y;
		Point2D_f(const float _x, const float _y) { x = _x; y = _y; }
		Point2D_f() {}
	};


	__align__(16) struct Point2D_d
	{
		double x, y;
		Point2D_d(const double _x, const double _y) { x = _x; y = _y; }
		Point2D_d() {}
	};


	__align__(4) struct uchar4
	{
		uint8_t x, y, z, w;
	};

	namespace vis
	{
		__align__(4) struct Pixel
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



#endif