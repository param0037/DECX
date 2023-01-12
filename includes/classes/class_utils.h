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
		Point2D(int _x, int _y) { x = _x; y = _y; }
		Point2D() {}
	};


	__align__(8) struct Point2D_f
	{
		float x, y;
		Point2D_f(const float _x, const float _y) { x = _x; y = _y; }
		Point2D_f() {}
	};
}

#endif
