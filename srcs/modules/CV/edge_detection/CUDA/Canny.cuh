#pragma once

#include "../../core/basic.h"
#include "../../classes/matrix.h"


#define Sobel_Thr_x 16
#define Sobel_Thr_y 16
#define Sobel_FC_x 6
#define Sobel_FC_y 30


typedef struct __align__(16) Grad_info
{
	int2 gxy;
	float _g;		//~.x -> the mod of gradient, ~.y the degree weigth
}GI;



__global__ 
/*
 colume convolution by [1, 
						0, 
						-1]
 */
void Sobel_base_x(uchar*		src,
				  int*			base,
				  const dim3	dstDim,
				  const size_t	_pitch)
{
	short idx_loc = threadIdx.y / Sobel_Thr_y;
	short idy_loc = threadIdx.y % Sobel_Thr_y;

	short idx_glo_dst = idx_loc + blockIdx.x * Sobel_Thr_x;
	short idy_glo_dst = idy_loc + blockIdx.y * Sobel_Thr_y;

	bool is_in_dst = (idx_glo_dst < dstDim.y) && (idy_glo_dst < dstDim.x);
	// do not need to be transposed
	int tmp_base = idx_glo_dst * dstDim.x;
	int tmp_src = idx_glo_dst * _pitch;
	
	int __a,	// the upper one
		__b;	// the bottom one

	// use the colume vector to convolution
	if (is_in_dst) {
		// if is in the very top of the matrix
		if (idx_glo_dst == 0) {
			__a = src[idy_glo_dst];
			__b = src[tmp_src + _pitch + idy_glo_dst];
		}
		// if is in the very bottom of the matrix
		else if (idx_glo_dst == dstDim.y - 1) {
			__a = src[tmp_src - _pitch + idy_glo_dst];
			__b = src[(dstDim.y - 1) * _pitch + idy_glo_dst];
		}
		else {
			__a = src[tmp_src - _pitch + idy_glo_dst];
			__b = src[tmp_src + _pitch + idy_glo_dst];
		}

		int __base_dex = tmp_base + idy_glo_dst;
		base[__base_dex] = __a - __b;
	}
	else { return; }
}




__global__
// row convolution by [-1, 0, 1]
void Sobel_base_y(uchar*		src,
				  int*			base,
				  const dim3	dstDim,
				  const size_t	_pitch)
{
	short idx_loc = threadIdx.y / Sobel_Thr_y;
	short idy_loc = threadIdx.y % Sobel_Thr_y;

	short idx_glo_dst = idx_loc + blockIdx.x * Sobel_Thr_x;
	short idy_glo_dst = idy_loc + blockIdx.y * Sobel_Thr_y;

	int __a,	// the upper one
		__b;	// the bottom one

	bool is_in_dst = (idx_glo_dst < dstDim.y) && (idy_glo_dst < dstDim.x);

	// need to be transposed
	int tmp_base = idy_glo_dst * dstDim.y;
	int tmp_src = idx_glo_dst * _pitch;
	int tmp_dex = tmp_src + idy_glo_dst - 1;

	// use the row vector to convolution, and then transpose the result matrix
	if (is_in_dst) {
		if (idy_glo_dst == 0) {		// if is in the very left of the matrix
			__a = src[tmp_src];
			__b = src[tmp_dex + 2];

		}
		else if (idy_glo_dst == dstDim.x - 1) {		// // if is in the very right of the matrix
			__a = src[tmp_dex];
			__b = src[tmp_src + dstDim.x - 1];
		}
		else {
			__a = src[tmp_dex];
			__b = src[tmp_dex + 2];
		}
		GetValue(base, idy_glo_dst, idx_glo_dst, dstDim.y) = __b - __a;
	}
	else { return; }
}


// apply the convolution with row vector, and the way to 
// access the elements of base matrix is normal
// base_x and base_y are in the full dimensions, not in __Proc
/*
*			dstDim.x
*		________________
*		|			   |
*		|			   |	dstDim.y
*		|			   |	
*		----------------
*/
__global__
void Soble_x(int*			base,
			 GI*			dst,
			 const uint		__step,
			 const dim3		dstDim,
			 const dim3		__Proc,
			 const size_t	_pitch)
{
	short idx_loc = threadIdx.y / Sobel_Thr_y;
	short idy_loc = threadIdx.y % Sobel_Thr_y;

	short idx_glo = idx_loc + blockIdx.x * Sobel_Thr_x;
	short idy_glo = idy_loc + blockIdx.y * Sobel_Thr_y;

	short idx_glo_dst = __step * idx_glo;
	short idy_glo_dst = __step * idy_glo;

	bool is_in_dst = (idx_glo_dst < dstDim.y) && (idy_glo_dst < dstDim.x);

	int dex_tmp_dst = idx_glo_dst * dstDim.x + idy_glo_dst - 1;

	int __a,	// the left one
		__b,	// the midium one
		__c;	// the right one

	// consuder the area of __proc
	if (is_in_dst) {
		// no matter where, __b is located in the midium of the matrix
		__b = base[dex_tmp_dst + 1];

		if (idy_glo_dst == 0) {		// the very left of __Proc
			__a = __b;
			__c = base[dex_tmp_dst + 2];
		}
		else if (idy_glo_dst == dstDim.x - 1) {		// the very right of __Proc
			__c = __b;
			__a = base[dex_tmp_dst];
		}
		else {
			__a = base[dex_tmp_dst];
			__c = base[dex_tmp_dst + 2];
		}

		GetValue(dst, idx_glo, idy_glo, _pitch).gxy.x =
			__a + (__b << 1) + __c;
	}
	else { return; }
}



__global__
// row convolution by [1, 1, 1]
void Prewitt_x(int*				base,
			   GI*				dst,
			   const uint		__step,
			   const dim3		dstDim,
			   const dim3		__Proc,
			   const size_t		_pitch)
{
	short idx_loc = threadIdx.y / Sobel_Thr_y;
	short idy_loc = threadIdx.y % Sobel_Thr_y;

	short idx_glo = idx_loc + blockIdx.x * Sobel_Thr_x;
	short idy_glo = idy_loc + blockIdx.y * Sobel_Thr_y;

	short idx_glo_dst = __step * idx_glo;
	short idy_glo_dst = __step * idy_glo;

	bool is_in_dst = (idx_glo_dst < dstDim.y) && (idy_glo_dst < dstDim.x);

	int dex_tmp_dst = idx_glo_dst * dstDim.x + idy_glo_dst - 1;

	int __a,	// the left one
		__b,	// the midium one
		__c;	// the right one

	// consuder the area of __proc
	if (is_in_dst) {
		// no matter where, __b is located in the midium of the matrix
		__b = base[dex_tmp_dst + 1];

		if (idy_glo_dst == 0) {		// the very left of __Proc
			__a = __b;
			__c = base[dex_tmp_dst + 2];
		}
		else if (idy_glo_dst == dstDim.x - 1) {		// the very right of __Proc
			__c = __b;
			__a = base[dex_tmp_dst];
		}
		else {
			__a = base[dex_tmp_dst];
			__c = base[dex_tmp_dst + 2];
		}

		GetValue(dst, idx_glo, idy_glo, _pitch).gxy.x =
			__a + __b + __c;
	}
	else { return; }
}




// the way to access the source matrix need to be transposed
// base is in the full dimensions
// dst is in __Proc dimensions
/*
*		dstDim.y => idx_glo_dst
*		__________
*		|		 |
*		|		 |
*		|		 |	dstDim.x => idy_glo_dst
*		|		 |
*		|		 |
*		----------
*/
__global__
void Soble_y(int*			base,
			 GI*			dst,
			 const uint		__step,
			 const dim3		dstDim,
			 const dim3		__Proc,
			 const size_t	_pitch)
{
	short idx_loc = threadIdx.y / Sobel_Thr_y;
	short idy_loc = threadIdx.y % Sobel_Thr_y;

	short idx_glo = idx_loc + blockIdx.x * Sobel_Thr_x;
	short idy_glo = idy_loc + blockIdx.y * Sobel_Thr_y;

	short idx_glo_dst = __step * idx_glo;
	short idy_glo_dst = __step * idy_glo;

	bool is_in_dst = (idx_glo_dst < dstDim.y) && (idy_glo_dst < dstDim.x);

	int dex_tmp_dst = idy_glo_dst * dstDim.y + idx_glo_dst - 1;

	int __a,
		__b,
		__c;

	if (is_in_dst) {
		// no matter where, __b is always loacated in the midium of the matrix
		__b = base[dex_tmp_dst + 1];

		if (idx_glo_dst == 0) {
			__a = __b;
			__c = base[dex_tmp_dst + 2];
		}
		else if (idx_glo_dst == dstDim.y - 1) {
			__c = __b;
			__a = base[dex_tmp_dst];
		}
		else {
			__a = base[dex_tmp_dst];
			__c = base[dex_tmp_dst + 2];
		}

		GetValue(dst, idx_glo, idy_glo, _pitch).gxy.y =
			__a + (__b << 1) + __c;
	}
	else { return; }
}


__global__
void Prewitt_y(int*			base,
			   GI*			dst,
			   const uint	__step,
			   const dim3	dstDim,
			   const dim3	__Proc,
			   const size_t _pitch)
{
	short idx_loc = threadIdx.y / Sobel_Thr_y;
	short idy_loc = threadIdx.y % Sobel_Thr_y;

	short idx_glo = idx_loc + blockIdx.x * Sobel_Thr_x;
	short idy_glo = idy_loc + blockIdx.y * Sobel_Thr_y;

	short idx_glo_dst = __step * idx_glo;
	short idy_glo_dst = __step * idy_glo;

	bool is_in_dst = (idx_glo_dst < dstDim.y) && (idy_glo_dst < dstDim.x);

	int dex_tmp_dst = idy_glo_dst * dstDim.y + idx_glo_dst - 1;

	int __a,
		__b,
		__c;

	if (is_in_dst) {
		// no matter where, __b is always loacated in the midium of the matrix
		__b = base[dex_tmp_dst + 1];

		if (idx_glo_dst == 0) {
			__a = __b;
			__c = base[dex_tmp_dst + 2];
		}
		else if (idx_glo_dst == dstDim.y - 1) {
			__c = __b;
			__a = base[dex_tmp_dst];
		}
		else {
			__a = base[dex_tmp_dst];
			__c = base[dex_tmp_dst + 2];
		}

		GetValue(dst, idx_glo, idy_glo, _pitch).gxy.y =
			__a + __b + __c;
	}
	else { return; }
}



// apply the convolution with row vector, and the way to 
// access the elements of base matrix is normal
// base_x and base_y are in the full dimensions, not in __Proc
/*
*			dstDim.x
*		________________
*		|			   |
*		|			   |	dstDim.y
*		|			   |
*		----------------
*/
#if 1
__global__
void Scharr_x(int*			 base,
			  GI*			 dst,
			  const uint	 __step,
			  const dim3	 dstDim,
			  const dim3	 __Proc,
			  const size_t  _pitch)
{
	short idx_loc = threadIdx.y / Sobel_Thr_y;
	short idy_loc = threadIdx.y % Sobel_Thr_y;

	short idx_glo = idx_loc + blockIdx.x * Sobel_Thr_x;
	short idy_glo = idy_loc + blockIdx.y * Sobel_Thr_y;

	short idx_glo_dst = __step * idx_glo;
	short idy_glo_dst = __step * idy_glo;

	bool is_in_dst = (idx_glo_dst < dstDim.y) && (idy_glo_dst < dstDim.x);

	int dex_tmp_dst = idx_glo_dst * dstDim.x + idy_glo_dst - 1;

	int __a,	// the left one
		__b,	// the midium one
		__c;	// the right one

	// consuder the area of __proc
	if (is_in_dst) {
		// no matter where, __b is located in the midium of the matrix
		__b = base[dex_tmp_dst + 1];

		if (idy_glo_dst == 0) {		// the very left of __Proc
			__a = __b;
			__c = base[dex_tmp_dst + 2];
		}
		else if (idy_glo_dst == dstDim.x - 1) {		// the very right of __Proc
			__c = __b;
			__a = base[dex_tmp_dst];
		}
		else {
			__a = base[dex_tmp_dst];
			__c = base[dex_tmp_dst + 2];
		}

		GetValue(dst, idx_glo, idy_glo, _pitch).gxy.x =
			__a * 3 + (__b * 10) + __c * 3;
	}
	else { return; }
}



// the way to access the source matrix need to be transposed
// base is in the full dimensions
// dst is in __Proc dimensions
/*
*		dstDim.y => idx_glo_dst
*		__________
*		|		 |
*		|		 |
*		|		 |	dstDim.x => idy_glo_dst
*		|		 |
*		|		 |
*		----------
*/
__global__
void Scharr_y(int*			base,
			  GI*			dst,
			  const uint	__step,
			  const dim3	dstDim,
			  const dim3	__Proc,
			  const size_t _pitch)
{
	short idx_loc = threadIdx.y / Sobel_Thr_y;
	short idy_loc = threadIdx.y % Sobel_Thr_y;

	short idx_glo = idx_loc + blockIdx.x * Sobel_Thr_x;
	short idy_glo = idy_loc + blockIdx.y * Sobel_Thr_y;

	short idx_glo_dst = __step * idx_glo;
	short idy_glo_dst = __step * idy_glo;

	bool is_in_dst = (idx_glo_dst < dstDim.y) && (idy_glo_dst < dstDim.x);

	int dex_tmp_dst = idy_glo_dst * dstDim.y + idx_glo_dst - 1;

	int __a,
		__b,
		__c;

	if (is_in_dst) {
		// no matter where, __b is always loacated in the midium of the matrix
		__b = base[dex_tmp_dst + 1];

		if (idx_glo_dst == 0) {
			__a = __b;
			__c = base[dex_tmp_dst + 2];
		}
		else if (idx_glo_dst == dstDim.y - 1) {
			__c = __b;
			__a = base[dex_tmp_dst];
		}
		else {
			__a = base[dex_tmp_dst];
			__c = base[dex_tmp_dst + 2];
		}

		GetValue(dst, idx_glo, idy_glo, _pitch).gxy.y =
			__a * 3 + (__b * 10) + __c * 3;
	}
	else { return; }
}
#endif

#define _LOWER_ 0.414213
#define _HIGHER_ 2.414213
#define _I_LOWER_ -0.414213
#define _I_HIGHER_ -2.414213


#define _DIV_(__y, __x) (__y / __x)


#ifndef ZERO_STATE
#define ZERO_STATE(div) (div > -_LOWER_ && div < _LOWER_)
#endif

#ifndef ONE_STATE
#define ONE_STATE(div) (div > _LOWER_ && div < _HIGHER_)
#endif

#ifndef TWO_STATE
#define TWO_STATE(div) (div < _I_LOWER_ && div > _I_HIGHER_)
#endif

#define FAST_ATAN(div, res)  \
                    if( ZERO_STATE(div) ){res = 0;} \
                    else if( ONE_STATE(div) ){res = 1;} \
                    else if( TWO_STATE(div) ){res = 2;} \
                    else{res = 3;} 

#ifndef SQRT
#define SQRT(__x) (__x * __x)
#endif


// return an integer, which repersents the angles£¨ÀëÉ¢»¯½Ç¶ÈÖµ£©
// _Tp -> int, float, double, long, etc...
/* the point2D maps the 1D coordinates of two point in a 3x3 local area, e.g.
*		0, 1, 2
*		3, 4, 5
*		6, 7, 8
*/
// _g->x : is the maxima?
// _g->y : the mod of the gradient
// in order to save memery, the (x, y) angle info will be wrote on gxy(int2)
// since the previous datas is no longer used
#define use_atan
__device__
void fast_atan(GI *cur)
{
	//float _div_ = (float)__y / (float)__x;
	// the current angle
	float __ang = atanf(__fdividef((float)cur->gxy.x, (float)cur->gxy.y));

	// -22.5 deg < ang < 22.5 deg, calculate id_3' s and id_5' s and compare
	// the interpolation value 
	if (__ang > -0.3927 && __ang < 0.3927) {
		cur->gxy.x = 3;
		cur->gxy.y = 5;
	}
	// 22.5 deg < ang < 67.5 deg, calculate id_2' s and id_6' s and compare
	else if (__ang > 0.3927 && __ang < 1.1781) {
		cur->gxy.x = 2;
		cur->gxy.y = 6;
	}
	// -67.5 deg < ang < -22.5 deg, calculate id_0' s and id_8' s and compare
	else if (__ang < -0.3927 && __ang > -1.1781) {
		cur->gxy.x = 0;
		cur->gxy.y = 8;
	}
	// ang > 67.5 deg, calculate id_1' s and id_7' s and compare
	else {
		cur->gxy.x = 1;
		cur->gxy.y = 7;
	}
}

#ifdef use_atan
#undef use_atan
#endif

__inline __device__ __host__
int dev_abs(int __x)
{
	return (__x ^ (__x >> 31)) - (__x >> 31);
}


__inline __device__ __host__
float dev_abs(float __x)
{
	*((int*)&__x) &= 0x7fffffff;
	return __x;
}


__inline __device__ __host__
double dev_abs(double __x)
{
	float __base = (float)__x;
	*((int*)&__base) &= 0x7fffffff;
	return (double)__base;
}

__global__
void Sobel_SummingXY(GI*			src,
					 const uint		__step,
					 const dim3		__Proc,
					 const size_t	_pitch)
{
	short N_idx_loc = threadIdx.y / Sobel_Thr_y;
	short N_idy_loc = threadIdx.y % Sobel_Thr_y;

	short N_idx_glo = N_idx_loc + blockIdx.x * Sobel_Thr_x;
	short N_idy_glo = N_idy_loc + blockIdx.y * Sobel_Thr_y;
	
	bool is_in_proc = (N_idx_glo < __Proc.y) && (N_idy_glo < __Proc.x);

	int __base = N_idx_glo * _pitch + N_idy_glo;

	if (is_in_proc) {
		GI* _GIptr = &src[__base];
		// calcuate the gradient
		float _tmp_G = (float)dev_abs(_GIptr->gxy.x) + (float)dev_abs(_GIptr->gxy.y);
		_GIptr->_g = _tmp_G;
		// calculate the angle
		fast_atan(_GIptr);
	}
}


// src is in proc dimensions
// dst_H and dst_H are in full dimensions
__global__
void Sobel_Final_Calc(GI*				src,
					  uchar*			dst_H,
					  const float			_L_THR,
					  const float			_H_THR,
					  const uint		__step,
					  const dim3		dstDim,
					  const dim3		__Proc,
					  const size_t		_pitch)
{
	int lin_tid = threadIdx.y;

	int2 loc_ID;	// the current point in block domain
	loc_ID.x = lin_tid / Sobel_Thr_y;
	loc_ID.y = lin_tid % Sobel_Thr_y;

	int2 proc_ID;	// the current point in proc domain => src
	proc_ID.x = loc_ID.x + blockIdx.x * Sobel_Thr_x;
	proc_ID.y = loc_ID.y + blockIdx.y * Sobel_Thr_y;
	
	int2 dst_ID;	// the current point in dst domain => dst_H & dst_L
	dst_ID.x = proc_ID.x * __step;
	dst_ID.y = proc_ID.y * __step;

	bool is_in_dst = proc_ID.x < __Proc.y && proc_ID.y < __Proc.x;

	if (is_in_dst) {
		int base_dex = proc_ID.x * _pitch + proc_ID.y;
		int lin_dst_dex = dst_ID.x * dstDim.x + dst_ID.y;

		GI* _GIptr = &(src[base_dex]);

		float __near[9];

		int roll_dex_base = base_dex - _pitch - 1;
		int chip_dex = 0;
		/*start to load the nearby gradient info into the local chip array
		* non-maxima suppression and lag edges tracing will be operated here
		*/
#pragma unroll 3
		for (int x = proc_ID.x - 1; x < proc_ID.x + 2; ++x) {
			if (x < 0 || x > __Proc.y - 1) {
				__near[chip_dex] = 0.f;
				++chip_dex;

				__near[chip_dex] = 0.f;
				++chip_dex;

				__near[chip_dex] = 0.f;
				++chip_dex;
				roll_dex_base += 3;
			}
			else {
#pragma unroll 3
				for (int y = proc_ID.y - 1; y < proc_ID.y + 2; ++y) {
					if (y < 0 || y > __Proc.x - 1) {
						__near[chip_dex] = 0.f;
					}
					else {
						__near[chip_dex] = src[roll_dex_base]._g;
					}
					++chip_dex;
					++roll_dex_base;
				}
			}
			roll_dex_base += _pitch - 3;
		}
		// loading is finished

		// non-maxima suppression
		bool is_max = _GIptr->_g > __near[_GIptr->gxy.x] && _GIptr->_g > __near[_GIptr->gxy.y];

		bool is_strong = _GIptr->_g > _H_THR;
		bool is_weak = !is_strong && (_GIptr->_g > _L_THR);

		uchar res = 255 * is_max;

		if (is_strong) {	// is a strong edge
			dst_H[lin_dst_dex] = res;
		}
		// lag edges tracing
		if (is_max && is_weak) {

			bool is_255 = false;
			
			for (int i = 0; i < 9; ++i) {
				if (__near[i] > _L_THR) {
					is_255 = true;
					break;
				}
			}
			dst_H[lin_dst_dex] = 255 * is_255;
		}
	}
	else { return; }
}


// the grid will cover dst matrix(the bigger one)
__global__
void img_make_border_mirr(uchar* src,
	uchar* dst,
	int radius,
	int2 srcDim,
	int2 dstDim)
{
	int2 loc_ID;
	loc_ID.x = threadIdx.x;
	loc_ID.y = threadIdx.y;

	int2 dst_ID;
	dst_ID.x = loc_ID.x + blockIdx.x * blockDim.x;
	dst_ID.y = loc_ID.y + blockIdx.y * blockDim.y;

	int2 src_ID;
	src_ID.x = dst_ID.x - radius;
	src_ID.y = dst_ID.y - radius;

	bool is_in_src = (src_ID.x > -1 && src_ID.x < srcDim.y) &&
		(src_ID.y > -1 && src_ID.y < srcDim.x);

	int src_base = src_ID.x * srcDim.x + src_ID.y;
	int dst_base = dst_ID.x * dstDim.x + dst_ID.y;

	// if the thread is in src, then copy the values normally
	if (is_in_src) {
		dst[dst_base] = src[src_base];
	}
	else {
		int2 new_src_dex;
		int2 lag_bounding;
		lag_bounding.x = srcDim.x - 1;
		lag_bounding.y = srcDim.y - 1;

		if (src_ID.x < 0) {
			new_src_dex.x = abs(src_ID.x);
		}
		if (src_ID.x > lag_bounding.y) {
			new_src_dex.x = (lag_bounding.y) << 1 - src_ID.x;
		}

		if (src_ID.y < 0) {
			new_src_dex.y = abs(src_ID.y);
		}
		if (src_ID.y > lag_bounding.x) {
			new_src_dex.y = (lag_bounding.x) << 1 - src_ID.y;
		}

		dst[dst_base] = GetValue(src, new_src_dex.x, new_src_dex.y, srcDim.x);
	}
}