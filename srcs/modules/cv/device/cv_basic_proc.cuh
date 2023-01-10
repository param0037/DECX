#pragma once
#include "..\..\basic\basic.h"
#include "..\..\algorithm\device\cu_math_functions.cuh"



#define Gauss_threads_x 8
#define Gauss_threads_y 8

__global__
void Gaussian_vec_Gen(float* G_vec, 
	const double __sigma, 
	const uint __radius,
	const uint __len)
{
	int idx = threadIdx.x;
	double x_axis = (double)(idx - (int)__radius);

	__shared__ float tmp_G_vec[_MTPB_];

	float res = 0.3989422 * dev_exp(-(x_axis * x_axis) / (2 * __sigma * __sigma)) / __sigma;

	tmp_G_vec[idx] = res;
	//G_vec[idx] = res;

	__threadfence();

	double _res = 0;
	for (int i = 0; i < __radius; ++i) {
		_res += tmp_G_vec[i];
	}
	_res *= 2;
	_res += tmp_G_vec[__radius];
	
	G_vec[idx] = __fdividef(res, _res);
}




__global__
// dstDim [~.x : width, ~.y : height]
void Gaussian_blur_hor_(uchar*		src,
					   float*		G_vec,
					   uchar*		mid,
					   const uint	G_len,
					   const uint	__radius,
					   const dim3	dstDim,
					   const dim3	srcDim)
{
	uint idx_loc = threadIdx.x;
	uint idy_loc = threadIdx.y;
	uint _linear_tid = idx_loc * blockDim.y + idy_loc;

	uint idx_glo = idx_loc + blockIdx.x * blockDim.x;
	uint idy_glo = idy_loc + blockIdx.y * blockDim.y;

	uint idx_glo_src = idx_glo;
	uint idy_glo_src = idy_glo + __radius;

	// allocate the shared memory to store the kernel vector
	__shared__ float tmp_vec[_MTPB_];

	// load the kernel vector to the shared mamory
	bool is_in_ker = _linear_tid < G_len;
	bool is_in_dst = (idx_glo < srcDim.y) && (idy_glo < dstDim.x);

	uint tmp_Wsrc = srcDim.x * is_in_dst;
	uint tmp_Hsrc = srcDim.y * is_in_dst;

	// load the kernel
	if (is_in_ker) {
		tmp_vec[_linear_tid] = G_vec[_linear_tid];
	}

	__syncthreads();

	if (is_in_dst) {

		float res = 0;
		//uint x_offset = idx_glo_src * tmp_Wsrc + idy_glo_src;
		uint x_offset = idx_glo_src * srcDim.x + idy_glo_src;
		// first apply the 1D convolution horizentally
		for (int x = x_offset - __radius, _x = 0; _x < G_len; ++x, ++_x)
		{
			res = fma((float)(src[x]), (float)tmp_vec[_x], res);
		}

		GetValue(mid, idy_glo, idx_glo_src, tmp_Hsrc) = (uchar)res;
	}
	else { return; }
}


__global__
// dstDim [~.x : width, ~.y : height]
void Gaussian_blur_hor(uchar*		src,
					   float*		G_vec,
					   uchar*		mid,
					   const uint	G_len,
					   const uint	__radius,
					   const dim3	dstDim,
					   const dim3	srcDim)
{
	uint idx_loc = threadIdx.x;
	uint idy_loc = threadIdx.y;
	uint _linear_tid = idx_loc * blockDim.y + idy_loc;

	uint idx_glo = idx_loc + blockIdx.x * blockDim.x;
	uint idy_glo = idy_loc + blockIdx.y * blockDim.y;

	uint idx_glo_src = idx_glo;
	uint idy_glo_src = idy_glo + __radius;

	// load the kernel vector to the shared mamory
	bool is_in_ker = _linear_tid < G_len;
	bool is_in_dst = (idx_glo < srcDim.y) && (idy_glo < dstDim.x);

	uint tmp_Wsrc = srcDim.x * is_in_dst;
	uint tmp_Hsrc = srcDim.y * is_in_dst;

	__syncthreads();

	if (is_in_dst) {

		float res = 0;
		uint x_offset = idx_glo_src * srcDim.x + idy_glo_src;
		
		for (int x = x_offset - __radius, _x = 0; _x < G_len; ++x, ++_x){
			res = fma((float)(src[x]), (float)((uchar*)Const_Mem)[_x], res);
		}

		GetValue(mid, idy_glo, idx_glo_src, tmp_Hsrc) = (uchar)res;
	}
	else { return; }
}


//
//__global__
//// dstDim [~.x : width, ~.y : height]
//void Gaussian_blur_hor3D(uchar4* src,
//	float* G_vec,
//	uchar4* mid,
//	const uint G_len,
//	const uint half_ker,
//	const dim3 dstDim,
//	const dim3 srcDim,
//	const uint __radius)
//{
//	uint idx_loc = threadIdx.x;
//	uint idy_loc = threadIdx.y;
//	uint _linear_tid = idx_loc * blockDim.y + idy_loc;
//
//	uint idx_glo = idx_loc + blockIdx.x * blockDim.x;
//	uint idy_glo = idy_loc + blockIdx.y * blockDim.y;
//
//	uint idx_glo_src = idx_glo;
//	uint idy_glo_src = idy_glo + half_ker;
//
//	// allocate the shared memory to store the kernel vector
//	__shared__ float tmp_vec[_MTPB_];
//
//	// load the kernel vector to the shared mamory
//	bool is_in_ker = _linear_tid < G_len;
//	bool is_in_dst = (idx_glo < srcDim.y) && (idy_glo < dstDim.x);
//
//	uint tmp_Wsrc = srcDim.x * is_in_dst;
//	uint tmp_Hsrc = srcDim.y * is_in_dst;
//
//	// load the kernel
//	if (is_in_ker) {
//		tmp_vec[_linear_tid] = G_vec[_linear_tid];
//	}
//	
//	__syncthreads();
//
//	if (is_in_dst) {
//
//		float res_x = 0, res_y = 0, res_z = 0;
//		//uint x_offset = idx_glo_src * tmp_Wsrc + idy_glo_src;
//		uint x_offset = idx_glo_src * srcDim.x + idy_glo_src;
//		// first apply the 1D convolution horizentally
//		for (int x = x_offset - half_ker, _x = 0; _x < G_len; ++x, ++_x) 
//		{
//			uchar4* src_Ptr = &src[x];
//			res_x = fma((float)(src_Ptr->x), (float)tmp_vec[_x], res_x);
//			res_y = fma((float)(src_Ptr->y), (float)tmp_vec[_x], res_y);
//			res_z = fma((float)(src_Ptr->z), (float)tmp_vec[_x], res_z);
//		}
//
//		uchar4* dst_ptr = &GetValue(mid, idy_glo, idx_glo_src, tmp_Hsrc);
//		dst_ptr->x = (uchar)res_x;
//		dst_ptr->y = (uchar)res_y;
//		dst_ptr->z = (uchar)res_z;
//	}
//	else { return; }
//}




__global__
// dstDim [~.x : width, ~.y : height]
// in horizental
void Gaussian_blur_hor_border(uchar* src,
	float* G_vec,
	uchar* mid,
	const uint G_len,
	const dim3 srcDim,
	const uint __radius,
	const float border)
{
	uint idx_loc = threadIdx.x;
	uint idy_loc = threadIdx.y;
	uint _linear_tid = idx_loc * blockDim.y + idy_loc;

	uint idx_glo = idx_loc + blockIdx.x * blockDim.x;
	uint idy_glo = idy_loc + blockIdx.y * blockDim.y;

	// allocate the shared memory to store the kernel vector
	__shared__ float tmp_vec[_MTPB_];

	// load the kernel vector to the shared mamory
	bool is_in_ker = _linear_tid < G_len;
	bool is_in_dst = (idx_glo < srcDim.y) && (idy_glo < srcDim.x);
	bool consider_edge_L = idy_glo < __radius;

	int R_edge = srcDim.x - __radius - 1;
	int Bound_offset = srcDim.x - 1;

	bool consider_edge_R = idy_glo > R_edge;

	// load the kernel
	if (is_in_ker) {
		tmp_vec[_linear_tid] = G_vec[_linear_tid];
	}
	
	__syncthreads();
	
	if (is_in_dst) {
		float res = 0;
		int x_offset = idx_glo * srcDim.x;

		// first apply the 1D convolution horizentally
		if (consider_edge_L) {
			float tmp_res = 0;
			for (int x = idy_glo - __radius, _x = 0; _x < G_len; ++x, ++_x) {
				if (x < 0) {
					tmp_res += tmp_vec[_x];
				}
				else {
					res = fma((float)src[x_offset + x], tmp_vec[_x], res);
				}
			}
			res = fma(tmp_res, border, res);
		}
		else if (consider_edge_R) {
			float tmp_res = 0;
			for (int x = idy_glo - __radius, _x = 0; _x < G_len; ++x, ++_x) {
				if (x > Bound_offset) {
					res += tmp_vec[_x];
				}
				else {
					res = fma((float)src[x_offset + x], tmp_vec[_x], res);
				}
			}
			res = fma(tmp_res, border, res);
		}
		else {
			int x_offset_lin = x_offset + idy_glo;
			for (int x = x_offset_lin - __radius, _x = 0; _x < G_len; ++x, ++_x) {
				res = fma((float)src[x], tmp_vec[_x], res);
			}
		}
		GetValue(mid, idy_glo, idx_glo, srcDim.y) = (uchar)res;
	}
	else { return; }
}



__global__
// dstDim [~.x : width, ~.y : height]
// in horizental
void Gaussian_blur_hor_border3D(uchar4* src,
	float* G_vec,
	uchar4* mid,
	const uint G_len,
	const dim3 srcDim,
	const uint __radius,
	const float border)
{
	uint idx_loc = threadIdx.x;
	uint idy_loc = threadIdx.y;
	uint _linear_tid = idx_loc * blockDim.y + idy_loc;

	uint idx_glo = idx_loc + blockIdx.x * blockDim.x;
	uint idy_glo = idy_loc + blockIdx.y * blockDim.y;

	// allocate the shared memory to store the kernel vector
	__shared__ float tmp_vec[_MTPB_];

	// load the kernel vector to the shared mamory
	bool is_in_ker = _linear_tid < G_len;
	bool is_in_dst = (idx_glo < srcDim.y) && (idy_glo < srcDim.x);
	bool consider_edge_L = idy_glo < __radius;

	int R_edge = srcDim.x - __radius - 1;
	int Bound_offset = srcDim.x - 1;

	bool consider_edge_R = idy_glo > R_edge;

	// load the kernel
	if (is_in_ker) {
		tmp_vec[_linear_tid] = G_vec[_linear_tid];
	}

	__syncthreads();

	if (is_in_dst) {
		float3 res;
		res.x = 0;
		res.y = 0;
		res.z = 0;

		int x_offset = idx_glo * srcDim.x;

		// first apply the 1D convolution horizentally
		if (consider_edge_L) {
			float tmp_res = 0;
			for (int x = idy_glo - __radius, _x = 0; _x < G_len; ++x, ++_x) {
				if (x < 0) {
					tmp_res += tmp_vec[_x];
				}
				else {
					uchar4* _ptr = &src[x_offset + x];
					res.x = fma((float)(_ptr->x), tmp_vec[_x], res.x);
					res.y = fma((float)(_ptr->y), tmp_vec[_x], res.y);
					res.z = fma((float)(_ptr->z), tmp_vec[_x], res.z);
				}
			}
			res.x = fma(tmp_res, border, res.x);
			res.y = fma(tmp_res, border, res.y);
			res.z = fma(tmp_res, border, res.z);
		}
		else if (consider_edge_R) {
			float tmp_res = 0;
			for (int x = idy_glo - __radius, _x = 0; _x < G_len; ++x, ++_x) {
				if (x > Bound_offset) {
					tmp_res += tmp_vec[_x];
				}
				else {
					uchar4* _ptr = &src[x_offset + x];
					res.x = fma((float)(_ptr->x), tmp_vec[_x], res.x);
					res.y = fma((float)(_ptr->y), tmp_vec[_x], res.y);
					res.z = fma((float)(_ptr->z), tmp_vec[_x], res.z);
				}
			}
			res.x = fma(tmp_res, border, res.x);
			res.y = fma(tmp_res, border, res.y);
			res.z = fma(tmp_res, border, res.z);
		}
		else {
			int x_offset_lin = x_offset + idy_glo;
			for (int x = x_offset_lin - __radius, _x = 0; _x < G_len; ++x, ++_x) {
				uchar4* _ptr = &src[x];
				res.x = fma((float)(_ptr->x), tmp_vec[_x], res.x);
				res.y = fma((float)(_ptr->y), tmp_vec[_x], res.y);
				res.z = fma((float)(_ptr->z), tmp_vec[_x], res.z);
			}
		}

		uchar4* dst_ptr = &GetValue(mid, idy_glo, idx_glo, srcDim.y);
		dst_ptr->x = (uchar)res.x;
		dst_ptr->y = (uchar)res.y;
		dst_ptr->z = (uchar)res.z;
	}
	else { return; }
}




__global__
// dstDim [~.x : width, ~.y : height]
// in horizental
void Gaussian_blur_hor_mirror(uchar* src,
	float* G_vec,
	uchar* mid,
	const uint G_len,
	const dim3 srcDim,
	const uint __radius)
{
	uint idx_loc = threadIdx.x;
	uint idy_loc = threadIdx.y;
	uint _linear_tid = idx_loc * blockDim.y + idy_loc;

	uint idx_glo = idx_loc + blockIdx.x * blockDim.x;
	uint idy_glo = idy_loc + blockIdx.y * blockDim.y;

	// allocate the shared memory to store the kernel vector
	__shared__ float tmp_vec[_MTPB_];

	// load the kernel vector to the shared mamory
	bool is_in_ker = _linear_tid < G_len;
	bool is_in_dst = (idx_glo < srcDim.y) && (idy_glo < srcDim.x);
	bool consider_edge_L = idy_glo < __radius;

	int R_edge = srcDim.x - __radius - 1;
	int Bound_offset = srcDim.x - 1;

	bool consider_edge_R = idy_glo > R_edge;

	// load the kernel
	if (is_in_ker) {
		tmp_vec[_linear_tid] = G_vec[_linear_tid];
	}

	__syncthreads();

	if (is_in_dst) {
		float res = 0;
		int x_offset = idx_glo * srcDim.x;

		// first apply the 1D convolution horizentally
		if (consider_edge_L) {
			float tmp_res = 0;
			for (int x = idy_glo - __radius, _x = 0; _x < G_len; ++x, ++_x) {
				if (x < 0) {
					res = fma((float)src[x_offset + abs(x)], tmp_vec[_x], res);
				}
				else {
					res = fma((float)src[x_offset + x], tmp_vec[_x], res);
				}
			}
		}
		else if (consider_edge_R) {
			int x_offset_more = x_offset + (srcDim.x << 1) - 1;
			for (int x = idy_glo - __radius, _x = 0; _x < G_len; ++x, ++_x) {
				if (x > Bound_offset) {
					res = fma((float)src[x_offset_more - x], tmp_vec[_x], res);
				}
				else {
					res = fma((float)src[x_offset + x], tmp_vec[_x], res);
				}
			}
		}
		else {
			int x_offset_lin = x_offset + idy_glo;
			for (int x = x_offset_lin - __radius, _x = 0; _x < G_len; ++x, ++_x) {
				res = fma((float)src[x], tmp_vec[_x], res);
			}
		}

		GetValue(mid, idy_glo, idx_glo, srcDim.y) = (uchar)res;
	}
	else { return; }
}



__global__
// dstDim [~.x : width, ~.y : height]
// in horizental
void Gaussian_blur_hor_mirror3D(uchar4* src,
	float* G_vec,
	uchar4* mid,
	const uint G_len,
	const dim3 srcDim,
	const uint __radius)
{
	uint idx_loc = threadIdx.x;
	uint idy_loc = threadIdx.y;
	uint _linear_tid = idx_loc * blockDim.y + idy_loc;

	uint idx_glo = idx_loc + blockIdx.x * blockDim.x;
	uint idy_glo = idy_loc + blockIdx.y * blockDim.y;

	// allocate the shared memory to store the kernel vector
	__shared__ float tmp_vec[_MTPB_];

	// load the kernel vector to the shared mamory
	bool is_in_ker = _linear_tid < G_len;
	bool is_in_dst = (idx_glo < srcDim.y) && (idy_glo < srcDim.x);
	bool consider_edge_L = idy_glo < __radius;

	int R_edge = srcDim.x - __radius - 1;
	int Bound_offset = srcDim.x - 1;

	bool consider_edge_R = idy_glo > R_edge;

	// load the kernel
	if (is_in_ker) {
		tmp_vec[_linear_tid] = G_vec[_linear_tid];
	}

	__syncthreads();

	if (is_in_dst) {
		float3 res;
		res.x = 0;
		res.y = 0;
		res.z = 0;

		int x_offset = idx_glo * srcDim.x;

		// first apply the 1D convolution horizentally
		if (consider_edge_L) {
			float tmp_res = 0;
			for (int x = idy_glo - __radius, _x = 0; _x < G_len; ++x, ++_x) {
				if (x < 0) {
					uchar4* _ptr = &src[x_offset + abs(x)];
					res.x = fma((float)(_ptr->x), tmp_vec[_x], res.x);
					res.y = fma((float)(_ptr->y), tmp_vec[_x], res.y);
					res.z = fma((float)(_ptr->z), tmp_vec[_x], res.z);
				}
				else {
					uchar4* _ptr = &src[x_offset + x];
					res.x = fma((float)(_ptr->x), tmp_vec[_x], res.x);
					res.y = fma((float)(_ptr->y), tmp_vec[_x], res.y);
					res.z = fma((float)(_ptr->z), tmp_vec[_x], res.z);
				}
			}
		}
		else if (consider_edge_R) {
			int x_offset_more = x_offset + (srcDim.x << 1) - 1;
			for (int x = idy_glo - __radius, _x = 0; _x < G_len; ++x, ++_x) {
				if (x > Bound_offset) {
					uchar4* _ptr = &src[x_offset_more - x];
					res.x = fma((float)(_ptr->x), tmp_vec[_x], res.x);
					res.y = fma((float)(_ptr->y), tmp_vec[_x], res.y);
					res.z = fma((float)(_ptr->z), tmp_vec[_x], res.z);
				}
				else {
					uchar4* _ptr = &src[x_offset + x];
					res.x = fma((float)(_ptr->x), tmp_vec[_x], res.x);
					res.y = fma((float)(_ptr->y), tmp_vec[_x], res.y);
					res.z = fma((float)(_ptr->z), tmp_vec[_x], res.z);
				}
			}
		}
		else {
			int x_offset_lin = x_offset + idy_glo;
			for (int x = x_offset_lin - __radius, _x = 0; _x < G_len; ++x, ++_x) {
				uchar4* _ptr = &src[x];
				res.x = fma((float)(_ptr->x), tmp_vec[_x], res.x);
				res.y = fma((float)(_ptr->y), tmp_vec[_x], res.y);
				res.z = fma((float)(_ptr->z), tmp_vec[_x], res.z);
			}
		}
		uchar4* dst_ptr = &GetValue(mid, idy_glo, idx_glo, srcDim.y);

		dst_ptr->x = (uchar)res.x;
		dst_ptr->y = (uchar)res.y;
		dst_ptr->z = (uchar)res.z;
	}
	else { return; }
}



__global__
void Gaussian_blur_ver(uchar* mid,
	float* G_vec,
	uchar* dst,
	const uint G_len,
	const uint half_ker,
	const dim3 dstDim,
	const dim3 srcDim)
{
	uint idx_loc = threadIdx.x;
	uint idy_loc = threadIdx.y;
	uint _linear_tid = idx_loc * blockDim.y + idy_loc;

	uint idx_glo = idx_loc + blockIdx.x * blockDim.x;
	uint idy_glo = idy_loc + blockIdx.y * blockDim.y;

	uint idx_glo_src = idx_glo + half_ker;
	uint idy_glo_src = idy_glo;

	// allocate the shared memory to store the kernel vector
	__shared__ float tmp_vec[_MTPB_];

	// load the kernel vector to the shared mamory
	bool is_in_ker = _linear_tid < G_len;
	bool is_in_dst = (idx_glo < dstDim.y) && (idy_glo < dstDim.x);

	uint tmp_Wsrc = srcDim.y * is_in_dst;
	// load the kernel
	if (is_in_ker) {
		tmp_vec[_linear_tid] = G_vec[_linear_tid * is_in_ker];
	}

	__syncthreads();

	float res = 0;
	uint dex_tmp = idy_glo_src * tmp_Wsrc + idx_glo_src;

	if (is_in_dst) {
		// and apply the 1D convolution vertically
		for (int y = dex_tmp - half_ker, _y = 0; _y < G_len; ++y, ++_y) {
			res = fma((float)(mid[y]), (float)tmp_vec[_y], res);
		}

		GetValue(dst, idx_glo, idy_glo, dstDim.x) = (uchar)res;
	}
	else { return; }
}


//
//__global__
//void Gaussian_blur_ver3D(uchar4* mid,
//	float* G_vec,
//	uchar4* dst,
//	const uint G_len,
//	const uint half_ker,
//	const dim3 dstDim,
//	const dim3 srcDim)
//{
//	uint idx_loc = threadIdx.x;
//	uint idy_loc = threadIdx.y;
//	uint _linear_tid = idx_loc * blockDim.y + idy_loc;
//
//	uint idx_glo = idx_loc + blockIdx.x * blockDim.x;
//	uint idy_glo = idy_loc + blockIdx.y * blockDim.y;
//
//	uint idx_glo_src = idx_glo + half_ker;
//	uint idy_glo_src = idy_glo;
//
//	// allocate the shared memory to store the kernel vector
//	__shared__ float tmp_vec[_MTPB_];
//
//	// load the kernel vector to the shared mamory
//	bool is_in_ker = _linear_tid < G_len;
//	bool is_in_dst = (idx_glo < dstDim.y) && (idy_glo < dstDim.x);
//
//	uint tmp_Wsrc = srcDim.y * is_in_dst;
//	// load the kernel
//	if (is_in_ker) {
//		tmp_vec[_linear_tid] = G_vec[_linear_tid * is_in_ker];
//	}
//
//	__syncthreads();
//
//	float4 res;
//	uint dex_tmp = idy_glo_src * tmp_Wsrc + idx_glo_src;
//
//	if (is_in_dst) {
//		// and apply the 1D convolution vertically
//		for (int y = dex_tmp - half_ker, _y = 0; _y < G_len; ++y, ++_y) {
//			uchar4* src_Ptr = mid + y;
//			res.x = fma((float)(src_Ptr->x), (float)tmp_vec[_y], res.x);
//			res.y = fma((float)(src_Ptr->y), (float)tmp_vec[_y], res.y);
//			res.z = fma((float)(src_Ptr->z), (float)tmp_vec[_y], res.z);
//		}
//
//		uchar4* dst_ptr = &GetValue(dst, idx_glo, idy_glo, dstDim.x);
//		dst_ptr->x = (uchar)res.x;
//		dst_ptr->y = (uchar)res.y;
//		dst_ptr->z = (uchar)res.z;
//		dst_ptr->w = 255;
//	}
//	else { return; }
//}



__global__
void Gaussian_blur_ver_border(uchar* mid,
	float* G_vec,
	uchar* dst,
	const uint G_len,
	const dim3 srcDim,
	const int __radius,
	const float border)
{
	uint idx_loc = threadIdx.x;
	uint idy_loc = threadIdx.y;
	uint _linear_tid = idx_loc * blockDim.y + idy_loc;

	uint idx_glo = idx_loc + blockIdx.x * blockDim.x;
	uint idy_glo = idy_loc + blockIdx.y * blockDim.y;

	bool consider_border_L = idx_glo < __radius;

	int Bottom_edge = srcDim.y - __radius - 1;
	int bound_offset = srcDim.y - 1;

	bool consider_border_R = idx_glo > Bottom_edge;

	// allocate the shared memory to store the kernel vector
	__shared__ float tmp_vec[_MTPB_];

	// load the kernel vector to the shared mamory
	bool is_in_ker = _linear_tid < G_len;
	bool is_in_dst = (idx_glo < srcDim.y) && (idy_glo < srcDim.x);

	// load the kernel
	if (is_in_ker) {
		tmp_vec[_linear_tid] = G_vec[_linear_tid * is_in_ker];
	}

	__syncthreads();

	if (is_in_dst) {
		float res = 0;
		int tmp_dex = idy_glo * srcDim.y;

		if (consider_border_L) {
			float tmp_res = 0;
			for (int y = idx_glo - __radius, _y = 0; _y < G_len; ++y, ++_y) {
				if (y < 0) {
					tmp_res += tmp_vec[_y];
				}
				else {
					res = fma((float)mid[tmp_dex + y], (float)tmp_vec[_y], res);
				}
			}
			res = fma(tmp_res, (float)border, res);
		}
		else if (consider_border_R) {
			float tmp_res = 0;
			for (int y = idx_glo - __radius, _y = 0; _y < G_len; ++y, ++_y) {
				if (y > bound_offset) {
					tmp_res += tmp_vec[_y];
				}
				else {
					res = fma((float)mid[tmp_dex + y], tmp_vec[_y], res);
				}
			}
			res = fma(tmp_res, (float)border, res);
		}
		else {
			int tmp_dex_lin = tmp_dex + idx_glo;
			for (int y = tmp_dex_lin - __radius, _y = 0; _y < G_len; ++y, ++_y) {
				res = fma((float)mid[y], tmp_vec[_y], res);
			}
		}

		GetValue(dst, idx_glo, idy_glo, srcDim.x) = (uchar)res;
	}
	else { return; }
}



__global__
void Gaussian_blur_ver_border3D(uchar4* mid,
	float* G_vec,
	uchar4* dst,
	const uint G_len,
	const dim3 srcDim,
	const int __radius,
	const float border)
{
	uint idx_loc = threadIdx.x;
	uint idy_loc = threadIdx.y;
	uint _linear_tid = idx_loc * blockDim.y + idy_loc;

	uint idx_glo = idx_loc + blockIdx.x * blockDim.x;
	uint idy_glo = idy_loc + blockIdx.y * blockDim.y;

	bool consider_border_L = idx_glo < __radius;

	int Bottom_edge = srcDim.y - __radius - 1;
	int bound_offset = srcDim.y - 1;

	bool consider_border_R = idx_glo > Bottom_edge;

	// allocate the shared memory to store the kernel vector
	__shared__ float tmp_vec[_MTPB_];

	// load the kernel vector to the shared mamory
	bool is_in_ker = _linear_tid < G_len;
	bool is_in_dst = (idx_glo < srcDim.y) && (idy_glo < srcDim.x);

	// load the kernel
	if (is_in_ker) {
		tmp_vec[_linear_tid] = G_vec[_linear_tid * is_in_ker];
	}

	__syncthreads();

	if (is_in_dst) {
		float3 res;
		res.x = 0;
		res.y = 0;
		res.z = 0;

		int tmp_dex = idy_glo * srcDim.y;

		if (consider_border_L) {
			float tmp_res = 0;
			for (int y = idx_glo - __radius, _y = 0; _y < G_len; ++y, ++_y) {
				if (y < 0) {
					tmp_res += tmp_vec[_y];
				}
				else {
					uchar4* _ptr = &mid[tmp_dex + y];
					res.x = fma((float)(_ptr->x), tmp_vec[_y], res.x);
					res.y = fma((float)(_ptr->y), tmp_vec[_y], res.y);
					res.z = fma((float)(_ptr->z), tmp_vec[_y], res.z);
				}
			}
			res.x = fma(tmp_res, border, res.x);
			res.y = fma(tmp_res, border, res.y);
			res.z = fma(tmp_res, border, res.z);
		}
		else if (consider_border_R) {
			float tmp_res = 0;
			for (int y = idx_glo - __radius, _y = 0; _y < G_len; ++y, ++_y) {
				if (y > bound_offset) {
					tmp_res += tmp_vec[_y];
				}
				else {
					uchar4* _ptr = &mid[tmp_dex + y];
					res.x = fma((float)(_ptr->x), tmp_vec[_y], res.x);
					res.y = fma((float)(_ptr->y), tmp_vec[_y], res.y);
					res.z = fma((float)(_ptr->z), tmp_vec[_y], res.z);
				}
			}
			res.x = fma(tmp_res, (float)border, res.x);
			res.y = fma(tmp_res, (float)border, res.y);
			res.z = fma(tmp_res, (float)border, res.z);
		}
		else {
			int tmp_dex_lin = tmp_dex + idx_glo;
			for (int y = tmp_dex_lin - __radius, _y = 0; _y < G_len; ++y, ++_y) {
				uchar4* _ptr = &mid[y];
				res.x = fma((float)(_ptr->x), tmp_vec[_y], res.x);
				res.y = fma((float)(_ptr->y), tmp_vec[_y], res.y);
				res.z = fma((float)(_ptr->z), tmp_vec[_y], res.z);
			}
		}

		uchar4* dst_ptr = &GetValue(dst, idx_glo, idy_glo, srcDim.x);
		dst_ptr->x = (uchar)res.x;
		dst_ptr->y = (uchar)res.y;
		dst_ptr->z = (uchar)res.z;
		dst_ptr->w = 255;
	}
	else { return; }
}




__global__
void Gaussian_blur_ver_mirror(uchar* mid,
	float* G_vec,
	uchar* dst,
	const uint G_len,
	const dim3 srcDim,
	const int __radius)
{
	uint idx_loc = threadIdx.x;
	uint idy_loc = threadIdx.y;
	uint _linear_tid = idx_loc * blockDim.y + idy_loc;

	uint idx_glo = idx_loc + blockIdx.x * blockDim.x;
	uint idy_glo = idy_loc + blockIdx.y * blockDim.y;

	bool consider_border_L = idx_glo < __radius;

	int Bottom_edge = srcDim.y - __radius - 1;
	int bound_offset = srcDim.y - 1;

	bool consider_border_R = idx_glo > Bottom_edge;

	// allocate the shared memory to store the kernel vector
	__shared__ float tmp_vec[_MTPB_];

	// load the kernel vector to the shared mamory
	bool is_in_ker = _linear_tid < G_len;
	bool is_in_dst = (idx_glo < srcDim.y) && (idy_glo < srcDim.x);

	// load the kernel
	if (is_in_ker) {
		tmp_vec[_linear_tid] = G_vec[_linear_tid * is_in_ker];
	}

	__syncthreads();

	if (is_in_dst) {
		float res = 0;
		int tmp_dex = idy_glo * srcDim.y;

		if (consider_border_L) {
			for (int y = idx_glo - __radius, _y = 0; _y < G_len; ++y, ++_y) {
				if (y < 0) {
					res = fma((float)mid[tmp_dex + abs(y)], tmp_vec[_y], res);
				}
				else {
					res = fma((float)mid[tmp_dex + y], tmp_vec[_y], res);
				}
			}
		}
		else if (consider_border_R) {
			int tmp_dex_more = tmp_dex + (srcDim.y << 1) - 1;
			for (int y = idx_glo - __radius, _y = 0; _y < G_len; ++y, ++_y) {
				if (y > bound_offset) {
					res = fma((float)mid[tmp_dex_more - y], tmp_vec[_y], res);
				}
				else {
					res = fma((float)mid[tmp_dex + y], tmp_vec[_y], res);
				}
			}
		}
		else {
			int tmp_dex_lin = tmp_dex + idx_glo;
			for (int y = tmp_dex_lin - __radius, _y = 0; _y < G_len; ++y, ++_y) {
				res = fma((float)mid[y], tmp_vec[_y], res);
			}
		}

		GetValue(dst, idx_glo, idy_glo, srcDim.x) = (uchar)res;
	}
	else { return; }
}



__global__
void Gaussian_blur_ver_mirror3D(uchar4* mid,
	float* G_vec,
	uchar4* dst,
	const uint G_len,
	const dim3 srcDim,
	const int __radius)
{
	uint idx_loc = threadIdx.x;
	uint idy_loc = threadIdx.y;
	uint _linear_tid = idx_loc * blockDim.y + idy_loc;

	uint idx_glo = idx_loc + blockIdx.x * blockDim.x;
	uint idy_glo = idy_loc + blockIdx.y * blockDim.y;

	bool consider_border_L = idx_glo < __radius;

	int Bottom_edge = srcDim.y - __radius - 1;
	int bound_offset = srcDim.y - 1;

	bool consider_border_R = idx_glo > Bottom_edge;

	// allocate the shared memory to store the kernel vector
	__shared__ float tmp_vec[_MTPB_];

	// load the kernel vector to the shared mamory
	bool is_in_ker = _linear_tid < G_len;
	bool is_in_dst = (idx_glo < srcDim.y) && (idy_glo < srcDim.x);

	// load the kernel
	if (is_in_ker) {
		tmp_vec[_linear_tid] = G_vec[_linear_tid * is_in_ker];
	}

	__syncthreads();

	if (is_in_dst) {
		float4 res;
		res.x = 0;
		res.y = 0;
		res.z = 0;

		int tmp_dex = idy_glo * srcDim.y;

		if (consider_border_L) {
			for (int y = idx_glo - __radius, _y = 0; _y < G_len; ++y, ++_y) {
				if (y < 0) {
					uchar4* _ptr = &mid[tmp_dex + abs(y)];
					res.x = fma((float)(_ptr->x), tmp_vec[_y], res.x);
					res.y = fma((float)(_ptr->y), tmp_vec[_y], res.y);
					res.z = fma((float)(_ptr->z), tmp_vec[_y], res.z);
				}
				else {
					uchar4* _ptr = &mid[tmp_dex + y];
					res.x = fma((float)(_ptr->x), tmp_vec[_y], res.x);
					res.y = fma((float)(_ptr->y), tmp_vec[_y], res.y);
					res.z = fma((float)(_ptr->z), tmp_vec[_y], res.z);
				}
			}
		}
		else if (consider_border_R) {
			int tmp_dex_more = tmp_dex + (srcDim.y << 1) - 1;
			for (int y = idx_glo - __radius, _y = 0; _y < G_len; ++y, ++_y) {
				if (y > bound_offset) {
					uchar4* _ptr = &mid[tmp_dex_more - y];
					res.x = fma((float)(_ptr->x), tmp_vec[_y], res.x);
					res.y = fma((float)(_ptr->y), tmp_vec[_y], res.y);
					res.z = fma((float)(_ptr->z), tmp_vec[_y], res.z);
				}
				else {
					uchar4* _ptr = &mid[tmp_dex + y];
					res.x = fma((float)(_ptr->x), tmp_vec[_y], res.x);
					res.y = fma((float)(_ptr->y), tmp_vec[_y], res.y);
					res.z = fma((float)(_ptr->z), tmp_vec[_y], res.z);
				}
			}
		}
		else {
			int tmp_dex_lin = tmp_dex + idx_glo;
			for (int y = tmp_dex_lin - __radius, _y = 0; _y < G_len; ++y, ++_y) {
				uchar4* _ptr = &mid[y];
				res.x = fma((float)(_ptr->x), tmp_vec[_y], res.x);
				res.y = fma((float)(_ptr->y), tmp_vec[_y], res.y);
				res.z = fma((float)(_ptr->z), tmp_vec[_y], res.z);
			}
		}
		uchar4* _ptr = &GetValue(dst, idx_glo, idy_glo, srcDim.x);
		
		_ptr->x = (uchar)res.x;
		_ptr->y = (uchar)res.y;
		_ptr->z = (uchar)res.z;
		_ptr->w = 255;
	}
	else { return; }
}




// NLM
#define NLM_Thr_blx 16
#define NLM_Thr_bly 16

__global__
/*first difference the image, and then square the difference
* the size of src must be larger than that of src, because this function 
* subtract the original element and its corresponding shifted element, the 
* shifting method is defined by _shift param
* attention: this function won't check if the index is out of range
* so make sure that the mapped element still located in src matrix
\param subed : the matrix been subtracted
*/
void cu_ImgDiff_Sq(uchar*		src,
				   float*		dst,
				   uchar*		subed,
				   int2			_shift,
				   int2			dstDim,
				   int			Wsrc)
{
	int2 dst_ID;
	dst_ID.x = threadIdx.x + blockIdx.x * blockDim.x;
	dst_ID.y = threadIdx.y + blockIdx.y * blockDim.y;

	int2 __shf;
	__shf.x = dst_ID.x + _shift.x;
	__shf.y = dst_ID.y + _shift.y;

	int lin_tid_pad = __shf.x * Wsrc + __shf.y;
	int lin_tid_dst = dst_ID.x * dstDim.x + dst_ID.y;

	if (dst_ID.x < dstDim.y && dst_ID.y < dstDim.x)
	{
		uchar2 tmp;
		tmp.x = subed[lin_tid_dst];
		tmp.y = GetValue(src, __shf.x, __shf.y, Wsrc);
		float __fval = 
			(float)(tmp.x) - (float)(tmp.y);
		
		dst[lin_tid_dst] = __fval * __fval;
	}
}



__global__
/*first difference the image, and then square the difference
* the size of src must be larger than that of src, because this function
* subtract the original element and its corresponding shifted element, the
* shifting method is defined by _shift param
* attention: this function won't check if the index is out of range
* so make sure that the mapped element still located in src matrix
\param subed : the matrix been subtracted
*/
void cu_ImgDiff_Sq3D(uchar*		src,
					 float4*	dst,
					 uchar*		subed,
					 int2		_shift,
					 int2		dstDim,
					 int		Wsrc)
{
	int2 dst_ID;
	dst_ID.x = threadIdx.x + blockIdx.x * blockDim.x;
	dst_ID.y = threadIdx.y + blockIdx.y * blockDim.y;

	int2 __shf;
	__shf.x = dst_ID.x + _shift.x;
	__shf.y = dst_ID.y + _shift.y;

	int lin_tid_pad = __shf.x * Wsrc + __shf.y;
	int lin_tid_dst = dst_ID.x * dstDim.x + dst_ID.y;

	if (dst_ID.x < dstDim.y && dst_ID.y < dstDim.x)
	{
		uchar4 _subed, _sub;
		*((int*)&_subed) = *((int*)(((uchar4*)subed) + lin_tid_dst));
		*((int*)&_sub) = *((int*)&GetValue((uchar4*)src, __shf.x, __shf.y, Wsrc));

		float3 __val;
		__val.x = _subed.x - _sub.x;
		__val.y = _subed.y - _sub.y;
		__val.z = _subed.z - _sub.z;

		__val.x = __val.x * __val.x;
		__val.y = __val.y * __val.y;
		__val.z = __val.z * __val.z;

		dst[lin_tid_dst].x = __val.x;
		dst[lin_tid_dst].y = __val.y;
		dst[lin_tid_dst].z = __val.z;
	}
}




__global__
void cu_NLM_calc(float		*diffSqImg,
				 uchar		*pixels,
				 float2		*accu,
				 int2		_V_shf,
				 int2		__Ne,
				 int		Ne_size,
				 int2		OriginDim,
				 int		Wdiff,
				 int		Wpix,
				 float		h_sq)
{
	int2 dst_ID;
	dst_ID.x = threadIdx.x + blockIdx.x * blockDim.x;
	dst_ID.y = threadIdx.y + blockIdx.y * blockDim.y;

	int2 diff_ID;
	diff_ID.x = dst_ID.x + __Ne.x;
	diff_ID.y = dst_ID.y + __Ne.y;

	int2 Ne_2;
	Ne_2.x = __Ne.x << 1;
	Ne_2.y = __Ne.y << 1;

	if (dst_ID.x < OriginDim.y && dst_ID.y < OriginDim.x)
	{
		float _sma_conv_res = 0;

		int base_dex = (diff_ID.x - __Ne.x) * Wdiff + diff_ID.y - __Ne.y;
		for (int i = 0; i < Ne_2.x + 1; ++i) {
			int __dex = base_dex;
			for (int j = 0; j < Ne_2.y + 1; ++j) {
				_sma_conv_res += diffSqImg[__dex];
				__dex++;
			}
			base_dex += Wdiff - Ne_2.y;
		}

		// avg = the sum of difference / d^2
		float __w = __fdividef(-_sma_conv_res, (float)Ne_size);
		// w = exp(-avg / h^2)
		__w = __expf(__fdividef(__w, h_sq));
		_sma_conv_res = __w * (float)GetValue(pixels, diff_ID.x + _V_shf.x, diff_ID.y + _V_shf.y, Wpix);
		
		// ~.x : result accumulator
		// ~.y : wights accumulator
		float2* _ptr = &(GetValue(accu, dst_ID.x, dst_ID.y, OriginDim.x));
		_ptr->x += _sma_conv_res;
		_ptr->y += __w;
	}
	else { return; }
}



struct __align__(32) float6
{
	float3 x, y;
};


__global__
void cu_NLM_calc_3D(float4*		diffSqImg,
					uchar*		pixels,
					float6*		accu,
					int2		_V_shf,
					int2		__Ne,
					int			Ne_size,
					int2		OriginDim,
					int			Wdiff,
					int			Wpix,
					float		h_sq)
{
	int2 dst_ID;
	dst_ID.x = threadIdx.x + blockIdx.x * blockDim.x;
	dst_ID.y = threadIdx.y + blockIdx.y * blockDim.y;

	int2 diff_ID;
	diff_ID.x = dst_ID.x + __Ne.x;
	diff_ID.y = dst_ID.y + __Ne.y;

	int2 Ne_2;
	Ne_2.x = __Ne.x << 1;
	Ne_2.y = __Ne.y << 1;

	if (dst_ID.x < OriginDim.y && dst_ID.y < OriginDim.x)
	{
		float3 _sma_conv_res;
		_sma_conv_res.x = 0;
		_sma_conv_res.y = 0;
		_sma_conv_res.z = 0;

		int base_dex = (diff_ID.x - __Ne.x) * Wdiff + diff_ID.y - __Ne.y;
		for (int i = 0; i < Ne_2.x + 1; ++i) {
			int __dex = base_dex;
			for (int j = 0; j < Ne_2.y + 1; ++j) {
				float4* diff_ptr = diffSqImg + __dex;
				_sma_conv_res.x += diff_ptr->x;
				_sma_conv_res.y += diff_ptr->y;
				_sma_conv_res.z += diff_ptr->z;
				__dex++;
			}
			base_dex += Wdiff - Ne_2.y;
		}

		// avg = the sum of difference / d^2
		float3 __w;
		__w.x = __fdividef(-_sma_conv_res.x, (float)Ne_size);
		__w.y = __fdividef(-_sma_conv_res.y, (float)Ne_size);
		__w.z = __fdividef(-_sma_conv_res.z, (float)Ne_size);
		// w = exp(-avg / h^2)
		__w.x = __expf(__fdividef(__w.x, h_sq));
		__w.y = __expf(__fdividef(__w.y, h_sq));
		__w.z = __expf(__fdividef(__w.z, h_sq));

		uchar4* pix_ptr = &GetValue((uchar4*)pixels, diff_ID.x + _V_shf.x, diff_ID.y + _V_shf.y, Wpix);
		
		_sma_conv_res.x = __w.x * (float)(pix_ptr->x);
		_sma_conv_res.y = __w.y * (float)(pix_ptr->y);
		_sma_conv_res.z = __w.z * (float)(pix_ptr->z);

		// ~.x : result accumulator
		// ~.y : wights accumulator
		float6* _ptr = &(GetValue(accu, dst_ID.x, dst_ID.y, OriginDim.x));
		_ptr->x.x += _sma_conv_res.x;
		_ptr->x.y += _sma_conv_res.y;
		_ptr->x.z += _sma_conv_res.z;

		_ptr->y.x += __w.x;
		_ptr->y.y += __w.y;
		_ptr->y.z += __w.z;
	}
	else { return; }
}



__global__
void cu_NLM_final(float2* src, uchar* dst, int2 dstDim)
{
	int2 dst_ID;
	dst_ID.x = threadIdx.x + blockIdx.x * blockDim.x;
	dst_ID.y = threadIdx.y + blockIdx.y * blockDim.y;

	if (dst_ID.x < dstDim.y && dst_ID.y < dstDim.x) 
	{
		int lin_tid = dst_ID.x * dstDim.x + dst_ID.y;
		float2* ptr = &(src[lin_tid]);

		dst[lin_tid] = (uchar)(__fdividef(ptr->x, ptr->y));
	}
	else { return; }
}



__global__
void cu_NLM_final_3D(float6* src, uchar* dst, int2 dstDim)
{
	int2 dst_ID;
	dst_ID.x = threadIdx.x + blockIdx.x * blockDim.x;
	dst_ID.y = threadIdx.y + blockIdx.y * blockDim.y;

	if (dst_ID.x < dstDim.y && dst_ID.y < dstDim.x)
	{
		int lin_tid = dst_ID.x * dstDim.x + dst_ID.y;
		float6* ptr = &src[lin_tid];
		uchar4* dst_ptr = (uchar4*)dst + lin_tid;

		*((uchar*)dst_ptr) = (uchar)(__fdividef(*((float*)ptr), *((float*)ptr + 3)));
		*((uchar*)dst_ptr + 1) = (uchar)(__fdividef(*((float*)ptr + 1), *((float*)ptr + 4)));
		*((uchar*)dst_ptr + 2) = (uchar)(__fdividef(*((float*)ptr + 2), *((float*)ptr + 5)));
	}
	else { return; }
}