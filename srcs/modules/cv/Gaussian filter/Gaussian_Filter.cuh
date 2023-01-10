#pragma once

#include "../../core/basic.h"


#define Gauss_threads_x 32
#define Gauss_threads_y 32




__global__
// dstDim [~.x : width, ~.y : height]
// 满高， 不满宽， 1D convolution horizentally
void Gaussian_blur_hor(uchar*			src,
					   float*			mid,
					   const uint		G_len,
					   const int2		dstDim,		// srcDim 就是传入图像的维度
					   const int2		srcDim)		// dstDim 就是传出图像的维度
{
	uint x_glo = threadIdx.x + blockIdx.x * blockDim.x;
	uint y_glo = threadIdx.y + blockIdx.y * blockDim.y;

	bool is_in = x_glo < dstDim.y && y_glo < dstDim.x;
	size_t lin_src = x_glo * srcDim.x + y_glo;

	if (is_in)
	{
		float tmp = 0, 
			  sum = 0;
		for (int i = 0; i < G_len; ++i) {
			tmp = (float)(src[lin_src]);
			sum += tmp * ((float*)Const_Mem)[i];
			++lin_src;
		}
		GetValue(mid, y_glo, x_glo, srcDim.y) = sum;
	}
}


__global__
// dstDim [~.x : width, ~.y : height]
// 不满高， 满宽， 1D convolution vertically
void Gaussian_blur_ver(float*			mid,
					   uchar*			dst,
					   const uint		G_len,
					   const int2		dstDim,		// srcDim 就是传入图像的维度
					   const int2		srcDim,
					   const uint		ker_offset)		// dstDim 就是传出图像的维度
{
	uint x_glo = threadIdx.x + blockIdx.x * blockDim.x;
	uint y_glo = threadIdx.y + blockIdx.y * blockDim.y;

	bool is_in = x_glo < dstDim.x && y_glo < dstDim.y;
	size_t lin_src = x_glo * srcDim.y + y_glo;

	if (is_in)
	{
		float tmp = 0, 
			  sum = 0;
		for (int i = ker_offset; i < G_len + ker_offset; ++i) {
			tmp = mid[lin_src];
			sum = fma(tmp, ((float*)Const_Mem)[i], sum);
			++lin_src;
		}
		GetValue(dst, y_glo, x_glo, dstDim.x) = (uchar)sum;
	}
}


// -------------------------------------------------------------------------------
//								fp16 
// -------------------------------------------------------------------------------

__global__
// dstDim [~.x : width, ~.y : height]
// 满高， 不满宽， 1D convolution horizentally
void Gaussian_blur_hor_fp16(uchar*			src,
							half*			mid,
							const uint		G_len,
							const int2		dstDim,		// srcDim 就是传入图像的维度
							const int2		srcDim)		// dstDim 就是传出图像的维度
{
#if __ABOVE_SM_53
	uint x_glo = threadIdx.x + blockIdx.x * blockDim.x;
	uint y_glo = threadIdx.y + blockIdx.y * blockDim.y;

	bool is_in = x_glo < dstDim.y && y_glo < dstDim.x;
	size_t lin_src = x_glo * srcDim.x + y_glo;

	if (is_in)
	{
		half tmp, sum, ker_val;
		*((short*)&sum) = 0;

		for (int i = 0; i < G_len; ++i) {
			tmp = (float)(src[lin_src]);
			ker_val = __float2half(((float*)Const_Mem)[i]);
			sum = __hfma(tmp, ker_val, sum);
			++lin_src;
		}
		GetValue(mid, y_glo, x_glo, srcDim.y) = sum;
	}
#endif
}


__global__
// dstDim [~.x : width, ~.y : height]
// 不满高， 满宽， 1D convolution vertically
void Gaussian_blur_ver_fp16(half*			mid,
							uchar*			dst,
							const uint		G_len,
							const int2		dstDim,		// srcDim 就是传入图像的维度
							const int2		srcDim,
							const uint		ker_offset)		// dstDim 就是传出图像的维度
{
#if __ABOVE_SM_53
	uint x_glo = threadIdx.x + blockIdx.x * blockDim.x;
	uint y_glo = threadIdx.y + blockIdx.y * blockDim.y;

	bool is_in = x_glo < dstDim.x && y_glo < dstDim.y;
	size_t lin_src = x_glo * srcDim.y + y_glo;

	if (is_in)
	{
		half tmp, sum, ker_val;
		*((short*)&sum) = 0;

		for (int i = ker_offset; i < G_len + ker_offset; ++i) {
			tmp = mid[lin_src];
			ker_val = __float2half(((float*)Const_Mem)[i]);
			sum = __hfma(tmp, ker_val, sum);
			++lin_src;
		}
		GetValue(dst, y_glo, x_glo, dstDim.x) = (uchar)__half2int_rn(sum);	
	}
#endif
}




// -------------------------------------------------------------------------------
//			3D image (RGBA)
// -------------------------------------------------------------------------------

__global__
// dstDim [~.x : width, ~.y : height]
// 满高， 不满宽， 1D convolution horizentally
void Gaussian_blur_hor3D(uchar4*		src,
						 float3*		mid,
						 const uint		G_len,
						 const int2		dstDim,		// srcDim 就是传入图像的维度
						 const int2		srcDim)		// dstDim 就是传出图像的维度
{
	uint x_glo = threadIdx.x + blockIdx.x * blockDim.x;
	uint y_glo = threadIdx.y + blockIdx.y * blockDim.y;

	bool is_in = x_glo < dstDim.y&& y_glo < dstDim.x;
	size_t lin_src = x_glo * srcDim.x + y_glo;

	if (is_in)
	{
		uchar4 tmp;
		float ker_val;
		float3 sum;
		sum.x = 0;
		sum.y = 0;
		sum.z = 0;
		uchar4* _mid_src;
		for (int i = 0; i < G_len; ++i) {
			*((int*)&tmp) = ((int*)src)[lin_src];
			ker_val = ((float*)Const_Mem)[i];

			sum.x += (float)tmp.x * ker_val;
			sum.y += (float)tmp.y * ker_val;
			sum.z += (float)tmp.z * ker_val;

			++lin_src;
		}
		float3* _mid_ptr = &GetValue(mid, y_glo, x_glo, srcDim.y);
		
		_mid_ptr->x = sum.x;
		_mid_ptr->y = sum.y;
		_mid_ptr->z = sum.z;
	}
}



__global__
// dstDim [~.x : width, ~.y : height]
// 不满高， 满宽， 1D convolution vertically
void Gaussian_blur_ver3D(float3*		mid,
						 uchar4*		dst,
						 const uint		G_len,
						 const int2		dstDim,		// srcDim 就是传入图像的维度
						 const int2		srcDim,		// dstDim 就是传出图像的维度
						 const uint		ker_offset)		
{
	uint x_glo = threadIdx.x + blockIdx.x * blockDim.x;
	uint y_glo = threadIdx.y + blockIdx.y * blockDim.y;

	bool is_in = x_glo < dstDim.x&& y_glo < dstDim.y;
	size_t lin_src = x_glo * srcDim.y + y_glo;

	if (is_in)
	{
		float4 tmp;
		float3 sum;
		sum.x = 0;		
		sum.y = 0;
		sum.z = 0;
		float3* _mid_src;
		for (int i = ker_offset; i < G_len + ker_offset; ++i) {
			_mid_src = &mid[lin_src];
			tmp.x = _mid_src->x;
			tmp.y = _mid_src->y;
			tmp.z = _mid_src->z;
			tmp.w = ((float*)Const_Mem)[i];

			sum.x += tmp.x * tmp.w;
			sum.y += tmp.y * tmp.w;
			sum.z += tmp.z * tmp.w;
			++lin_src;
		}
		uchar4* _dst_ptr = &GetValue(dst, y_glo, x_glo, dstDim.x);
		_dst_ptr->x = (uchar)(sum.x);
		_dst_ptr->y = (uchar)(sum.y);
		_dst_ptr->z = (uchar)(sum.z);
	}
}





// ---------------------------------------------------------------------------------------
//					fp16 function
// ---------------------------------------------------------------------------------------


__global__
// dstDim [~.x : width, ~.y : height]
// 满高， 不满宽， 1D convolution horizentally
void Gaussian_blur_hor3D_fp16(uchar4*			src,
							  half4*			mid,
							  const uint		G_len,
							  const int2		dstDim,		// srcDim 就是传入图像的维度
							  const int2		srcDim)		// dstDim 就是传出图像的维度
{
#if __ABOVE_SM_53
	uint x_glo = threadIdx.x + blockIdx.x * blockDim.x;
	uint y_glo = threadIdx.y + blockIdx.y * blockDim.y;

	bool is_in = x_glo < dstDim.y&& y_glo < dstDim.x;
	size_t lin_src = x_glo * srcDim.x + y_glo;

	if (is_in)
	{
		uchar4 tmp;
		half ker_val;
		half4 sum;
		*((double*)&sum) = 0.0;

		uchar4* _mid_src;
		for (int i = 0; i < G_len; ++i) {
			*((int*)&tmp) = ((int*)src)[lin_src];
			ker_val = __float2half(((float*)Const_Mem)[i]);

			/*sum.x += (float)tmp.x * ker_val;
			sum.y += (float)tmp.y * ker_val;
			sum.z += (float)tmp.z * ker_val;*/
			sum.x = __hfma(__int2half_rn((int)tmp.x), ker_val, sum.x);
			sum.y = __hfma(__int2half_rn((int)tmp.y), ker_val, sum.y);
			sum.z = __hfma(__int2half_rn((int)tmp.z), ker_val, sum.z);

			++lin_src;
		}
		half4* _mid_ptr = &GetValue(mid, y_glo, x_glo, srcDim.y);
		
		_mid_ptr->x = sum.x;
		_mid_ptr->y = sum.y;
		_mid_ptr->z = sum.z;
	}
#endif
}



__global__
// dstDim [~.x : width, ~.y : height]
// 不满高， 满宽， 1D convolution vertically
void Gaussian_blur_ver3D_fp16(half4*			mid,
							  uchar4*			dst,
							  const uint		G_len,
							  const int2		dstDim,		// srcDim 就是传入图像的维度
							  const int2		srcDim,		// dstDim 就是传出图像的维度
							  const uint		ker_offset)		
{
#if __ABOVE_SM_53
	uint x_glo = threadIdx.x + blockIdx.x * blockDim.x;
	uint y_glo = threadIdx.y + blockIdx.y * blockDim.y;

	bool is_in = x_glo < dstDim.x&& y_glo < dstDim.y;
	size_t lin_src = x_glo * srcDim.y + y_glo;

	if (is_in)
	{
		half4 tmp, sum;
		half ker_val;
		*((double*)&sum) = 0.0;

		half4* _mid_src;
		for (int i = ker_offset; i < G_len + ker_offset; ++i) {
			/*_mid_src = &mid[lin_src];
			tmp.x = _mid_src->x;
			tmp.y = _mid_src->y;
			tmp.z = _mid_src->z;
			tmp.w = ((float*)Const_Mem)[i];*/
			*((double*)&tmp) = ((double*)mid)[lin_src];
			ker_val = __float2half(((float*)Const_Mem)[i]);

			/*sum.x += tmp.x * tmp.w;
			sum.y += tmp.y * tmp.w;
			sum.z += tmp.z * tmp.w;*/
			sum.x = __hfma(tmp.x, ker_val, sum.x);
			sum.y = __hfma(tmp.y, ker_val, sum.y);
			sum.z = __hfma(tmp.z, ker_val, sum.z);

			++lin_src;
		}
		uchar4* _dst_ptr = &GetValue(dst, y_glo, x_glo, dstDim.x);
		_dst_ptr->x = (uchar)__half2int_rn((sum.x));
		_dst_ptr->y = (uchar)__half2int_rn((sum.y));
		_dst_ptr->z = (uchar)__half2int_rn((sum.z));
	}
#endif
}