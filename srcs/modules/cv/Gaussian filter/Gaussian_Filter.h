#pragma once


#include "Gaussian_Filter.cuh"
#include "../cv_classes/cv_classes.h"


static
void Gaussian_vec_gen(float			*arr, 
					  const int		__radius, 
					  const double	__sigma,
					  const int		len)
{
	float res = 0, sum = 0;
	const double tmp = 2 * __sigma * __sigma;
	double x_axis = 0;

	for (int i = 0; i < len; ++i) {
		x_axis = static_cast<double>(i - __radius);
		res = 0.3989422 * exp(-(x_axis * x_axis) / tmp) / __sigma;
		sum += res;
		arr[i] = res;
	}
	for (int i = 0; i < len; ++i) {
		arr[i] /= sum;
	}
}



namespace de
{
	namespace vis
	{
		_DECX_API_
		de::DH GaussianBlur2D(de::vis::Img& src, de::vis::Img& dst, const de::Point2D radius, const de::Point2D_d sigma);


		_DECX_API_
		de::DH GaussianBlur3D(de::vis::Img& src, de::vis::Img& dst, const de::Point2D radius, const de::Point2D_d sigma);
	}
}



static
void _GaussianBlur2D(_Img& src, _Img& dst, const de::Point2D radius, const de::Point2D_d sigma)
{
	int2 srcDim, midDim, kerDim, dstDim;
	srcDim.x = src.width;						srcDim.y = src.height;
	midDim.x = dst.width;						midDim.y = src.height;
	kerDim.x = (radius.x * 2) + 1;				kerDim.y = (radius.y * 2) + 1;
	dstDim.x = src.width - (radius.x * 2);		dstDim.y = src.height - (radius.y * 2);

	uchar* dev_src;
	float* dev_mid;

	const size_t src_bytes = static_cast<size_t>(srcDim.x) * static_cast<size_t>(srcDim.y) * sizeof(uchar);
	checkCudaErrors(cudaMalloc(&dev_src, src_bytes));
	checkCudaErrors(cudaMalloc(
		&dev_mid, static_cast<size_t>(midDim.x) * static_cast<size_t>(midDim.y) * sizeof(float)));

	float* kernel_x,
		 * kernel_y;

	kernel_x = (float*)malloc((kerDim.x + kerDim.y) * sizeof(float));
	kernel_y = kernel_x + kerDim.x;

	cudaStream_t S0, S1;
	checkCudaErrors(cudaStreamCreate(&S0));
	checkCudaErrors(cudaStreamCreate(&S1));

	checkCudaErrors(cudaMemcpyAsync(dev_src, src.Mat, src_bytes, cudaMemcpyHostToDevice, S0));

	// generate the Gaussian kernel
	Gaussian_vec_gen(kernel_x, radius.x, sigma.x, kerDim.x);
	Gaussian_vec_gen(kernel_y, radius.y, sigma.y, kerDim.y);

	checkCudaErrors(cudaMemcpyToSymbolAsync(
		Const_Mem, kernel_x, (kerDim.x + kerDim.y) * sizeof(float), 0, cudaMemcpyHostToDevice, S0));

	const dim3 block(Gauss_threads_x, Gauss_threads_y);
	// 横向一维卷积，满高，不满宽
	dim3 grid(_cu_ceil(srcDim.y, Gauss_threads_x), _cu_ceil(dstDim.x, Gauss_threads_y));

	checkCudaErrors(cudaDeviceSynchronize());

	// hor
	Gaussian_blur_hor << <grid, block, 0, S1 >> > (
		dev_src, dev_mid, kerDim.x, dstDim, srcDim);

	// 每次配置grid时，考虑每次输出矩阵转置前的维度
	grid.x = _cu_ceil(dstDim.x, Gauss_threads_y);
	grid.y = _cu_ceil(dstDim.y, Gauss_threads_x);
	
	// ver
	Gaussian_blur_ver << <grid, block, 0, S1 >> > (
		dev_mid, dev_src, kerDim.y, dstDim, srcDim, kerDim.x);

	checkCudaErrors(cudaMemcpyAsync(
		dst.Mat, dev_src, static_cast<size_t>(dstDim.x) * static_cast<size_t>(dstDim.y) * sizeof(uchar), cudaMemcpyDeviceToHost, S1));

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(dev_src));
	checkCudaErrors(cudaFree(dev_mid));
	free(kernel_x);
	checkCudaErrors(cudaStreamDestroy(S0));
	checkCudaErrors(cudaStreamDestroy(S1));
}




static
void _GaussianBlur2D_fp16(_Img& src, _Img& dst, const de::Point2D radius, const de::Point2D_d sigma)
{
	int2 srcDim, midDim, kerDim, dstDim;
	srcDim.x = src.width;						srcDim.y = src.height;
	midDim.x = dst.width;						midDim.y = src.height;
	kerDim.x = (radius.x * 2) + 1;				kerDim.y = (radius.y * 2) + 1;
	dstDim.x = src.width - (radius.x * 2);		dstDim.y = src.height - (radius.y * 2);

	uchar* dev_src;
	half* dev_mid;

	const size_t src_bytes = static_cast<size_t>(srcDim.x) * static_cast<size_t>(srcDim.y) * sizeof(uchar);
	checkCudaErrors(cudaMalloc(&dev_src, src_bytes));
	checkCudaErrors(cudaMalloc(
		&dev_mid, static_cast<size_t>(midDim.x) * static_cast<size_t>(midDim.y) * sizeof(half)));

	float* kernel_x,
		* kernel_y;

	kernel_x = (float*)malloc((kerDim.x + kerDim.y) * sizeof(float));
	kernel_y = kernel_x + kerDim.x;

	cudaStream_t S0, S1;
	checkCudaErrors(cudaStreamCreate(&S0));
	checkCudaErrors(cudaStreamCreate(&S1));

	checkCudaErrors(cudaMemcpyAsync(dev_src, src.Mat, src_bytes, cudaMemcpyHostToDevice, S0));

	// generate the Gaussian kernel
	Gaussian_vec_gen(kernel_x, radius.x, sigma.x, kerDim.x);
	Gaussian_vec_gen(kernel_y, radius.y, sigma.y, kerDim.y);

	checkCudaErrors(cudaMemcpyToSymbolAsync(
		Const_Mem, kernel_x, (kerDim.x + kerDim.y) * sizeof(float), 0, cudaMemcpyHostToDevice, S0));

	const dim3 block(Gauss_threads_x, Gauss_threads_y);
	// 横向一维卷积，满高，不满宽
	dim3 grid(_cu_ceil(srcDim.y, Gauss_threads_x), _cu_ceil(dstDim.x, Gauss_threads_y));

	checkCudaErrors(cudaDeviceSynchronize());

	// hor
	Gaussian_blur_hor_fp16 << <grid, block, 0, S1 >> > (
		dev_src, dev_mid, kerDim.x, dstDim, srcDim);

	// 每次配置grid时，考虑每次输出矩阵转置前的维度
	grid.x = _cu_ceil(dstDim.x, Gauss_threads_y);
	grid.y = _cu_ceil(dstDim.y, Gauss_threads_x);

	// ver
	Gaussian_blur_ver_fp16 << <grid, block, 0, S1 >> > (
		dev_mid, dev_src, kerDim.y, dstDim, srcDim, kerDim.x);

	checkCudaErrors(cudaMemcpyAsync(
		dst.Mat, dev_src, static_cast<size_t>(dstDim.x) * static_cast<size_t>(dstDim.y) * sizeof(uchar), cudaMemcpyDeviceToHost, S1));

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(dev_src));
	checkCudaErrors(cudaFree(dev_mid));
	free(kernel_x);
	checkCudaErrors(cudaStreamDestroy(S0));
	checkCudaErrors(cudaStreamDestroy(S1));
}


// --------------------------------------------------------------------------
//				3D Image (4 bytes aligned)
// --------------------------------------------------------------------------


static
void _GaussianBlur3D_fp16(_Img& src, _Img& dst, const de::Point2D radius, const de::Point2D_d sigma)
{
	int2 srcDim, midDim, kerDim, dstDim;
	srcDim.x = src.width;						srcDim.y = src.height;
	midDim.x = dst.width;						midDim.y = src.height;
	kerDim.x = (radius.x * 2) + 1;				kerDim.y = (radius.y * 2) + 1;
	dstDim.x = src.width - (radius.x * 2);		dstDim.y = src.height - (radius.y * 2);

	uchar4* dev_src;
	half4* dev_mid;

	const size_t src_bytes = static_cast<size_t>(srcDim.x) * static_cast<size_t>(srcDim.y) * sizeof(uchar4);
	checkCudaErrors(cudaMalloc(&dev_src, src_bytes));
	checkCudaErrors(cudaMalloc(
		&dev_mid, static_cast<size_t>(midDim.x) * static_cast<size_t>(midDim.y) * sizeof(half4)));

	float* kernel_x,
		* kernel_y;

	kernel_x = (float*)malloc((kerDim.x + kerDim.y) * sizeof(float));
	kernel_y = kernel_x + kerDim.x;

	cudaStream_t S0, S1;
	checkCudaErrors(cudaStreamCreate(&S0));
	checkCudaErrors(cudaStreamCreate(&S1));

	checkCudaErrors(cudaMemcpyAsync(dev_src, src.Mat, src_bytes, cudaMemcpyHostToDevice, S0));

	// generate the Gaussian kernel
	Gaussian_vec_gen(kernel_x, radius.x, sigma.x, kerDim.x);
	Gaussian_vec_gen(kernel_y, radius.y, sigma.y, kerDim.y);

	checkCudaErrors(cudaMemcpyToSymbolAsync(
		Const_Mem, kernel_x, (kerDim.x + kerDim.y) * sizeof(float), 0, cudaMemcpyHostToDevice, S0));

	const dim3 block(Gauss_threads_x, Gauss_threads_y);
	// 横向一维卷积，满高，不满宽
	dim3 grid(_cu_ceil(srcDim.y, Gauss_threads_x), _cu_ceil(dstDim.x, Gauss_threads_y));

	checkCudaErrors(cudaDeviceSynchronize());

	// hor
	Gaussian_blur_hor3D_fp16 << <grid, block, 0, S1 >> > (
		dev_src, dev_mid, kerDim.x, dstDim, srcDim);

	// 每次配置grid时，考虑每次输出矩阵转置前的维度
	grid.x = _cu_ceil(dstDim.x, Gauss_threads_y);
	grid.y = _cu_ceil(dstDim.y, Gauss_threads_x);

	// ver
	Gaussian_blur_ver3D_fp16 << <grid, block, 0, S1 >> > (
		dev_mid, dev_src, kerDim.y, dstDim, srcDim, kerDim.x);

	checkCudaErrors(cudaMemcpyAsync(
		dst.Mat, dev_src, static_cast<size_t>(dstDim.x) * static_cast<size_t>(dstDim.y) * sizeof(uchar4), cudaMemcpyDeviceToHost, S1));

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(dev_src));
	checkCudaErrors(cudaFree(dev_mid));
	free(kernel_x);
	checkCudaErrors(cudaStreamDestroy(S0));
	checkCudaErrors(cudaStreamDestroy(S1));
}




static
void _GaussianBlur3D(_Img& src, _Img& dst, const de::Point2D radius, const de::Point2D_d sigma)
{
	int2 srcDim, midDim, kerDim, dstDim;
	srcDim.x = src.width;						srcDim.y = src.height;
	midDim.x = dst.width;						midDim.y = src.height;
	kerDim.x = (radius.x * 2) + 1;				kerDim.y = (radius.y * 2) + 1;
	dstDim.x = src.width - (radius.x * 2);		dstDim.y = src.height - (radius.y * 2);

	uchar4* dev_src;
	float3* dev_mid;

	const size_t src_bytes = static_cast<size_t>(srcDim.x) * static_cast<size_t>(srcDim.y) * sizeof(uchar4);
	checkCudaErrors(cudaMalloc(&dev_src, src_bytes));

	checkCudaErrors(cudaMalloc(
		&dev_mid, static_cast<size_t>(midDim.x) * static_cast<size_t>(midDim.y) * sizeof(float3)));

	float* kernel_x,
		* kernel_y;

	kernel_x = (float*)malloc((kerDim.x + kerDim.y) * sizeof(float));
	kernel_y = kernel_x + kerDim.x;

	cudaStream_t S0, S1;
	checkCudaErrors(cudaStreamCreate(&S0));
	checkCudaErrors(cudaStreamCreate(&S1));

	checkCudaErrors(cudaMemcpyAsync(dev_src, src.Mat, src_bytes, cudaMemcpyHostToDevice, S0));

	// generate the Gaussian kernel
	Gaussian_vec_gen(kernel_x, radius.x, sigma.x, kerDim.x);
	Gaussian_vec_gen(kernel_y, radius.y, sigma.y, kerDim.y);

	checkCudaErrors(cudaMemcpyToSymbolAsync(
		Const_Mem, kernel_x, (kerDim.x + kerDim.y) * sizeof(float), 0, cudaMemcpyHostToDevice, S0));

	const dim3 block(Gauss_threads_x, Gauss_threads_y);
	// 横向一维卷积，满高，不满宽
	dim3 grid(_cu_ceil(srcDim.y, Gauss_threads_x), _cu_ceil(dstDim.x, Gauss_threads_y));

	checkCudaErrors(cudaDeviceSynchronize());

	// hor
	Gaussian_blur_hor3D << <grid, block, 0, S1 >> > (
		dev_src, dev_mid, kerDim.x, dstDim, srcDim);

	// 每次配置grid时，考虑每次输出矩阵转置前的维度
	grid.x = _cu_ceil(dstDim.x, Gauss_threads_y);
	grid.y = _cu_ceil(dstDim.y, Gauss_threads_x);

	// ver
	Gaussian_blur_ver3D << <grid, block, 0, S1 >> > (
		dev_mid, dev_src, kerDim.y, dstDim, srcDim, kerDim.x);

	checkCudaErrors(cudaMemcpyAsync(
		dst.Mat, dev_src, static_cast<size_t>(dstDim.x) * static_cast<size_t>(dstDim.y) * sizeof(uchar4), cudaMemcpyDeviceToHost, S1));

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(dev_src));
	checkCudaErrors(cudaFree(dev_mid));
	free(kernel_x);
	checkCudaErrors(cudaStreamDestroy(S0));
	checkCudaErrors(cudaStreamDestroy(S1));
}




de::DH de::vis::GaussianBlur2D(de::vis::Img& src, de::vis::Img& dst, const de::Point2D radius, const de::Point2D_d sigma)
{
	de::DH handle;

	if (cuP.is_init != true) {
		handle.error_string = "CUDA should be initialize first";
		handle.error_type = decx::DECX_FAIL_not_init;
		return handle;
	}

	_Img& _src = dynamic_cast<_Img&>(src);
	_Img& _dst = dynamic_cast<_Img&>(dst);

	if (_src.channel != 1) {
		handle.error_string = "Source matrix must be a vis::Img class with DE_UC4";
		handle.error_type = decx::DECX_FAIL_ChannelError;
		return handle;
	}
	if (cuP.major > 6) {
		_GaussianBlur2D(_src, _dst, radius, sigma);
	}
	else{
		_GaussianBlur2D_fp16(_src, _dst, radius, sigma);
	}

	handle.error_string = "No error";
	handle.error_type = decx::DECX_SUCCESS;
	return handle;
}




de::DH de::vis::GaussianBlur3D(de::vis::Img& src, de::vis::Img& dst, const de::Point2D radius, const de::Point2D_d sigma)
{
	de::DH handle;

	if (cuP.is_init != true) {
		handle.error_string = "CUDA should be initialize first";
		handle.error_type = decx::DECX_FAIL_not_init;
		return handle;
	}

	_Img& _src = dynamic_cast<_Img&>(src);
	_Img& _dst = dynamic_cast<_Img&>(dst);

	if (_src.channel != 4) {
		handle.error_string = "Source matrix must be a vis::Img class with DE_UC4";
		handle.error_type = decx::DECX_FAIL_ChannelError;
		return handle;
	}

	// if the computebility is larger than 6
	if (cuP.major > 6) {
		_GaussianBlur3D_fp16(_src, _dst, radius, sigma);
	}
	else {
		_GaussianBlur3D(_src, _dst, radius, sigma);
	}

	handle.error_string = "No error";
	handle.error_type = decx::DECX_SUCCESS;
	return handle;
}