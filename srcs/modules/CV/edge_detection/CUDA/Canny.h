#pragma once

#include "../../core/basic.h"
#include "../../classes/matrix.h"
#include "Canny.cuh"
#include "../../cv/cv_classes/cv_classes.h"



namespace de
{
	namespace vis
	{
		_DECX_API_
		void Canny_Sobel(de::vis::Img& src, de::vis::Img& dst, const float _L_thr, const float _H_thr, const uint step);


		_DECX_API_
		void Canny_Scharr(de::vis::Img& src, de::vis::Img& dst, const float _L_thr, const float _H_thr, const uint step);


		_DECX_API_
		void Canny_Prewitt(de::vis::Img& src, de::vis::Img& dst, const float _L_thr, const float _H_thr, const uint step);


		_DECX_API_
		void _Canny_Sobel(de::vis::dev_Img& src, de::vis::dev_Img& dst, const float _L_thr, const float _H_thr, const uint step);


		_DECX_API_
		void _Canny_Scharr(de::vis::dev_Img& src, de::vis::dev_Img& dst, const float _L_thr, const float _H_thr, const uint step);


		_DECX_API_
		void _Canny_Prewitt(de::vis::dev_Img& src, de::vis::dev_Img& dst, const float _L_thr, const float _H_thr, const uint step);
	}
}




void de::vis::Canny_Sobel(de::vis::Img& src, de::vis::Img& dst, const float _L_thr, const float _H_thr, const uint step)
{
	if (cuP.is_init != true) {
		printf("CUDA should be initialized first\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	_Img& _src = dynamic_cast<_Img&>(src);
	_Img& _dst = dynamic_cast<_Img&>(dst);

	if (_src.channel != 1) {
		printf("The input Image array is not a 1D array\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	// ~.x : width; ~.y : height
	const dim3 srcDim(_src.width, _src.height);
	// ~.x : width; ~.y : height
	const dim3 dstDim = srcDim;

	uchar* dev_src, * dev_dst;
	int* dev_base_x, * dev_base_y;

	GI* GI_tmp;
	size_t src_pitch;
	size_t srcWidth = _src.width * sizeof(uchar);
	checkCudaErrors(cudaMallocPitch(reinterpret_cast<void**>(&dev_src), &src_pitch, srcWidth, _src.height));

	cudaStream_t S_C, S_K_0, S_K_1;
	checkCudaErrors(cudaStreamCreateWithFlags(&S_C, cudaStreamNonBlocking));

	// copy the source datas from host to device
	checkCudaErrors(cudaMemcpy2DAsync(dev_src, src_pitch, _src.Mat, srcWidth, srcWidth, _src.height, cudaMemcpyHostToDevice, S_C));

	const dim3 Proc(dstDim.x / step, dstDim.y / step);

	size_t GI_pitch;
	checkCudaErrors(cudaMallocPitch(reinterpret_cast<void**>(&GI_tmp), &GI_pitch, Proc.x * sizeof(GI), Proc.y));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_base_x), (srcDim.x * dstDim.y + dstDim.x * srcDim.y) * sizeof(int)));

	dev_base_y = dev_base_x + srcDim.x * dstDim.y;
	GI_pitch /= sizeof(GI);
	src_pitch /= sizeof(uchar);

	checkCudaErrors(cudaStreamCreateWithFlags(&S_K_0, cudaStreamNonBlocking));
	checkCudaErrors(cudaStreamCreateWithFlags(&S_K_1, cudaStreamNonBlocking));

	// configure the launch parameters, under the global domain
	uint _W_over_quo = _cu_ceil(dstDim.x, Sobel_Thr_y);
	uint _H_over_quo = _cu_ceil(dstDim.y, Sobel_Thr_x);
	dim3 grid(_H_over_quo, _W_over_quo);
	dim3 threads(1, Sobel_Thr_x * Sobel_Thr_y);

	// the parameters under the process domain
	uint _PW_over_quo = _cu_ceil(Proc.x, Sobel_Thr_y);
	uint _PH_over_quo = _cu_ceil(Proc.y, Sobel_Thr_x);

	dim3 P_grid(_PH_over_quo, _PW_over_quo);

	checkCudaErrors(cudaDeviceSynchronize());

	Sobel_base_x << <grid, threads, 0, S_K_0 >> > (dev_src,
		dev_base_x,
		dstDim,
		src_pitch);

	Sobel_base_y << <grid, threads, 0, S_K_1 >> > (dev_src,
		dev_base_y,
		dstDim,
		src_pitch);

	Soble_x << <P_grid, threads, 0, S_K_0 >> > (dev_base_x,
		GI_tmp,
		step,
		dstDim,
		Proc,
		GI_pitch);

	Soble_y << <P_grid, threads, 0, S_K_1 >> > (dev_base_y,
		GI_tmp,
		step,
		dstDim,
		Proc,
		GI_pitch);

	checkCudaErrors(cudaHostGetDevicePointer(&dev_dst, _dst.Mat, 0));

	checkCudaErrors(cudaDeviceSynchronize());
	//越界
	Sobel_SummingXY << <P_grid, threads, 0, S_K_0 >> > (GI_tmp,
		step,
		Proc,
		GI_pitch);

	Sobel_Final_Calc << <P_grid, threads, 0, S_K_0 >> > (GI_tmp,
		dev_dst,
		_L_thr,
		_H_thr,
		step,
		dstDim,
		Proc,
		GI_pitch);

	checkCudaErrors(cudaFree(dev_src));
	checkCudaErrors(cudaFree(dev_base_x));
	// synchronize S_K_0, and finish the final calculaation
	checkCudaErrors(cudaStreamSynchronize(S_K_0));

	checkCudaErrors(cudaStreamDestroy(S_K_1));
	checkCudaErrors(cudaStreamSynchronize(S_C));
	checkCudaErrors(cudaStreamDestroy(S_C));
}




void de::vis::Canny_Prewitt(de::vis::Img& src, de::vis::Img& dst, const float _L_thr, const float _H_thr, const uint step)
{
	if (cuP.is_init != true) {
		printf("CUDA should be initialized first\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	_Img& _src = dynamic_cast<_Img&>(src);
	_Img& _dst = dynamic_cast<_Img&>(dst);

	if (_src.channel != 1) {
		printf("The input Image array is not a 1D array\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	// ~.x : width; ~.y : height
	const dim3 srcDim(_src.width, _src.height);
	// ~.x : width; ~.y : height
	const dim3 dstDim = srcDim;

	uchar* dev_src, * dev_dst;
	int* dev_base_x, * dev_base_y;

	GI* GI_tmp;
	size_t src_pitch;
	size_t srcWidth = _src.width * sizeof(uchar);
	checkCudaErrors(cudaMallocPitch(reinterpret_cast<void**>(&dev_src), &src_pitch, srcWidth, _src.height));

	cudaStream_t S_C, S_K_0, S_K_1;
	checkCudaErrors(cudaStreamCreateWithFlags(&S_C, cudaStreamNonBlocking));

	// copy the source datas from host to device
	checkCudaErrors(cudaMemcpy2DAsync(dev_src, src_pitch, _src.Mat, srcWidth, srcWidth, _src.height, cudaMemcpyHostToDevice, S_C));

	const dim3 Proc(dstDim.x / step, dstDim.y / step);

	size_t GI_pitch;
	checkCudaErrors(cudaMallocPitch(reinterpret_cast<void**>(&GI_tmp), &GI_pitch, Proc.x * sizeof(GI), Proc.y));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_base_x), (srcDim.x * dstDim.y + dstDim.x * srcDim.y) * sizeof(int)));

	dev_base_y = dev_base_x + srcDim.x * dstDim.y;
	GI_pitch /= sizeof(GI);
	src_pitch /= sizeof(uchar);

	checkCudaErrors(cudaStreamCreateWithFlags(&S_K_0, cudaStreamNonBlocking));
	checkCudaErrors(cudaStreamCreateWithFlags(&S_K_1, cudaStreamNonBlocking));

	// configure the launch parameters, under the global domain
	uint _W_over_quo = _cu_ceil(dstDim.x, Sobel_Thr_y);
	uint _H_over_quo = _cu_ceil(dstDim.y, Sobel_Thr_x);
	dim3 grid(_H_over_quo, _W_over_quo);
	dim3 threads(1, Sobel_Thr_x * Sobel_Thr_y);

	// the parameters under the process domain
	uint _PW_over_quo = _cu_ceil(Proc.x, Sobel_Thr_y);
	uint _PH_over_quo = _cu_ceil(Proc.y, Sobel_Thr_x);

	dim3 P_grid(_PH_over_quo, _PW_over_quo);

	checkCudaErrors(cudaDeviceSynchronize());

	Sobel_base_x << <grid, threads, 0, S_K_0 >> > (dev_src,
		dev_base_x,
		dstDim,
		src_pitch);

	Sobel_base_y << <grid, threads, 0, S_K_1 >> > (dev_src,
		dev_base_y,
		dstDim,
		src_pitch);

	Prewitt_x << <P_grid, threads, 0, S_K_0 >> > (dev_base_x,
		GI_tmp,
		step,
		dstDim,
		Proc,
		GI_pitch);

	Prewitt_y << <P_grid, threads, 0, S_K_1 >> > (dev_base_y,
		GI_tmp,
		step,
		dstDim,
		Proc,
		GI_pitch);

	checkCudaErrors(cudaHostGetDevicePointer(&dev_dst, _dst.Mat, 0));

	checkCudaErrors(cudaDeviceSynchronize());
	//越界
	Sobel_SummingXY << <P_grid, threads, 0, S_K_0 >> > (GI_tmp,
		step,
		Proc,
		GI_pitch);

	Sobel_Final_Calc << <P_grid, threads, 0, S_K_0 >> > (GI_tmp,
		dev_dst,
		_L_thr,
		_H_thr,
		step,
		dstDim,
		Proc,
		GI_pitch);

	checkCudaErrors(cudaFree(dev_src));
	checkCudaErrors(cudaFree(GI_tmp));
	checkCudaErrors(cudaFree(dev_base_x));
	// synchronize S_K_0, and finish the final calculaation
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaStreamDestroy(S_K_1));
	checkCudaErrors(cudaStreamDestroy(S_C));
}






void de::vis::Canny_Scharr(de::vis::Img& src, de::vis::Img& dst, const float _L_thr, const float _H_thr, const uint step)
{
	if (cuP.is_init != true) {
		printf("CUDA should be initialized first\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	_Img& _src = dynamic_cast<_Img&>(src);
	_Img& _dst = dynamic_cast<_Img&>(dst);

	if (_src.channel != 1) {
		printf("The input Image array is not a 1D array\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	// ~.x : width; ~.y : height
	const dim3 srcDim(_src.width, _src.height);
	// ~.x : width; ~.y : height
	const dim3 dstDim = srcDim;

	uchar* dev_src, * dev_dst;
	int* dev_base_x, * dev_base_y;

	GI* GI_tmp;
	size_t src_pitch;
	size_t srcWidth = _src.width * sizeof(uchar);
	checkCudaErrors(cudaMallocPitch(reinterpret_cast<void**>(&dev_src), &src_pitch, srcWidth, _src.height));

	cudaStream_t S_C, S_K_0, S_K_1;
	checkCudaErrors(cudaStreamCreateWithFlags(&S_C, cudaStreamNonBlocking));

	// copy the source datas from host to device
	checkCudaErrors(cudaMemcpy2DAsync(dev_src, src_pitch, _src.Mat, srcWidth, srcWidth, _src.height, cudaMemcpyHostToDevice, S_C));

	const dim3 Proc(dstDim.x / step, dstDim.y / step);

	size_t GI_pitch;
	checkCudaErrors(cudaMallocPitch(reinterpret_cast<void**>(&GI_tmp), &GI_pitch, Proc.x * sizeof(GI), Proc.y));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_base_x), (srcDim.x * dstDim.y + dstDim.x * srcDim.y) * sizeof(int)));

	dev_base_y = dev_base_x + srcDim.x * dstDim.y;
	GI_pitch /= sizeof(GI);
	src_pitch /= sizeof(uchar);

	checkCudaErrors(cudaStreamCreateWithFlags(&S_K_0, cudaStreamNonBlocking));
	checkCudaErrors(cudaStreamCreateWithFlags(&S_K_1, cudaStreamNonBlocking));

	// configure the launch parameters, under the global domain
	uint _W_over_quo = _cu_ceil(dstDim.x, Sobel_Thr_y);
	uint _H_over_quo = _cu_ceil(dstDim.y, Sobel_Thr_x);
	dim3 grid(_H_over_quo, _W_over_quo);
	dim3 threads(1, Sobel_Thr_x * Sobel_Thr_y);

	// the parameters under the process domain
	uint _PW_over_quo = _cu_ceil(Proc.x, Sobel_Thr_y);
	uint _PH_over_quo = _cu_ceil(Proc.y, Sobel_Thr_x);

	dim3 P_grid(_PH_over_quo, _PW_over_quo);

	checkCudaErrors(cudaDeviceSynchronize());

	Sobel_base_x << <grid, threads, 0, S_K_0 >> > (dev_src,
		dev_base_x,
		dstDim,
		src_pitch);

	Sobel_base_y << <grid, threads, 0, S_K_1 >> > (dev_src,
		dev_base_y,
		dstDim,
		src_pitch);

	Scharr_x << <P_grid, threads, 0, S_K_0 >> > (dev_base_x,
		GI_tmp,
		step,
		dstDim,
		Proc,
		GI_pitch);

	Scharr_y << <P_grid, threads, 0, S_K_1 >> > (dev_base_y,
		GI_tmp,
		step,
		dstDim,
		Proc,
		GI_pitch);

	checkCudaErrors(cudaHostGetDevicePointer(&dev_dst, _dst.Mat, 0));

	checkCudaErrors(cudaDeviceSynchronize());
	//越界
	Sobel_SummingXY << <P_grid, threads, 0, S_K_0 >> > (GI_tmp,
		step,
		Proc,
		GI_pitch);

	Sobel_Final_Calc << <P_grid, threads, 0, S_K_0 >> > (GI_tmp,
		dev_dst,
		_L_thr,
		_H_thr,
		step,
		dstDim,
		Proc,
		GI_pitch);

	checkCudaErrors(cudaFree(dev_src));
	checkCudaErrors(cudaFree(dev_base_x));
	checkCudaErrors(cudaFree(GI_tmp));
	// synchronize S_K_0, and finish the final calculaation
	checkCudaErrors(cudaStreamSynchronize(S_K_0));

	checkCudaErrors(cudaStreamDestroy(S_K_1));
	checkCudaErrors(cudaStreamSynchronize(S_C));
	checkCudaErrors(cudaStreamDestroy(S_C));
}




void de::vis::_Canny_Sobel(de::vis::dev_Img& src, de::vis::dev_Img& dst, const float _L_thr, const float _H_thr, const uint step)
{
	if (cuP.is_init != true) {
		printf("CUDA should be initialized first\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	_Img& _src = dynamic_cast<_Img&>(src);
	_Img& _dst = dynamic_cast<_Img&>(dst);

	if (_src.channel != 1) {
		printf("The input Image array is not a 1D array\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	__dev_Img* sub_ptr_src = dynamic_cast<__dev_Img*>(&_src);
	__dev_Img* sub_ptr_dst = dynamic_cast<__dev_Img*>(&_dst);

	// ~.x : width; ~.y : height
	const dim3 srcDim(_src.width, _src.height);
	// ~.x : width; ~.y : height
	const dim3 dstDim = srcDim;

	uchar* dev_src = sub_ptr_src->dev_Mat,
		* dev_dst = sub_ptr_dst->dev_Mat;
	int* dev_base_x, * dev_base_y;

	GI* GI_tmp;
	size_t src_pitch = _src.width;
	size_t srcWidth = _src.width * sizeof(uchar);

	cudaStream_t S_K_0, S_K_1;

	const dim3 Proc(dstDim.x / step, dstDim.y / step);

	size_t GI_pitch;
	checkCudaErrors(cudaMallocPitch(reinterpret_cast<void**>(&GI_tmp), &GI_pitch, Proc.x * sizeof(GI), Proc.y));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_base_x), (srcDim.x * dstDim.y + dstDim.x * srcDim.y) * sizeof(int)));

	dev_base_y = dev_base_x + srcDim.x * dstDim.y;
	GI_pitch /= sizeof(GI);
	src_pitch /= sizeof(uchar);

	checkCudaErrors(cudaStreamCreateWithFlags(&S_K_0, cudaStreamNonBlocking));
	checkCudaErrors(cudaStreamCreateWithFlags(&S_K_1, cudaStreamNonBlocking));

	// configure the launch parameters, under the global domain
	uint _W_over_quo = _cu_ceil(dstDim.x, Sobel_Thr_y);
	uint _H_over_quo = _cu_ceil(dstDim.y, Sobel_Thr_x);
	dim3 grid(_H_over_quo, _W_over_quo);
	dim3 threads(1, Sobel_Thr_x * Sobel_Thr_y);

	// the parameters under the process domain
	uint _PW_over_quo = _cu_ceil(Proc.x, Sobel_Thr_y);
	uint _PH_over_quo = _cu_ceil(Proc.y, Sobel_Thr_x);

	dim3 P_grid(_PH_over_quo, _PW_over_quo);

	checkCudaErrors(cudaDeviceSynchronize());

	Sobel_base_x << <grid, threads, 0, S_K_0 >> > (dev_src,
		dev_base_x,
		dstDim,
		src_pitch);

	Sobel_base_y << <grid, threads, 0, S_K_1 >> > (dev_src,
		dev_base_y,
		dstDim,
		src_pitch);

	Soble_x << <P_grid, threads, 0, S_K_0 >> > (dev_base_x,
		GI_tmp,
		step,
		dstDim,
		Proc,
		GI_pitch);

	Soble_y << <P_grid, threads, 0, S_K_1 >> > (dev_base_y,
		GI_tmp,
		step,
		dstDim,
		Proc,
		GI_pitch);

	checkCudaErrors(cudaDeviceSynchronize());

	Sobel_SummingXY << <P_grid, threads, 0, S_K_0 >> > (GI_tmp,
		step,
		Proc,
		GI_pitch);

	Sobel_Final_Calc << <P_grid, threads, 0, S_K_0 >> > (GI_tmp,
		dev_dst,
		_L_thr,
		_H_thr,
		step,
		dstDim,
		Proc,
		GI_pitch);

	checkCudaErrors(cudaFree(dev_base_x));
	checkCudaErrors(cudaFree(GI_tmp));
	// synchronize S_K_0, and finish the final calculaation
	checkCudaErrors(cudaStreamSynchronize(S_K_0));

	checkCudaErrors(cudaStreamDestroy(S_K_1));
}



void de::vis::_Canny_Prewitt(de::vis::dev_Img& src, de::vis::dev_Img& dst, const float _L_thr, const float _H_thr, const uint step)
{
	if (cuP.is_init != true) {
		printf("CUDA should be initialized first\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	__dev_Img& _src = dynamic_cast<__dev_Img&>(src);
	__dev_Img& _dst = dynamic_cast<__dev_Img&>(dst);

	if (_src.channel != 1) {
		printf("The input Image array is not a 1D array\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	// ~.x : width; ~.y : height
	const dim3 srcDim(_src.width, _src.height);
	// ~.x : width; ~.y : height
	const dim3 dstDim = srcDim;

	uchar* dev_src = _src.dev_Mat,
		* dev_dst = _dst.dev_Mat;
	int* dev_base_x, * dev_base_y;

	GI* GI_tmp;
	size_t src_pitch = _src.width;
	size_t srcWidth = _src.width * sizeof(uchar);

	cudaStream_t S_K_0, S_K_1;

	const dim3 Proc(dstDim.x / step, dstDim.y / step);

	size_t GI_pitch;
	checkCudaErrors(cudaMallocPitch(reinterpret_cast<void**>(&GI_tmp), &GI_pitch, Proc.x * sizeof(GI), Proc.y));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_base_x), (srcDim.x * dstDim.y + dstDim.x * srcDim.y) * sizeof(int)));

	dev_base_y = dev_base_x + srcDim.x * dstDim.y;
	GI_pitch /= sizeof(GI);
	src_pitch /= sizeof(uchar);

	checkCudaErrors(cudaStreamCreateWithFlags(&S_K_0, cudaStreamNonBlocking));
	checkCudaErrors(cudaStreamCreateWithFlags(&S_K_1, cudaStreamNonBlocking));

	// configure the launch parameters, under the global domain
	uint _W_over_quo = _cu_ceil(dstDim.x, Sobel_Thr_y);
	uint _H_over_quo = _cu_ceil(dstDim.y, Sobel_Thr_x);
	dim3 grid(_H_over_quo, _W_over_quo);
	dim3 threads(1, Sobel_Thr_x * Sobel_Thr_y);

	// the parameters under the process domain
	uint _PW_over_quo = _cu_ceil(Proc.x, Sobel_Thr_y);
	uint _PH_over_quo = _cu_ceil(Proc.y, Sobel_Thr_x);

	dim3 P_grid(_PH_over_quo, _PW_over_quo);

	checkCudaErrors(cudaDeviceSynchronize());

	Sobel_base_x << <grid, threads, 0, S_K_0 >> > (dev_src,
		dev_base_x,
		dstDim,
		src_pitch);

	Sobel_base_y << <grid, threads, 0, S_K_1 >> > (dev_src,
		dev_base_y,
		dstDim,
		src_pitch);

	Prewitt_x << <P_grid, threads, 0, S_K_0 >> > (dev_base_x,
		GI_tmp,
		step,
		dstDim,
		Proc,
		GI_pitch);

	Prewitt_y << <P_grid, threads, 0, S_K_1 >> > (dev_base_y,
		GI_tmp,
		step,
		dstDim,
		Proc,
		GI_pitch);

	checkCudaErrors(cudaDeviceSynchronize());

	Sobel_SummingXY << <P_grid, threads, 0, S_K_0 >> > (GI_tmp,
		step,
		Proc,
		GI_pitch);

	Sobel_Final_Calc << <P_grid, threads, 0, S_K_0 >> > (GI_tmp,
		dev_dst,
		_L_thr,
		_H_thr,
		step,
		dstDim,
		Proc,
		GI_pitch);

	checkCudaErrors(cudaFree(dev_base_x));
	checkCudaErrors(cudaFree(GI_tmp));
	// synchronize S_K_0, and finish the final calculaation
	checkCudaErrors(cudaStreamSynchronize(S_K_0));

	checkCudaErrors(cudaStreamDestroy(S_K_1));
}




void de::vis::_Canny_Scharr(de::vis::dev_Img& src, de::vis::dev_Img& dst, const float _L_thr, const float _H_thr, const uint step)
{
	if (cuP.is_init != true) {
		printf("CUDA should be initialized first\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	_Img& _src = dynamic_cast<_Img&>(src);
	_Img& _dst = dynamic_cast<_Img&>(dst);

	if (_src.channel != 1) {
		printf("The input Image array is not a 1D array\n");
		system("pause");
		exit(EXIT_FAILURE);
	}

	__dev_Img* sub_ptr_src = dynamic_cast<__dev_Img*>(&_src);
	__dev_Img* sub_ptr_dst = dynamic_cast<__dev_Img*>(&_dst);

	// ~.x : width; ~.y : height
	const dim3 srcDim(_src.width, _src.height);
	// ~.x : width; ~.y : height
	const dim3 dstDim = srcDim;

	uchar* dev_src = sub_ptr_src->dev_Mat,
		* dev_dst = sub_ptr_dst->dev_Mat;
	int* dev_base_x, * dev_base_y;

	GI* GI_tmp;
	size_t src_pitch = _src.width;
	size_t srcWidth = _src.width * sizeof(uchar);

	cudaStream_t S_K_0, S_K_1;

	const dim3 Proc(dstDim.x / step, dstDim.y / step);

	size_t GI_pitch;
	checkCudaErrors(cudaMallocPitch(reinterpret_cast<void**>(&GI_tmp), &GI_pitch, Proc.x * sizeof(GI), Proc.y));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_base_x), (srcDim.x * dstDim.y + dstDim.x * srcDim.y) * sizeof(int)));

	dev_base_y = dev_base_x + srcDim.x * dstDim.y;
	GI_pitch /= sizeof(GI);
	src_pitch /= sizeof(uchar);

	checkCudaErrors(cudaStreamCreateWithFlags(&S_K_0, cudaStreamNonBlocking));
	checkCudaErrors(cudaStreamCreateWithFlags(&S_K_1, cudaStreamNonBlocking));

	// configure the launch parameters, under the global domain
	uint _W_over_quo = _cu_ceil(dstDim.x, Sobel_Thr_y);
	uint _H_over_quo = _cu_ceil(dstDim.y, Sobel_Thr_x);
	dim3 grid(_H_over_quo, _W_over_quo);
	dim3 threads(1, Sobel_Thr_x * Sobel_Thr_y);

	// the parameters under the process domain
	uint _PW_over_quo = _cu_ceil(Proc.x, Sobel_Thr_y);
	uint _PH_over_quo = _cu_ceil(Proc.y, Sobel_Thr_x);

	dim3 P_grid(_PH_over_quo, _PW_over_quo);

	checkCudaErrors(cudaDeviceSynchronize());

	Sobel_base_x << <grid, threads, 0, S_K_0 >> > (dev_src,
		dev_base_x,
		dstDim,
		src_pitch);

	Sobel_base_y << <grid, threads, 0, S_K_1 >> > (dev_src,
		dev_base_y,
		dstDim,
		src_pitch);

	Scharr_x << <P_grid, threads, 0, S_K_0 >> > (dev_base_x,
		GI_tmp,
		step,
		dstDim,
		Proc,
		GI_pitch);

	Scharr_y << <P_grid, threads, 0, S_K_1 >> > (dev_base_y,
		GI_tmp,
		step,
		dstDim,
		Proc,
		GI_pitch);

	checkCudaErrors(cudaDeviceSynchronize());

	Sobel_SummingXY << <P_grid, threads, 0, S_K_0 >> > (GI_tmp,
		step,
		Proc,
		GI_pitch);

	Sobel_Final_Calc << <P_grid, threads, 0, S_K_0 >> > (GI_tmp,
		dev_dst,
		_L_thr,
		_H_thr,
		step,
		dstDim,
		Proc,
		GI_pitch);

	checkCudaErrors(cudaFree(dev_base_x));
	checkCudaErrors(cudaFree(GI_tmp));
	// synchronize S_K_0, and finish the final calculaation
	checkCudaErrors(cudaStreamSynchronize(S_K_0));

	checkCudaErrors(cudaStreamDestroy(S_K_1));
}