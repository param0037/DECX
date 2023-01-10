#pragma once


#include "../../classes/core_types.h"



template<typename T>
__global__
void cu_exp(T* src, T* dst)
{
	uint index = threadIdx.x + blockIdx.x * blockDim.x;
	T tmp = src[index];
	dst[index] = __expf(tmp);
}


template<typename T>
__global__
void cu_exp_offset(T* src, T* dst, const uint offset)
{
	uint index = offset + threadIdx.x + blockIdx.x * blockDim.x;
	T tmp = src[index];
	dst[index] = __expf(tmp);
}


__global__
void cu_exp_fp16(half2* src, half2* dst, const size_t __len)
{
#if __ABOVE_SM_53
	size_t index = threadIdx.x + blockIdx.x * blockDim.x;
	if (index < __len) {
		half2 tmp = src[index];
		tmp = h2exp(tmp);
		dst[index] = tmp;
	}
#endif
}


template <typename T>
static void Kexp(T*				Hsrc,
				 T*				Hdst,
				 const size_t	_total)
{
	cudaStream_t stream_0,
		stream_1;
	checkCudaErrors(cudaStreamCreateWithFlags(&stream_0, cudaStreamNonBlocking));
	checkCudaErrors(cudaStreamCreateWithFlags(&stream_1, cudaStreamNonBlocking));

	T* dev_src, * dev_dst;
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_src), _total * __SPACE__));
	checkCudaErrors(cudaMalloc(reinterpret_cast<void**>(&dev_dst), _total * __SPACE__));
	// copy the datas from host to device
	checkCudaErrors(cudaMemcpyAsync(dev_src, Hsrc, _total * __SPACE__, cudaMemcpyHostToDevice, stream_0));


	// configure the launch parameters
	const uint max_tpb = cuP.max_tpb;
	Num_uint conf(_total, max_tpb);

	bool is_left = (conf._mod != 0);

	uint threads = max_tpb;
	uint grid = conf.unsat_quo;

	checkCudaErrors(cudaDeviceSynchronize());
	cu_exp << <grid, threads, 0, stream_1 >> > (dev_src, dev_dst);

	if (is_left) {
		cu_exp_offset << <1, conf._mod, 0, stream_0 >> > (dev_src, dev_dst, conf.unsatur);
	}
	// synchronize the streams
	checkCudaErrors(cudaDeviceSynchronize());

	// copy back the datas from device to host
	checkCudaErrors(cudaMemcpyAsync(Hdst, dev_dst, _total * __SPACE__, cudaMemcpyDeviceToHost, stream_0));
	// free the device memories
	checkCudaErrors(cudaFree(dev_src));

	checkCudaErrors(cudaStreamDestroy(stream_1));

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaFree(dev_dst));
	checkCudaErrors(cudaStreamDestroy(stream_0));
}



static void Kexp_fp16(de::Half*			Hsrc,
					  de::Half*			Hdst,
					  const size_t		_total)
{
	// make the length of device array even
	const size_t total = _total % 2 == 0 ? _total : _total + 1;

	half2* dev_src, * dev_dst;
	checkCudaErrors(cudaMalloc(&dev_src, total * sizeof(half)));
	checkCudaErrors(cudaMalloc(&dev_dst, total * sizeof(half)));

	cudaStream_t S;
	checkCudaErrors(cudaStreamCreate(&S));

	checkCudaErrors(cudaMemcpyAsync(
		dev_src, Hsrc, _total * sizeof(de::Half), cudaMemcpyHostToDevice, S));

	cu_exp_fp16 << <decx::utils::ceil<size_t>(total / 2, cuP.max_tpb), cuP.max_tpb, 0, S >> > (
		dev_src, dev_dst, total / 2);

	checkCudaErrors(cudaMemcpyAsync(
		Hdst, dev_dst, _total * sizeof(de::Half), cudaMemcpyDeviceToHost, S));

	checkCudaErrors(cudaDeviceSynchronize());

	checkCudaErrors(cudaFree(dev_src));
	checkCudaErrors(cudaFree(dev_dst));
	checkCudaErrors(cudaStreamDestroy(S));
}



// ----------------------------------------------------------------------------------
//									DEVICE
// ----------------------------------------------------------------------------------


template <typename T>
static void dev_Kexp(T*				Dsrc,
					 T*				Ddst,
					 const size_t	_total)
{
	cudaStream_t stream_0,
		stream_1;
	checkCudaErrors(cudaStreamCreateWithFlags(&stream_0, cudaStreamNonBlocking));
	checkCudaErrors(cudaStreamCreateWithFlags(&stream_1, cudaStreamNonBlocking));

	// configure the launch parameters
	const uint max_tpb = cuP.max_tpb;
	Num_uint conf(_total, max_tpb);

	bool is_left = (conf._mod != 0);

	uint threads = max_tpb;
	uint grid = conf.unsat_quo;

	checkCudaErrors(cudaDeviceSynchronize());
	cu_exp << <grid, threads, 0, stream_1 >> > (Dsrc, Ddst);

	if (is_left) {
		cu_exp_offset << <1, conf._mod, 0, stream_0 >> > (Dsrc, Ddst, conf.unsatur);
	}
	// synchronize the streams
	checkCudaErrors(cudaDeviceSynchronize());

	// free the device memories
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaStreamDestroy(stream_0));
	checkCudaErrors(cudaStreamDestroy(stream_1));
}



static void dev_Kexp_fp16(de::Half*			Dsrc,
					      de::Half*			Ddst,
					      const size_t		_total)		// _total have included the 2N padding
{
	// make the length of device array even
	cudaStream_t S;
	checkCudaErrors(cudaStreamCreate(&S));

	cu_exp_fp16 << <decx::utils::ceil<size_t>(_total / 2, cuP.max_tpb), cuP.max_tpb, 0, S >> > (
		reinterpret_cast<half2*>(Dsrc),
		reinterpret_cast<half2*>(Ddst),
		_total / 2);

	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaStreamDestroy(S));
}