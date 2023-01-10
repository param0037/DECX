/**
*	---------------------------------------------------------------------
*	Author : Wayne
*   Date   : 2021.9.16
*	---------------------------------------------------------------------
*	This is a part of the open source program named "DECX", copyright c Wayne,
*	2021.9.16
*/
#pragma once

#include "../cv_classes/cv_classes.h"
#include "../cv_classes/cv_cls_MFuncs.h"


namespace de
{
	namespace vis
	{
		_DECX_API_
		de::DH ImgToMatrix(de::vis::Img& src, _FLOAT_& dst, const int flag, const int thread_num = 1);


		_DECX_API_
		de::DH ImgToMatrix(de::vis::Img& src, _INT_& dst, const int flag, const int thread_num = 1);


		_DECX_API_
		de::DH ImgToMatrix(de::vis::Img& src, _DOUBLE_& dst, const int flag, const int thread_num = 1);


		_DECX_API_
		de::DH ImgToMatrix(de::vis::Img& src, _UCHAR_& dst);


		_DECX_API_
		de::DH ImgToTensor(de::vis::Img& src, T_FLOAT& dst, const int flag, const int thread_num = 1);


		_DECX_API_
		de::DH ImgToTensor(de::vis::Img& src, T_INT& dst, const int flag, const int thread_num = 1);


		_DECX_API_
		de::DH ImgToTensor(de::vis::Img& src, T_DOUBLE& dst, const int flag, const int thread_num = 1);


		_DECX_API_
		de::DH ImgToTensor(de::vis::Img& src, T_UCHAR& dst);




		_DECX_API_
		de::DH MatrixToImg(_FLOAT_& src, de::vis::Img& dst, const int flag, const int thread_num = 1);
	}
}


template <typename T>
static
void _Host_Convert_MT(uchar* src, T* dst, const size_t offset, const size_t proc_len)
{
	for (size_t i = offset; i < proc_len + offset; ++i) {
		dst[i] = static_cast<T>(src[i]);
	}
}


template <typename T>
static
void Host_Convert_MT(uchar *src, T *dst, const size_t num, const int thr_num)
{
	//std::vector<std::thread> thr_arr;
	std::vector<std::future<void>> thr_arr;
	Num_size_t __conf(num, static_cast<size_t>(thr_num));

	size_t offset = 0;
	/*for (int i = 0; i < thr_num; ++i) {
		thr_arr.push_back(std::thread(_Host_Convert_MT<T>, src, dst, offset, __conf.unsat_quo));
		offset += __conf.unsat_quo;
	}

	if (__conf._mod != 0) {
		thr_arr.push_back(std::thread(_Host_Convert_MT<T>, src, dst, offset, __conf._mod));
	}

	for (int i = 0; i < thr_arr.size(); ++i) {
		thr_arr[i].join();
	}*/

	for (int i = 0; i < thr_num; ++i) {
		thr_arr.emplace_back(decx::thread_pool.register_task(_Host_Convert_MT<T>, src, dst, offset, __conf.unsat_quo));
		offset += __conf.unsat_quo;
	}
	if (__conf._mod != 0) {
		thr_arr.emplace_back(decx::thread_pool.register_task(_Host_Convert_MT<T>, src, dst, offset, __conf._mod));
	}
	for (int i = 0; i < thr_arr.size(); ++i) {
		thr_arr[i].get();
	}
}


template <typename T>
static
void Host_Convert_ST(uchar* src, T* dst, const size_t num)
{
	for (size_t i = 0; i < num; ++i) {
		dst[i] = static_cast<T>(src[i]);
	}
}


static void convert_error(de::DH *handle)
{
	handle->error_string = "The channel of image must be one";
	handle->error_type = decx::DECX_FAIL_DimError;
}



de::DH de::vis::ImgToMatrix(de::vis::Img& src, _FLOAT_& dst, const int flag, const int thread_num)
{
	de::DH handle;

	_Img& _src = dynamic_cast<_Img&>(src);
	_Matrix<float>& sub_dst = dynamic_cast<_Matrix<float>&>(dst);
	
	if (_src.channel != 1) {
		convert_error(&handle);
	}

	const size_t len = _src.element_num;
	switch (flag)
	{
	case SINGLE_HOST_THREADS:
		Host_Convert_ST<float>(_src.Mat, sub_dst.Mat.ptr, thread_num);
		break;

	case MULTI_HOST_THREADS:
		Host_Convert_MT<float>(_src.Mat, sub_dst.Mat.ptr, len, thread_num);
		break;
	default:
		decx::MeaninglessFlag(&handle);
		break;
	}
	decx::Success(&handle);
	return handle;
}





de::DH de::vis::ImgToMatrix(de::vis::Img& src, _INT_& dst, const int flag, const int thread_num)
{
	de::DH handle;

	_Img& _src = dynamic_cast<_Img&>(src);
	_Matrix<int>& sub_dst = dynamic_cast<_Matrix<int>&>(dst);

	if (_src.channel != 1) {
		convert_error(&handle);
	}

	const size_t len = _src.element_num;
	switch (flag)
	{
	case SINGLE_HOST_THREADS:
		Host_Convert_ST<int>(_src.Mat, sub_dst.Mat.ptr, thread_num);
		break;

	case MULTI_HOST_THREADS:
		Host_Convert_MT<int>(_src.Mat, sub_dst.Mat.ptr, len, thread_num);
		break;
	default:
		decx::MeaninglessFlag(&handle);
		break;
	}
	decx::Success(&handle);
	return handle;
}



de::DH de::vis::ImgToMatrix(de::vis::Img& src, _DOUBLE_& dst, const int flag, const int thread_num)
{
	de::DH handle;

	_Img& _src = dynamic_cast<_Img&>(src);
	_Matrix<double>& sub_dst = dynamic_cast<_Matrix<double>&>(dst);

	if (_src.channel != 1) {
		convert_error(&handle);
	}

	const size_t len = _src.element_num;
	switch (flag)
	{
	case SINGLE_HOST_THREADS:
		Host_Convert_ST<double>(_src.Mat, sub_dst.Mat.ptr, thread_num);
		break;

	case MULTI_HOST_THREADS:
		Host_Convert_MT<double>(_src.Mat, sub_dst.Mat.ptr, len, thread_num);
		break;
	default:
		decx::MeaninglessFlag(&handle);
		break;
	}
	decx::Success(&handle);
	return handle;
}




de::DH de::vis::ImgToMatrix(de::vis::Img& src, _UCHAR_& dst)
{
	de::DH handle;

	_Img& _src = dynamic_cast<_Img&>(src);
	_Matrix<double>& sub_dst = dynamic_cast<_Matrix<double>&>(dst);

	if (_src.channel != 1) {
		convert_error(&handle);
	}

	const size_t bits = _src.total_bytes;
	memcpy(sub_dst.Mat.ptr, _src.Mat, bits);

	decx::Success(&handle);
	return handle;
}





de::DH de::vis::ImgToTensor(de::vis::Img& src, T_FLOAT& dst, const int flag, const int thread_num)
{
	de::DH handle;

	_Img& _src = dynamic_cast<_Img&>(src);
	_Tensor<float>& sub_dst = dynamic_cast<_Tensor<float>&>(dst);

	if (_src.channel != 4) {
		convert_error(&handle);
	}

	const size_t len = _src.element_num;
	switch (flag)
	{
	case SINGLE_HOST_THREADS:
		Host_Convert_ST<float>(_src.Mat, sub_dst.Tens, thread_num);
		break;

	case MULTI_HOST_THREADS:
		Host_Convert_MT<float>(_src.Mat, sub_dst.Tens, len, thread_num);
		break;
	default:
		decx::MeaninglessFlag(&handle);
		break;
	}
	decx::Success(&handle);
	return handle;
}





de::DH de::vis::ImgToTensor(de::vis::Img& src, T_INT& dst, const int flag, const int thread_num)
{
	de::DH handle;

	_Img& _src = dynamic_cast<_Img&>(src);
	_Tensor<int>& sub_dst = dynamic_cast<_Tensor<int>&>(dst);

	if (_src.channel != 1) {
		convert_error(&handle);
	}

	const size_t len = _src.element_num;
	switch (flag)
	{
	case SINGLE_HOST_THREADS:
		Host_Convert_ST<int>(_src.Mat, sub_dst.Tens, thread_num);
		break;

	case MULTI_HOST_THREADS:
		Host_Convert_MT<int>(_src.Mat, sub_dst.Tens, len, thread_num);
		break;
	default:
		decx::MeaninglessFlag(&handle);
		break;
	}
	decx::Success(&handle);
	return handle;
}



de::DH de::vis::ImgToTensor(de::vis::Img& src, T_DOUBLE& dst, const int flag, const int thread_num)
{
	de::DH handle;

	_Img& _src = dynamic_cast<_Img&>(src);
	_Tensor<double>& sub_dst = dynamic_cast<_Tensor<double>&>(dst);

	if (_src.channel != 1) {
		convert_error(&handle);
	}

	const size_t len = _src.element_num;
	switch (flag)
	{
	case SINGLE_HOST_THREADS:
		Host_Convert_ST<double>(_src.Mat, sub_dst.Tens, thread_num);
		break;

	case MULTI_HOST_THREADS:
		Host_Convert_MT<double>(_src.Mat, sub_dst.Tens, len, thread_num);
		break;
	default:
		decx::MeaninglessFlag(&handle);
		break;
	}
	decx::Success(&handle);
	return handle;
}




de::DH de::vis::ImgToTensor(de::vis::Img& src, T_UCHAR& dst)
{
	de::DH handle;

	_Img& _src = dynamic_cast<_Img&>(src);
	_Tensor<double>& sub_dst = dynamic_cast<_Tensor<double>&>(dst);

	if (_src.channel != 1) {
		convert_error(&handle);
	}

	const size_t bits = _src.total_bytes;
	memcpy(sub_dst.Tens, _src.Mat, bits);

	decx::Success(&handle);
	return handle;
}