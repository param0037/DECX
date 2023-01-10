#pragma once


#include "../../classes/core_types.h"
#include "Kabs.h"


namespace de
{
	template<typename T>
	de::DH abs(__TYPE__& src, __TYPE__& dst);


	template<typename T>
	de::DH abs(_T_TYPE_& src, _T_TYPE_& dst);


	_DECX_API_ de::DH abs_fp16(_HALF_& src, _HALF_& dst);


	_DECX_API_ de::DH abs_fp16(T_HALF& src, T_HALF& dst);
}


template<typename T>
de::DH de::abs(__TYPE__& src, __TYPE__& dst)
{
	de::DH handle;
	if (!cuP.is_init) {
		decx::Not_init(&handle);
		return handle;
	}

	_Matrix<T>* _src = dynamic_cast<_Matrix<T>*>(&src);
	_Matrix<T>* _dst = dynamic_cast<_Matrix<T>*>(&dst);

	const size_t _total = _src->element_num;

	Kabs<T>(_src->Mat.ptr, _dst->Mat.ptr, _total);

	decx::Success(&handle);
	return handle;
}

template _DECX_API_ de::DH de::abs(_INT_& src, _INT_& dst);

template _DECX_API_ de::DH de::abs(_FLOAT_& src, _FLOAT_& dst);

template _DECX_API_ de::DH de::abs(_DOUBLE_& src, _DOUBLE_& dst);



template<typename T>
de::DH de::abs(_T_TYPE_& src, _T_TYPE_& dst)
{
	_Tensor<T>* _src = dynamic_cast<_Tensor<T>*>(&src);
	_Tensor<T>* _dst = dynamic_cast<_Tensor<T>*>(&dst);

	const size_t _total = _src->element_num;
	de::DH handle;
	if (!cuP.is_init) {
		decx::Not_init(&handle);
		return handle;
	}

	Kabs<T>(_src->Tens, _dst->Tens, _total);

	decx::Success(&handle);
	return handle;
}

template _DECX_API_ de::DH de::abs(T_INT& src, T_INT& dst);

template _DECX_API_ de::DH de::abs(T_FLOAT& src, T_FLOAT& dst);

template _DECX_API_ de::DH de::abs(T_DOUBLE& src, T_DOUBLE& dst);



de::DH de::abs_fp16(_HALF_& src, _HALF_& dst)
{
	de::DH handle;
	if (!cuP.is_init) {
		decx::Not_init(&handle);
		return handle;
	}

	_Matrix<de::Half>* _src = dynamic_cast<_Matrix<de::Half>*>(&src);
	_Matrix<de::Half>* _dst = dynamic_cast<_Matrix<de::Half>*>(&dst);

	const size_t true_total = _src->element_num % 2 ? _src->element_num + 1 : 
		_src->element_num;
	// make the length of device array even
	Kabs_fp16(_src->Mat.ptr, _dst->Mat.ptr, true_total);
	
	decx::Success(&handle);
	return handle;
}



de::DH de::abs_fp16(T_HALF& src, T_HALF& dst)
{
	de::DH handle;
	if (!cuP.is_init) {
		decx::Not_init(&handle);
		return handle;
	}

	_Tensor<de::Half>* _src = dynamic_cast<_Tensor<de::Half>*>(&src);
	_Tensor<de::Half>* _dst = dynamic_cast<_Tensor<de::Half>*>(&dst);

	const size_t true_total = _src->element_num % 2 ? _src->element_num + 1 :
		_src->element_num;
	// make the length of device array even
	Kabs_fp16(_src->Tens, _dst->Tens, true_total);

	decx::Success(&handle);
	return handle;
}