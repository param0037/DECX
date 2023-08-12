/**
*	---------------------------------------------------------------------
*	Author : Wayne Anderson
*   Date   : 2021.04.16
*	---------------------------------------------------------------------
*	This is a part of the open source program named "DECX", copyright c Wayne,
*	2021.04.16
*/


#ifndef _BASIC_H_
#define _BASIC_H_

#include <iostream>
#include <initializer_list>
//#include <Windows.h>


typedef unsigned char uchar;
typedef unsigned int uint;


#if defined(WIN64) || defined(_WIN64) || defined(_WIN64_) || defined(WIN32)
#define Windows
#endif



#if defined(__linux__) || defined(__GNUC__)
#define Linux
#endif


#ifdef Windows
#define _DECX_API_ __declspec(dllexport)
#define __align__(n) __declspec(align(n))
#endif

#ifdef Linux
#define _DECX_API_ __attribute__((visibility("default")))
#define __align__(n) __attribute__(align(n))
#endif

namespace de
{
	typedef struct DECX_Handle
	{
		int error_type;
		char error_string[100];
	}DH;
}


namespace de
{
	_DECX_API_ void InitCuda();


	_DECX_API_ void InitCPUInfo();

	namespace cuda {
		_DECX_API_ void DECX_CUDA_exit();
	}

	namespace cpu {
		_DECX_API_ de::DH DecxSetThreadingNum(const size_t _thread_num);
	}
}


namespace de
{
	// Realized by DECX_allocations
	_DECX_API_ void DecxEnableLogPrint();


	// Realized by DECX_allocations
	_DECX_API_ void DecxDisableLogPrint();


	_DECX_API_ void DecxEnableWarningPrint();


	_DECX_API_ void DecxDisableWarningPrint();


	_DECX_API_ void DecxEnableSuccessfulPrint();


	_DECX_API_ void DecxDisableSuccessfulPrint();
}


namespace de
{
	enum DECX_error_types
	{
		DECX_SUCCESS = 0x00,

		DECX_FAIL_not_init = 0x01,

		DECX_FAIL_FFT_error_length = 0x02,

		DECX_FAIL_DimsNotMatching = 0x03,
		DECX_FAIL_Complex_comparing = 0x04,

		DECX_FAIL_ConvBadKernel = 0x05,
		DECX_FAIL_StepError = 0x06,

		DECX_FAIL_ChannelError = 0x07,
		DECX_FAIL_CVNLM_BadArea = 0x08,

		DECX_FAIL_FileNotExist = 0x09,

		DECX_GEMM_DimsError = 0x0a,

		DECX_FAIL_ErrorFlag = 0x0b,

		DECX_FAIL_DimError = 0x0c,

		DECX_FAIL_ErrorParams = 0x0d,

		DECX_FAIL_StoreError = 0x0e,

		DECX_FAIL_MNumNotMatching = 0x0f,

		DECX_FAIL_ALLOCATION = 0x10
	};
}


#endif
