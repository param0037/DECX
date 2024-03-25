/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "convolution/CUDA/fp32/cuda_conv2D_fp32_im2col_planner.cuh"


#ifdef _MSC_VER
bool WINAPI DllMain(HMODULE hModule,
                    DWORD	ul_reason_for_call,
                    LPVOID	lpReserved)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
		decx::nn::InitCUDAConv2DResource();
		break;
	case DLL_PROCESS_DETACH:
		break;
	default:
		break;
	}
	return true;
}
#endif


#ifdef __GNUC__
__attribute__((destructor)) void InitCPU_DSP_Resources()
{
	decx::nn::InitCUDAConv2DResource();
}

//
//__attribute__((destructor)) void FreeCPU_DSP_Resources()
//{
//	decx::dsp::FreeFFT3Resources();
//	decx::dsp::FreeFFT2Resources();
//	decx::dsp::FreeFFT1Resources();
//}

#endif