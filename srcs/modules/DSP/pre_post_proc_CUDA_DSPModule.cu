/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "FFT/CUDA/CUDA_FFTs.cuh"


#ifdef _MSC_VER
bool WINAPI DllMain(HMODULE hModule,
                    DWORD	ul_reason_for_call,
                    LPVOID	lpReserved)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
		decx::dsp::InitCUDA_FFT3D_Resources();
		decx::dsp::InitCUDA_FFT2D_Resources();
		decx::dsp::InitCUDA_FFT1D_Resources();
		break;
	case DLL_PROCESS_DETACH:
		decx::dsp::FreeCUDA_FFT3D_Resources();
		decx::dsp::FreeCUDA_FFT2D_Resources();
		decx::dsp::FreeCUDA_FFT1D_Resources();
		break;
	case DLL_THREAD_ATTACH:
		/*decx::dsp::InitCUDA_FFT3D_Resources();
		decx::dsp::InitCUDA_FFT2D_Resources();
		decx::dsp::InitCUDA_FFT1D_Resources();*/
		break;
	case DLL_THREAD_DETACH:
		/*decx::dsp::FreeCUDA_FFT3D_Resources();
		decx::dsp::FreeCUDA_FFT2D_Resources();
		decx::dsp::FreeCUDA_FFT1D_Resources();*/
		break;
	default:
		break;
	}
	return true;
}
#endif


#ifdef __GNUC__
__attribute__((constructor)) void InitCUDA_DSP_Resources()
{
	decx::dsp::InitCUDA_FFT3D_Resources();
	decx::dsp::InitCUDA_FFT2D_Resources();
	decx::dsp::InitCUDA_FFT1D_Resources();
}


__attribute__((destructor)) void FreeCUDA_DSP_Resources()
{
	decx::dsp::FreeCUDA_FFT3D_Resources();
	decx::dsp::FreeCUDA_FFT2D_Resources();
	decx::dsp::FreeCUDA_FFT1D_Resources();
}

#endif