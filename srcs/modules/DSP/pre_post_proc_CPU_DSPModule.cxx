/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "FFT/CPU/2D/FFT2D.h"


#ifdef _MSC_VER
bool WINAPI DllMain(HMODULE hModule,
                    DWORD	ul_reason_for_call,
                    LPVOID	lpReserved)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
		decx::dsp::InitFFT3Resources();
		decx::dsp::InitFFT2Resources();
		decx::dsp::InitFFT1Resources();
		break;
	case DLL_PROCESS_DETACH:
		decx::dsp::FreeFFT3Resources();
		decx::dsp::FreeFFT2Resources();
		decx::dsp::FreeFFT1Resources();
		break;
	case DLL_THREAD_ATTACH:
		/*decx::dsp::InitFFT3Resources();
		decx::dsp::InitFFT2Resources();
		decx::dsp::InitFFT1Resources();*/
		break;
	case DLL_THREAD_DETACH:
		/*decx::dsp::FreeFFT3Resources();
		decx::dsp::FreeFFT2Resources();
		decx::dsp::FreeFFT1Resources();*/
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
	decx::dsp::InitFFT3Resources();
	decx::dsp::InitFFT2Resources();
	decx::dsp::InitFFT1Resources();
}


__attribute__((destructor)) void FreeCPU_DSP_Resources()
{
	decx::dsp::FreeFFT3Resources();
	decx::dsp::FreeFFT2Resources();
	decx::dsp::FreeFFT1Resources();
}

#endif