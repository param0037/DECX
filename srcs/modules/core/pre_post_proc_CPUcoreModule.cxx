/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "thread_management/thread_pool.h"
#include "memory_management/MemoryPool_Hv.h"
#include "resources_manager/decx_resource.h"


#ifdef _MSC_VER
bool WINAPI DllMain(HMODULE hModule,
					DWORD	ul_reason_for_call,
					LPVOID	lpReserved)
{
	switch (ul_reason_for_call)
	{
	case DLL_PROCESS_ATTACH:
		/*
		* Note : The memory pool should be initialized first, since the dynamic_array in
		* threadpool is based on the host memory pool.
		*/
		decx::mem_pool_Hv = decx::MemPool_Hv::GetInstance();
		decx::thread_pool = new decx::ThreadPool(std::thread::hardware_concurrency(), true);
		decx::_res_mgr = new decx::ResMgr;
		break;
	case DLL_PROCESS_DETACH:

		delete decx::thread_pool;
		delete decx::_res_mgr;
		break;
	default:
		break;
	}
	return true;
}
#endif



#ifdef __GNUC__
__attribute__((constructor)) void InitHost_Core_Resources()
{
	/*
	* Note : The memory pool should be initialized first, since the dynamic_array in
	* threadpool is based on the host memory pool.
	*/
	decx::mem_pool_Hv = decx::MemPool_Hv::GetInstance();
	decx::thread_pool = new decx::ThreadPool(std::thread::hardware_concurrency(), true);
	decx::_res_mgr = new decx::ResMgr;
}


__attribute__((destructor)) void FreeHost_Core_Resources()
{
	delete decx::thread_pool;
	delete decx::_res_mgr;
}

#endif