/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
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