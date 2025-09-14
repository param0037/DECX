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

#include "decx_handle_impl.h"
#include <log_console.h>
#include <thread_management/utils/global_mutex_array.h>

#define MODULE_TAG "Handle"

namespace decx
{
static de::DECX_Handle g_last_handle;
}


_DECX_API_ int32_t DecxAssignLastHandle(const DecxErrorTypes_e _error_type, const char* _err_statement)
{
    DecxGlobalMtx_Lock(Decx_GMtxId_LastHandle);

    decx::g_last_handle.error_type = _error_type;
    decx::utils::decx_strcpy<100>(decx::g_last_handle.error_string, _err_statement);

    DecxGlobalMtx_Unlock(Decx_GMtxId_LastHandle);
    return 0;
}


_DECX_API_ int32_t DecxGetLastErrStatus(DecxErrorTypes_e* p_error_type)
{
    if (p_error_type == nullptr){
        DECX_LOG_ERR("Pointer to error type is NULL");
        return -1;
    }
    DecxGlobalMtx_Lock(Decx_GMtxId_LastHandle);

    *p_error_type = decx::g_last_handle.error_type;

    DecxGlobalMtx_Unlock(Decx_GMtxId_LastHandle);
    return 0;
}


_DECX_API_ int32_t DecxResetLastHandle()
{
    DecxGlobalMtx_Lock(Decx_GMtxId_LastHandle);

    decx::g_last_handle.error_type = DecxErrorTypes_e::DECX_SUCCESS;
    decx::utils::decx_strcpy<100>(decx::g_last_handle.error_string, SUCCESS);

    DecxGlobalMtx_Unlock(Decx_GMtxId_LastHandle);
    return 0;
}