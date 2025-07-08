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


#ifndef _DECX_HANDLE_IMPL_H_
#define _DECX_HANDLE_IMPL_H_

#include <Handle/decx_handle.h>


#ifdef __cplusplus
namespace de
{
typedef struct DECX_Handle
{
    // indicates the type index of error
    DecxErrorTypes_e error_type;

    // describes the error statements
    char error_string[100];


    DECX_Handle() 
    {
        decx::utils::decx_strcpy<100>(this->error_string, SUCCESS);
        this->error_type = DecxErrorTypes_e::DECX_SUCCESS;
    }


    DECX_Handle(const char* _string, const DecxErrorTypes_e _err_code) 
    {
        decx::utils::decx_strcpy<100>(this->error_string, _string);
        this->error_type = _err_code;
    }
}DH;
}


namespace decx
{
#ifdef _DECX_CORE_CPU_
    extern de::DH _last_error;
#endif
}

namespace de {
    _DECX_API_ de::DH* GetLastError();


    _DECX_API_ void ResetLastError();
}
#endif      // #ifdef __cplusplus


// #if _C_EXPORT_ENABLED_
// #ifdef __cplusplus
// extern "C"
// {
// #endif
//     typedef struct DECX_Handle_t 
//     {
//         // indicates the type index of error
//         int error_type;

//         // describes the error statements
//         char error_string[100];
//     }DECX_Handle;

// #ifdef __cplusplus
// #define _CAST_HANDLE_(dst_handle_type, src) *((dst_handle_type*)&(src))
// }
// #endif
// #endif


#endif