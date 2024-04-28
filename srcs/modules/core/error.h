/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved. 
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _ERROR_H_
#define _ERROR_H_


#include "../handles/decx_handles.h"
#include "utils/decx_utils_functions.h"

#ifdef Windows
#include <Windows.h>
#endif


// APIs from DECX_core_CPU
namespace decx
{
    /*_DECX_API_ bool Get_enable_log_print();


    _DECX_API_ bool Get_ignore_successful_print();


    _DECX_API_ bool Get_ignore_warnings();*/
}


#ifdef Windows
#define SetConsoleColor(_color_flag)    \
    SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), _color_flag)   \

#define ResetConsoleColor SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), 7)
#endif


#ifdef Linux 
#define SetConsoleColor(_color_flag)        \
    printf("\033[0;32;31m");                \

#define ResetConsoleColor printf("\033[m");
#endif

/**
* Color :
*   error   : 4 (red)
*   warning : 6 (bright yellow)
*/
#define Print_Error_Message(_color, _statement)     \
{                                                   \
    SetConsoleColor(_color);                        \
    printf(_statement);                             \
    ResetConsoleColor;                              \
}


#ifdef _DECX_CUDA_PARTS_
static inline const char* _cudaGetErrorEnum(cudaError_t error) noexcept
{
    return cudaGetErrorName(error);
}


template <typename T>
void check(T result, char const* const func, const char* const file, int const line) 
{
    if (result) {
#ifdef Windows
        SetConsoleColor(4);
#endif
        
#ifdef Linux
        printf("\033[0;32;31m");
#endif
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
        ResetConsoleColor;
        exit(EXIT_FAILURE);
    }
}


#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
#endif


#ifdef Windows
#define _CONSOLE_COLOR_RED_ 4
#define _CONSOLE_COLOR_YELLOW 6
#define _CONSOLE_COLOR_GREEN 7
#endif

#ifdef Linux
#define _CONSOLE_COLOR_RED_ 4
#define _CONSOLE_COLOR_YELLOW 6
#define _CONSOLE_COLOR_GREEN 7
#endif


namespace decx
{
    namespace err
    {
        template <bool _sync = true>
        static void Success(de::DH* handle)
        {
            if (_sync) {
                /*if ((!decx::Get_ignore_successful_print()) && decx::Get_enable_log_print()) {
                    Print_Error_Message(0x0E, SUCCESS);
                }*/
            }
            decx::utils::decx_strcpy<100>(handle->error_string, SUCCESS);
            handle->error_type = decx::DECX_error_types::DECX_SUCCESS;
        }

        template <bool _print_to_console = false, int _console_color = 4>
        static void handle_error_info_modify(de::DH* handle, decx::DECX_error_types _error_type, const char* _err_statement) 
        {
            handle->error_type = _error_type;
            decx::utils::decx_strcpy<100>(handle->error_string, _err_statement);
            if (_print_to_console) {
                Print_Error_Message(_console_color, _err_statement);
            }
        }
    }
}


namespace decx
{
    namespace warn
    {
        static void CPU_Hyper_Threading(de::DH* handle)
        {
            /*if (!decx::Get_ignore_warnings()) {
                Print_Error_Message(6, CPU_HYPER_THREADING);
            }*/
            decx::utils::decx_strcpy<100>(handle->error_string, CPU_HYPER_THREADING);
            handle->error_type = decx::DECX_error_types::DECX_SUCCESS;
        }


        template <bool _sync = true>
        static void Memcpy_different_types(de::DH* handle)
        {
            if (_sync) {
                /*if ((!decx::Get_ignore_warnings()) && decx::Get_enable_log_print()) {
                    Print_Error_Message(0x0E, MEMCPY_DIFFERENT_DATA_TYPE);
                }*/
            }
            decx::utils::decx_strcpy<100>(handle->error_string, MEMCPY_DIFFERENT_DATA_TYPE);
            handle->error_type = decx::DECX_error_types::MEMCPY_DIFFERENT_TYPES;
        }

    }
}


#endif
