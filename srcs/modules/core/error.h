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
    _DECX_API_ bool Get_enable_log_print();


    _DECX_API_ bool Get_ignore_successful_print();


    _DECX_API_ bool Get_ignore_warnings();
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



namespace decx
{
    namespace err
    {
        template <bool _print = true>
        static void CUDA_Not_init(de::DH* handle)
        {
            if (_print) {
                if (decx::Get_enable_log_print()) {
                    Print_Error_Message(4, CUDA_NOT_INIT);
                }
            }
            decx::utils::decx_strcpy<100>(handle->error_string, CUDA_NOT_INIT);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_not_init;
        }


        template <bool _print = true>
        static void CPU_Not_init(de::DH* handle)
        {
            if (_print) {
                if (decx::Get_enable_log_print()) {
                    Print_Error_Message(4, CPU_NOT_INIT);
                }
            }
            decx::utils::decx_strcpy<100>(handle->error_string, CPU_NOT_INIT);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_not_init;
        }

        template <bool _print = true>
        static void FFT_Error_length(de::DH* handle)
        {
            if (_print) {
                if (decx::Get_enable_log_print()) {
                    Print_Error_Message(4, FFT_ERROR_LENGTH);
                }
            }
            decx::utils::decx_strcpy<100>(handle->error_string, FFT_ERROR_LENGTH);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_FFT_error_length;
        }

        template <bool _print = true>
        static void AllocateFailure(de::DH* handle)
        {
            if (_print) {
                if (decx::Get_enable_log_print()) {
                    Print_Error_Message(4, ALLOC_FAIL);
                }
            }
            decx::utils::decx_strcpy<100>(handle->error_string, ALLOC_FAIL);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_ALLOCATION;
        }

        template <bool _print = true>
        static void InvalidParam(de::DH* handle)
        {
            if (_print) {
                if (decx::Get_enable_log_print()) {
                    Print_Error_Message(4, INVALID_PARAM);
                }
            }
            decx::utils::decx_strcpy<100>(handle->error_string, INVALID_PARAM);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_INVALID_PARAM;
        }


        template <bool _print = true>
        static void MeaningLessFlag(de::DH* handle)
        {
            if (_print) {
                if (decx::Get_enable_log_print()) {
                    Print_Error_Message(4, MEANINGLESS_FLAG);
                }
            }
            decx::utils::decx_strcpy<100>(handle->error_string, MEANINGLESS_FLAG);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_INVALID_PARAM;
        }


        template <bool _print = true>
        static void device_AllocateFailure(de::DH* handle)
        {
            if (_print) {
                if (decx::Get_enable_log_print()) {
                    Print_Error_Message(4, DEV_ALLOC_FAIL);
                }
            }
            decx::utils::decx_strcpy<100>(handle->error_string, DEV_ALLOC_FAIL);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_ALLOCATION;
        }

        template <bool _print = true>
        static void Mat_Dim_Not_Matching(de::DH* handle)
        {
            if (_print) {
                if (decx::Get_enable_log_print()) {
                    Print_Error_Message(4, MAT_DIM_NOT_MATCH);
                }
            }
            decx::utils::decx_strcpy<100>(handle->error_string, MAT_DIM_NOT_MATCH);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_DimsNotMatching;
        }

        template <bool _print = true>
        static void CUDA_Stream_access_fail(de::DH* handle)
        {
            if (_print) {
                if (decx::Get_enable_log_print()) {
                    Print_Error_Message(4, CUDA_STREAM_ACCESS_FAIL);
                }
            }
            decx::utils::decx_strcpy<100>(handle->error_string, CUDA_STREAM_ACCESS_FAIL);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_CUDA_STREAM;
        }

        template <bool _print = true>
        static void CUDA_Event_access_fail(de::DH* handle)
        {
            if (_print) {
                if (decx::Get_enable_log_print()) {
                    Print_Error_Message(4, CUDA_EVENT_ACCESS_FAIL);
                }
            }
            decx::utils::decx_strcpy<100>(handle->error_string, CUDA_EVENT_ACCESS_FAIL);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_CUDA_EVENT;
        }

        template <bool _sync = true>
        static void Success(de::DH* handle)
        {
            if (_sync) {
                if ((!decx::Get_ignore_successful_print()) && decx::Get_enable_log_print()) {
                    Print_Error_Message(0x0E, SUCCESS);
                }
            }
            decx::utils::decx_strcpy<100>(handle->error_string, SUCCESS);
            handle->error_type = decx::DECX_error_types::DECX_SUCCESS;
        }

        template <bool _sync = true>
        static void ClassNotInit(de::DH* handle)
        {
            if (_sync) {
                if (decx::Get_enable_log_print()) {
                    Print_Error_Message(0x0E, CLASS_NOT_INIT);
                }
            }
            decx::utils::decx_strcpy<100>(handle->error_string, CLASS_NOT_INIT);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT;
        }


        template <bool _sync = true>
        static void TypeError_NotMatch(de::DH* handle)
        {
            if (_sync) {
                if ((!decx::Get_ignore_successful_print()) && decx::Get_enable_log_print()) {
                    Print_Error_Message(4, TYPE_ERROR_NOT_MATCH);
                }
            }
            decx::utils::decx_strcpy<100>(handle->error_string, TYPE_ERROR_NOT_MATCH);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_TYPE_MOT_MATCH;
        }


        template <bool _sync = true>
        static void ImageLoadFailed(de::DH* handle)
        {
            if (_sync) {
                if ((!decx::Get_ignore_successful_print()) && decx::Get_enable_log_print()) {
                    Print_Error_Message(4, IMAGE_LOAD_FAIL);
                }
            }
            decx::utils::decx_strcpy<100>(handle->error_string, IMAGE_LOAD_FAIL);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_IMAGE_LOAD_FAILED;
        }


        template <bool _sync = true>
        static void Memcpy_overranged(de::DH* handle)
        {
            if (_sync) {
                if ((!decx::Get_ignore_successful_print()) && decx::Get_enable_log_print()) {
                    Print_Error_Message(4, MEMCPY_OVERRANGED);
                }
            }
            decx::utils::decx_strcpy<100>(handle->error_string, MEMCPY_OVERRANGED);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_MEMCPY_OVERRANGED;
        }


        template <bool _sync = true>
        static void Unsupported_Type(de::DH* handle)
        {
            if (_sync) {
                if ((!decx::Get_ignore_successful_print()) && decx::Get_enable_log_print()) {
                    Print_Error_Message(4, UNSUPPORTED_TYPE);
                }
            }
            decx::utils::decx_strcpy<100>(handle->error_string, UNSUPPORTED_TYPE);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_UNSUPPORTED_TYPE;
        }
    }
}


namespace decx
{
    namespace warn
    {
        static void CPU_Hyper_Threading(de::DH* handle)
        {
            if (!decx::Get_ignore_warnings()) {
                Print_Error_Message(6, CPU_HYPER_THREADING);
            }
            decx::utils::decx_strcpy<100>(handle->error_string, CPU_HYPER_THREADING);
            handle->error_type = decx::DECX_error_types::DECX_SUCCESS;
        }


        template <bool _sync = true>
        static void Memcpy_different_types(de::DH* handle)
        {
            if (_sync) {
                if ((!decx::Get_ignore_warnings()) && decx::Get_enable_log_print()) {
                    Print_Error_Message(0x0E, MEMCPY_DIFFERENT_DATA_TYPE);
                }
            }
            decx::utils::decx_strcpy<100>(handle->error_string, MEMCPY_DIFFERENT_DATA_TYPE);
            handle->error_type = decx::DECX_error_types::MEMCPY_DIFFERENT_TYPES;
        }
    }
}


#endif