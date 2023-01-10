/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved. 
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _ERROR_H_
#define _ERROR_H_


#include "../handles/decx_handles.h"
#include "utils/decx_utils_functions.h"


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



#if defined _DECX_CUDA_CODES_ || defined _DECX_ALLOC_CODES_ || defined(_DECX_CLASSES_CODES_)
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

//#ifdef Windows
#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
#endif



#define Error_Handle(__handle, _operator, _return)    \
{    \
if (__handle _operator error_type != decx::DECX_SUCCESS) {    \
    return _return;    \
}    \
}


// --------------------------- ERROR_STATEMENTS -------------------------------

#define CUDA_NOT_INIT                               "CUDA should be initialized first\n"
#define CPU_NOT_INIT                                "DECX_cpu should be initialized first\n"
#define SUCCESS                                     "No error\n"
#define FFT_ERROR_LENGTH                            "Each dim should be able to be separated by 2, 3 and 5\n"
#define FFT_ERROR_WIDTH                             "Width should be able to be separated by 2, 3 and 5\n"
#define FFT_ERROR_HEIGHT                            "Height should be able to be separated by 2, 3 and 5\n"
#define ALLOC_FAIL                                  "Fail to allocate memory on RAM\n"
#define DEV_ALLOC_FAIL                              "Fail to allocate memory on device\n"
#define MAT_DIM_NOT_MATCH                           "Dim(s) is(are) not equal to each other\n"
#define MEANINGLESS_FLAG                            "This flag is meaningless in current context\n"
#define CUDA_STREAM_ACCESS_FAIL                     "Fail to access cuda stream\n"
#define CALSS_NOT_INIT                              "The class is not initialized\n"
#define CHANNEL_ERROR                               "The number of channel is incorrect\n"
#define TYPE_ERROR_NOT_MATCH                        "The types of the two objects are not equal"
#define INVALID_PARAM                               "The parameter(s) is(are) invalid\n"
#define IMAGE_LOAD_FAIL                             "Failed to load the image from file\n"


//#ifdef Windows
#define Print_Error_Message(_color, _statement)     \
{                                                   \
    SetConsoleColor(_color);                        \
    printf(_statement);                             \
    ResetConsoleColor;                              \
}
//#endif


// #ifdef Linux
// #define Print_Error_Message(_color, _statement)     \
// {                                                   \
//     printf("\033[0;32;31m");                        \
//     printf(_statement);                             \
//     printf("\033[m");                               \
// }
// #endif



namespace decx
{
#ifdef _DECX_CUDA_CODES_
    static void Not_init(de::DH* handle)
    {
        //handle->error_string = (char*)NOT_INIT;
        handle->error_type = decx::DECX_FAIL_not_init;
    }
#endif


    static void Success(de::DH* handle)
    {
        //handle->error_string = (char*)SUCCESS;
        handle->error_type = decx::DECX_SUCCESS;
    }



    static void MDim_Not_Matching(de::DH* handle)
    {
        //handle->error_string = (char*)"Matrix Dims don't match each other";
        handle->error_type = decx::DECX_FAIL_DimsNotMatching;
    }


    static void Matrix_number_not_matching(de::DH* handle)
    {
        //handle->error_string = (char*)"The number of matrices don't match each other";
        handle->error_type = decx::DECX_FAIL_MNumNotMatching;
    }



    static void GEMM_DimNMatch(de::DH* handle)
    {
        //handle->error_type = decx::DECX_FAIL_DimsNotMatching;
        //handle->error_string = (char*)"The width of matrix A and the height of matrix B are required to be same";
    }



    static void TDim_Not_Matching(de::DH* handle)
    {
        //handle->error_string = (char*)"Tensor Dims don't match each other";
        handle->error_type = decx::DECX_FAIL_DimsNotMatching;
    }


    static void StoreFormatError(de::DH* handle)
    {
        //handle->error_string = (char*)"The store type is not suitable";
        handle->error_type = decx::DECX_FAIL_StoreError;
    }



    static void MeaninglessFlag(de::DH* handle)
    {
        //handle->error_string = (char*)"This flag is meaningless";
        handle->error_type = decx::DECX_FAIL_ErrorFlag;
    }


    namespace err
    {
        static void CUDA_Not_init(de::DH* handle)
        {
            decx::utils::decx_strcpy<100>(handle->error_string, CUDA_NOT_INIT);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_not_init;
        }

        static void CPU_Not_init(de::DH* handle)
        {
            decx::utils::decx_strcpy<100>(handle->error_string, CPU_NOT_INIT);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_not_init;
        }


        static void FFT_Error_length(de::DH* handle)
        {
            decx::utils::decx_strcpy<100>(handle->error_string, FFT_ERROR_LENGTH);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_FFT_error_length;
        }


        static void AllocateFailure(de::DH* handle)
        {
            decx::utils::decx_strcpy<100>(handle->error_string, ALLOC_FAIL);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_ALLOCATION;
        }

        static void InvalidParam(de::DH* handle)
        {
            decx::utils::decx_strcpy<100>(handle->error_string, INVALID_PARAM);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_INVALID_PARAM;
        }

        static void device_AllocateFailure(de::DH* handle)
        {
            decx::utils::decx_strcpy<100>(handle->error_string, DEV_ALLOC_FAIL);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_ALLOCATION;
        }


        static void Mat_Dim_Not_Matching(de::DH* handle)
        {
            decx::utils::decx_strcpy<100>(handle->error_string, MAT_DIM_NOT_MATCH);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_DimsNotMatching;
        }


        static void CUDA_Stream_access_fail(de::DH* handle)
        {
            decx::utils::decx_strcpy<100>(handle->error_string, CUDA_STREAM_ACCESS_FAIL);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_CUDA_STREAM;
        }


        static void Success(de::DH* handle)
        {
            decx::utils::decx_strcpy<100>(handle->error_string, SUCCESS);
            handle->error_type = decx::DECX_error_types::DECX_SUCCESS;
        }

        static void ClassNotInit(de::DH* handle)
        {
            decx::utils::decx_strcpy<100>(handle->error_string, CALSS_NOT_INIT);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_CLASS_NOT_INIT;
        }

        static void Channel_Error(de::DH* handle)
        {
            decx::utils::decx_strcpy<100>(handle->error_string, CHANNEL_ERROR);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_CHANNEL_ERROR;
        }

        static void TypeError_NotMatch(de::DH* handle)
        {
            decx::utils::decx_strcpy<100>(handle->error_string, TYPE_ERROR_NOT_MATCH);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_TYPE_MOT_MATCH;
        }


        static void ImageLoadFailed(de::DH* handle)
        {
            decx::utils::decx_strcpy<100>(handle->error_string, IMAGE_LOAD_FAIL);
            handle->error_type = decx::DECX_error_types::DECX_FAIL_IMAGE_LOAD_FAILED;
        }
    }
}


#endif