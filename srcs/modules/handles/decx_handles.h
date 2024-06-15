/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _DECX_HANDLES_H_
#define _DECX_HANDLES_H_

#include "../core/utils/decx_utils_functions.h"


#define Check_Runtime_Error(handle_ptr) {                                   \
if ((handle_ptr)->error_type != decx::DECX_error_types::DECX_SUCCESS) {     \
    return;                                                                 \
}                                                                           \
}                                                                           \

#ifdef __cplusplus
namespace decx
{
    enum DECX_error_types
    {
        /*
        No error occurs. The error code is 0x00. However please notice that 
        warning might occur even process returns a handle with error type of DECX_SUCCESS.
        */
        DECX_SUCCESS                    = 0x00,

        /*
        * User does not initialize DECX information before calling any of the APIs. 
        Initialization includes CPU information and CUDA runtime information. Please call 
        de::InitCuda() and de::InitCPUInfo() before using any function in DECX.
        */
        DECX_FAIL_CPU_not_init          = 0x01,
        DECX_FAIL_CUDA_not_init         = 0x16,


        /*
        * The length of vector or the dimensions of matrix in fast Fourier transform should 
        be able to divided into one of the factors including 2,3, 5 and 7. Otherwise, process 
        returns a handle with error type of DECX_FAIL_FFT_error_length.
        */
        DECX_FAIL_FFT_error_length      = 0x02,

        /*
        * The dimensions of the input object are not matched the requirements.
        */
        DECX_FAIL_DimsNotMatching       = 0x03,

        /*
        * Comparison between two complex numbers is illegal.
        */
        DECX_FAIL_Complex_comparing     = 0x04,

        /*
        * The dimension of the convolutional kernel is illegal. E.g., The width and height 
        of the kernel in 2D convolution should be odd numbers.
        */
        DECX_FAIL_ConvBadKernel         = 0x05,

        /*
        * In Non-local Means filter (NLM), the dimensions of searching windows and template 
        windows should meet the requirements. Please refer to chapter introduction on NLM 
        for more information.
        */
        DECX_FAIL_CVNLM_BadArea         = 0x06,

        /*
        * The flags need to be indicated in, e.g., matrix multiplication on half-precision 
        floating point matrices indicating the calculation accuracy, and in extension 
        indicating the extending type. Process return error in type of DECX_FAIL_ErrorFlag 
        if user indicates the illegal flag.
        */
        DECX_FAIL_ErrorFlag             = 0x07,

        /*
        * Process return error in type of DECX_FAIL_ALLOCATION as long as the failure occurs 
        on any of the memory allocation, which means that all types of memory allocations are 
        included.
        */
        DECX_FAIL_ALLOCATION            = 0x8,
        DECX_FAIL_CUDA_ALLOCATION       = 0x17,

        /*
        * Process return error in type of DECX_FAIL_CUDA_STREAM as long as the failure occurs 
        on accessing CUDA stream. 
        */
        DECX_FAIL_CUDA_STREAM           = 0x9,

        /*
        * Process return error in type of DECX_FAIL_CLASS_NOT_INIT if user input an uninitialized 
        object if initialization is needed. An uninitialized object is the object constructed 
        via default constructor.
        */
        DECX_FAIL_CLASS_NOT_INIT        = 0x0a,

        /*
        * Process return error in type of DECX_FAIL_TYPE_MOT_MATCH if the data type of input object 
        is not matched the requirement.
        */
        DECX_FAIL_TYPE_MOT_MATCH        = 0x0b,

        /*
        * Process return error in type of DECX_FAIL_INVALID_PARAM if the input parameter is not 
        matched the requirement.
        */
        DECX_FAIL_INVALID_PARAM         = 0x0c,

        /*
        * This error type is only considered when loading an image from disk. Process returns error 
        if failure occurs on file loading, e.g., file does not exist.
        */
        DECX_FAIL_IMAGE_LOAD_FAILED     = 0x0d,

        /*
        * Process return error in type of DECX_FAIL_CUDA_EVENT as long as the failure occurs on 
        accessing CUDA event.
        */
        DECX_FAIL_CUDA_EVENT            = 0x0e,

        /*
        * Please note that this flag is not the error but warning. When deallocate a memory that is 
        referred multiple times, process returns error in type of MULTIPLE_REFERENCES. Please make 
        sure that the other objects that refer to the same memory block are useless, or the valuable 
        data will be lost.
        */
        MULTIPLE_REFERENCES             = 0x0f,

        /*
        * When the number of the concurrent threads exceed that of hardware concurrency, process returns 
        error in type of CPU_HYPER_THREADING. Please avoid hyper-threading the CPU in case of the performance 
        loss.
        */
        CPU_HYPER_THREADING             = 0x10,

        

        MEMCPY_DIFFERENT_TYPES          = 0x11,



        DECX_FAIL_MEMCPY_OVERRANGED     = 0x12,



        DECX_FAIL_UNSUPPORTED_TYPE      = 0x13,


        DECX_FAIL_HOST_MEM_REGISTERED   = 0x14,


        DECX_FAIL_HOST_MEM_UNREGISTERED = 0x15
    };
}
#endif


// --------------------------- ERROR_STATEMENTS -------------------------------

#define CUDA_NOT_INIT                               "error: CUDA should be initialized first\n"
#define CPU_NOT_INIT                                "error: DECX_cpu should be initialized first\n"
#define SUCCESS                                     "success: No error\n"
#define FFT_ERROR_LENGTH                            "error: Each dim should be able to be separated by 2, 3 and 5\n"
#define ALLOC_FAIL                                  "error: Fail to allocate memory on RAM\n"
#define DEV_ALLOC_FAIL                              "error: Fail to allocate memory on device\n"
#define CU_FILTER2D_KERNEL_OVERRANGED               "error: The kernel width is too large\n"
#define MAT_DIM_NOT_MATCH                           "error: Dim(s) is(are) not equal to each other\n"
#define MEANINGLESS_FLAG                            "error: This flag is meaningless in current context\n"
#define CUDA_STREAM_ACCESS_FAIL                     "error: Fail to access cuda stream\n"
#define CUDA_EVENT_ACCESS_FAIL                      "error: Fail to access cuda event\n"
#define CLASS_NOT_INIT                              "error: The class is not initialized\n"
#define CHANNEL_ERROR                               "error: The number of channel is incorrect\n"
#define TYPE_ERROR_NOT_MATCH                        "error: The types of the two objects are not equal"
#define INVALID_PARAM                               "error: The parameter(s) is(are) invalid\n"
#define IMAGE_LOAD_FAIL                             "error: Failed to load the image from file\n"
#define INVALID_PTR                                 "error: Failed to access the pointer, since it is invalid\n"
#define MEMCPY_OVERRANGED                           "error: Failed to transfer data since it is out of range\n"
#define UNSUPPORTED_TYPE                            "error: The data type of the input onject is not supported\n"
#define INTERNAL_ERROR                              "error: Internal error\n"
#define HOST_MEM_REGISTERED                         "error: The host memory has already been pinned\n"
#define HOST_MEM_UNREGISTERED                       "error: The host memory has not been pinned yet\n"


// --------------------------- WARNING_STATEMENTS -------------------------------

#define MULTIPLE_REFERENCES                         "warning: This space is refered by multiple simbols\n"
#define CPU_HYPER_THREADING                         "warning: Threading number exceeds the hardware concurrency, which may reduce performance\n"
#define MEMCPY_DIFFERENT_DATA_TYPE                  "warning: The data types of the two objects are different, which reinterprets the data\n"


// Lable a function who pass a pointer of handle as "The runtime status handle might change through this function"
#define _CHECK_RUNTIME_STATUS_REQUIRED_

#ifdef _CHECK_RUNTIME_STATUS_REQUIRED_
#define _CRSR_ _CHECK_RUNTIME_STATUS_REQUIRED_
#else
#define _CRSR_
#endif


#ifdef __cplusplus
namespace de
{
    typedef struct DECX_Handle
    {
        // indicates the type index of error
        int error_type;

        // describes the error statements
        char error_string[100];


        DECX_Handle() 
        {
            decx::utils::decx_strcpy<100>(this->error_string, SUCCESS);
            this->error_type = decx::DECX_error_types::DECX_SUCCESS;
        }


        DECX_Handle(const char* _string, const int _err_code) 
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


#if _C_EXPORT_ENABLED_
#ifdef __cplusplus
extern "C"
{
#endif
    typedef struct DECX_Handle_t 
    {
        // indicates the type index of error
        int error_type;

        // describes the error statements
        char error_string[100];
    }DECX_Handle;

#ifdef __cplusplus
#define _CAST_HANDLE_(dst_handle_type, src) *((dst_handle_type*)&(src))
}
#endif
#endif


#endif
