/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _DECX_HANDLES_H_
#define _DECX_HANDLES_H_


namespace decx
{
    enum DECX_error_types
    {
        DECX_SUCCESS                    = 0x00,

        DECX_FAIL_not_init              = 0x01,

        DECX_FAIL_FFT_error_length      = 0x02,

        DECX_FAIL_DimsNotMatching       = 0x03,

        DECX_FAIL_Complex_comparing     = 0x04,

        DECX_FAIL_ConvBadKernel         = 0x05,
        DECX_FAIL_StepError             = 0x06,

        DECX_FAIL_ChannelError          = 0x07,
        DECX_FAIL_CVNLM_BadArea         = 0x08,

        DECX_FAIL_FileNotExist          = 0x09,

        DECX_GEMM_DimsError             = 0x0a,

        DECX_FAIL_ErrorFlag             = 0x0b,

        DECX_FAIL_DimError              = 0x0c,

        DECX_FAIL_ErrorParams           = 0x0d,

        DECX_FAIL_StoreError            = 0x0e,

        DECX_FAIL_MNumNotMatching       = 0x0f,

        DECX_FAIL_ALLOCATION            = 0x10,

        DECX_FAIL_CUDA_STREAM           = 0x11,

        DECX_FAIL_CLASS_NOT_INIT        = 0x12,

        DECX_FAIL_CHANNEL_ERROR         = 0x13,

        DECX_FAIL_TYPE_MOT_MATCH        = 0x14,

        DECX_FAIL_INVALID_PARAM         = 0x15,

        DECX_FAIL_IMAGE_LOAD_FAILED     = 0x16
    };
}


namespace de
{
    typedef struct DECX_Handle
    {
        // indicates the type index of error
        int error_type;

        // describes the error statements
        char error_string[100];
    }DH;
}


#endif