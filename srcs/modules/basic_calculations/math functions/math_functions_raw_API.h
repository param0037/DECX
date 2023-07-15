/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _MATH_FUNCTIONS_RAW_APIS_H_
#define _MATH_FUNCTIONS_RAW_APIS_H_


#include "math_functions_exec.h"
#include "../operators_frame_exec.h"
#include "../../classes/type_info.h"


namespace decx
{
    namespace cpu {
        template <bool _print>
        void Log10_Raw_API(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);

        template <bool _print>
        void Log2_Raw_API(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);

        template <bool _print>
        void Exp_Raw_API(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);

        template <bool _print>
        void Sin_Raw_API(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);

        template <bool _print>
        void Cos_Raw_API(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);

        template <bool _print>
        void Tan_Raw_API(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);

        template <bool _print>
        void Asin_Raw_API(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);

        template <bool _print>
        void Acos_Raw_API(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);

        template <bool _print>
        void Atan_Raw_API(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);

        template <bool _print>
        void Sqrt_Raw_API(const float* src, float* dst, const uint64_t len, const uint32_t type, de::DH* handle);
    }
}


#endif