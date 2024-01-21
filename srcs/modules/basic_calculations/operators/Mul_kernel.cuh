/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

/* /////////////////////////////////////////////////////////////////////////////
* for the small matrix or tensor, the calculation assignment will be sent to CPU
* vectorize the memory access, 通常，对于四字节的元素我用 vec4 类型访问全局内存；
* 对于二字节的元素我用 vec8 类型访问全局内存，对于 x8 的列数，都可以整除，因此，核函数不用考虑边界
*/

#ifndef _MULTIPLY_KERNEL_CUH_
#define _MULTIPLY_KERNEL_CUH_

#include "../../core/basic.h"
#include "../../core/configs/config.h"
#include "../../core/utils/decx_utils_functions.h"
#include "../../classes/classes_util.h"

#include "../../core/cudaStream_management/cudaEvent_queue.h"
#include "../../core/cudaStream_management/cudaStream_queue.h"


// ------------------------- M ------------------------------------------

namespace decx
{
    namespace calc {
        namespace GPUK {

            __global__
            /**
            * int* x2, add together
            * @param len : have considered vec4
            */
            void mul_m_ivec4(float4* A, float4* B, float4* dst, const size_t len);


            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void mul_m_ivec4_2D(float4* A, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds);


            __global__
            /**
            * int* x2, add together
            * @param len : have considered vec4
            */
            void mul_m_fvec4(float4* A, float4* B, float4* dst, const size_t len);



            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void mul_m_fvec4_2D(float4* A, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds);



            __global__
            void mul_m_hvec8(float4* A, float4* B, float4* dst, const size_t len);



            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void mul_m_hvec8_2D(float4* A, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds);



            __global__
            /**
            * int* x2, add together
            * @param len : have considered vec4
            */
            void mul_m_dvec2(float4* A, float4* B, float4* dst, const size_t len);



            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void mul_m_dvec2_2D(float4* A, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds);



            // ----------------------------- C --------------------------------------


            __global__
            void mul_c_ivec4(float4* src, int __x, float4* dst, const size_t len);


            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void mul_c_ivec4_2D(float4* src, int __x, float4* dst, const size_t eq_pitch, const uint2 bounds);


            __global__
            void mul_c_fvec4(float4* src, float __x, float4* dst, const size_t len);



            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void mul_c_fvec4_2D(float4* src, float __x, float4* dst, const size_t eq_pitch, const uint2 bounds);


            __global__
            void mul_c_hvec8(float4* src, half2 __x, float4* dst, const size_t len);



            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void mul_c_hvec8_2D(float4* src, half2 __x, float4* dst, const size_t eq_pitch, const uint2 bounds);



            __global__
            void mul_c_dvec2(float4* src, double __x, float4* dst, const size_t len);



            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void mul_c_dvec2_2D(float4* src, double __x, float4* dst, const size_t eq_pitch, const uint2 bounds);

        }
    }
}

// ---------------------------------------------------------------------------------------
// necessary single proccess from host to device and back to host
// use zero stream (defualt stream avoiding the time cost on cudaDeviceSynchronize()
/** HOST float
* @param len : height * pitch
*/


namespace decx
{
    namespace calc {
        static void dev_Kmul_m(float* DA, float* DB, float* Ddst, const size_t len, decx::cuda_stream* S) {
            //device_M(mul_m_fvec4, Vec4);
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::mul_m_fvec4 << <decx::utils::ceil<size_t>(len / 4, decx::cuda::_get_cuda_prop().maxThreadsPerBlock), decx::cuda::_get_cuda_prop().maxThreadsPerBlock,
                0, S->get_raw_stream_ref() >> > (
                _DAptr, _DBptr, _Ddstptr, len / 4);
        }

        static void dev_Kmul_m_2D(float* DA, float* DB, float* Ddst, const size_t eq_pitch, const uint2 bounds, decx::cuda_stream* S) {
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 4, 16));

            decx::calc::GPUK::mul_m_fvec4_2D << <grid, block,
                0, S->get_raw_stream_ref() >> > (_DAptr, _DBptr, _Ddstptr, eq_pitch / 4, bounds);
        }

        static void dev_Kmul_m(int* DA, int* DB, int* Ddst, const size_t len, decx::cuda_stream* S) {
            //device_M(mul_m_ivec4, Vec4);
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::mul_m_ivec4 << <decx::utils::ceil<size_t>(len / 4, decx::cuda::_get_cuda_prop().maxThreadsPerBlock), decx::cuda::_get_cuda_prop().maxThreadsPerBlock,
                0, S->get_raw_stream_ref() >> > (
                _DAptr, _DBptr, _Ddstptr, len / 4);
        }

        static void dev_Kmul_m_2D(int* DA, int* DB, int* Ddst, const size_t eq_pitch, const uint2 bounds, decx::cuda_stream* S) {
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 4, 16));

            decx::calc::GPUK::mul_m_ivec4_2D << <grid, block,
                0, S->get_raw_stream_ref() >> > (_DAptr, _DBptr, _Ddstptr, eq_pitch / 4, bounds);
        }

        static void dev_Kmul_m(de::Half* DA, de::Half* DB, de::Half* Ddst, const size_t len, decx::cuda_stream* S) {
            //device_M(mul_m_hvec8, Vec8);
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::mul_m_hvec8 << <decx::utils::ceil<size_t>(len / 8, decx::cuda::_get_cuda_prop().maxThreadsPerBlock), decx::cuda::_get_cuda_prop().maxThreadsPerBlock,
                0, S->get_raw_stream_ref() >> > (
                _DAptr, _DBptr, _Ddstptr, len / 8);
        }

        static void dev_Kmul_m_2D(de::Half* DA, de::Half* DB, de::Half* Ddst, const size_t eq_pitch, const uint2 bounds, decx::cuda_stream* S) {
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 8, 16));

            decx::calc::GPUK::mul_m_hvec8_2D << <grid, block,
                0, S->get_raw_stream_ref() >> > (_DAptr, _DBptr, _Ddstptr, eq_pitch / 8, bounds);
        }

        static void dev_Kmul_m(double* DA, double* DB, double* Ddst, const size_t len, decx::cuda_stream* S) {
            //device_M(mul_m_dvec2, Vec2);
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::mul_m_dvec2 << <decx::utils::ceil<size_t>(len / 2, decx::cuda::_get_cuda_prop().maxThreadsPerBlock), decx::cuda::_get_cuda_prop().maxThreadsPerBlock,
                0, S->get_raw_stream_ref() >> > (
                _DAptr, _DBptr, _Ddstptr, len / 2);
        }

        static void dev_Kmul_m_2D(double* DA, double* DB, double* Ddst, const size_t eq_pitch, const uint2 bounds, decx::cuda_stream* S) {
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 2, 16));

            decx::calc::GPUK::mul_m_dvec2_2D << <grid, block,
                0, S->get_raw_stream_ref() >> > (_DAptr, _DBptr, _Ddstptr, eq_pitch / 2, bounds);
        }


        // ----------------------------------------------- C -------------------------------------------------------------


        static void dev_Kmul_c(float* Dsrc, float __x, float* Ddst, const size_t len, decx::cuda_stream* S) {
            //device_C(mul_c_fvec4, Vec4, __x);
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::mul_c_fvec4 << <decx::utils::ceil<size_t>(len / 4, decx::cuda::_get_cuda_prop().maxThreadsPerBlock), decx::cuda::_get_cuda_prop().maxThreadsPerBlock,
                0, S->get_raw_stream_ref() >> > (
                _Dsrcptr, __x, _Ddstptr, len / 4);
        }

        static void dev_Kmul_c_2D(float* Dsrc, float __x, float* Ddst, const size_t eq_pitch, const uint2 bounds, decx::cuda_stream* S) {
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 4, 16));

            decx::calc::GPUK::mul_c_fvec4_2D << <grid, block,
                0, S->get_raw_stream_ref() >> > (_Dsrcptr, __x, _Ddstptr, eq_pitch / 4, bounds);
        }

        static void dev_Kmul_c(int* Dsrc, int __x, int* Ddst, const size_t len, decx::cuda_stream* S) {
            //device_C(mul_c_ivec4, Vec4, __x);
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::mul_c_ivec4 << <decx::utils::ceil<size_t>(len / 4, decx::cuda::_get_cuda_prop().maxThreadsPerBlock), decx::cuda::_get_cuda_prop().maxThreadsPerBlock,
                0, S->get_raw_stream_ref() >> > (
                _Dsrcptr, __x, _Ddstptr, len / 4);
        }

        static void dev_Kmul_c_2D(int* Dsrc, int __x, int* Ddst, const size_t eq_pitch, const uint2 bounds, decx::cuda_stream* S) {
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 4, 16));

            decx::calc::GPUK::mul_c_ivec4_2D << <grid, block,
                0, S->get_raw_stream_ref() >> > (_Dsrcptr, __x, _Ddstptr, eq_pitch / 4, bounds);
        }

        static void dev_Kmul_c(de::Half* Dsrc, de::Half __x, de::Half* Ddst, const size_t len, decx::cuda_stream* S) {
            half2 _x;
            _x.x = *((__half*)&__x);                _x.y = *((__half*)&__x);
            //device_C(mul_c_hvec8, Vec8, _x);
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::mul_c_hvec8 << <decx::utils::ceil<size_t>(len / 8, decx::cuda::_get_cuda_prop().maxThreadsPerBlock), decx::cuda::_get_cuda_prop().maxThreadsPerBlock,
                0, S->get_raw_stream_ref() >> > (
                _Dsrcptr, _x, _Ddstptr, len / 8);
        }

        static void dev_Kmul_c_2D(de::Half* Dsrc, de::Half __x, de::Half* Ddst, const size_t eq_pitch, const uint2 bounds, decx::cuda_stream* S) {
            half2 _x;
            _x.x = *((__half*)&__x);                _x.y = *((__half*)&__x);
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 8, 16));

            decx::calc::GPUK::mul_c_hvec8_2D << <grid, block >> > (_Dsrcptr, _x, _Ddstptr, eq_pitch / 8, bounds);
        }

        static void dev_Kmul_c(double* Dsrc, double __x, double* Ddst, const size_t len, decx::cuda_stream* S) {
            //device_C(mul_c_dvec2, Vec2, __x);
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::mul_c_dvec2 << <decx::utils::ceil<size_t>(len / 2, decx::cuda::_get_cuda_prop().maxThreadsPerBlock), decx::cuda::_get_cuda_prop().maxThreadsPerBlock,
                0, S->get_raw_stream_ref() >> > (
                _Dsrcptr, __x, _Ddstptr, len / 2);
        }

        static void dev_Kmul_c_2D(double* Dsrc, double __x, double* Ddst, const size_t eq_pitch, const uint2 bounds, decx::cuda_stream* S) {
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 2, 16));

            decx::calc::GPUK::mul_c_dvec2_2D << <grid, block,
                0, S->get_raw_stream_ref() >> > (_Dsrcptr, __x, _Ddstptr, eq_pitch / 2, bounds);
        }
    }
}


#endif