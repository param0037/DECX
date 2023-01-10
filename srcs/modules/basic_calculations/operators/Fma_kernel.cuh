/**
*    ---------------------------------------------------------------------
*    Author : Wayne Anderson
*    Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

/* /////////////////////////////////////////////////////////////////////////////
* for the small matrix or tensor, the calculation assignment will be sent to CPU
* vectorize the memory access, 通常，对于四字节的元素我用 vec4 类型访问全局内存；
* 对于二字节的元素我用 vec8 类型访问全局内存，对于 x8 的列数，都可以整除，因此，核函数不用考虑边界
*/

#ifndef _FMA_KERNEL_CUH_
#define _FMA_KERNEL_CUH_

#include "../../core/basic.h"
#include "../../core/thread_management/thread_pool.h"
#include "../../classes/classes_util.h"
#include "../../core/utils/fragment_arrangment.h"


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
            void fma_m_ivec4(float4* A, float4* B, float4* C, float4* dst, const size_t len);



            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void fma_m_ivec4_2D(float4* A, float4* B, float4* C, float4* dst, const size_t eq_pitch, const uint2 bounds);



            __global__
            /**
            * int* x2, add together
            * @param len : have considered vec4
            */
            void fma_m_fvec4(float4* A, float4* B, float4* C, float4* dst, const size_t len);


            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void fma_m_fvec4_2D(float4* A, float4* B, float4* C, float4* dst, const size_t eq_pitch, const uint2 bounds);


            __global__
            void fma_m_hvec8(float4* A, float4* B, float4* C, float4* dst, const size_t len);



            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void fma_m_hvec8_2D(float4* A, float4* B, float4* C, float4* dst, const size_t eq_pitch, const uint2 bounds);



            __global__
            /**
            * int* x2, add together
            * @param len : have considered vec4
            */
            void fma_m_dvec2(float4* A, float4* B, float4* C, float4* dst, const size_t len);



            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void fma_m_dvec2_2D(float4* A, float4* B, float4* C, float4* dst, const size_t eq_pitch, const uint2 bounds);



            // ----------------------------- C --------------------------------------


            __global__
            void fma_c_ivec4(float4* A, int __x, float4* B, float4* dst, const size_t len);


            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void fma_c_ivec4_2D(float4* A, int __x, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds);



            __global__
            void fma_c_fvec4(float4* A, float __x, float4* B, float4* dst, const size_t len);



            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void fma_c_fvec4_2D(float4* A, float __x, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds);



            __global__
            void fma_c_hvec8(float4* A, half2 __x, float4* B, float4* dst, const size_t len);


            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void fma_c_hvec8_2D(float4* A, half2 __x, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds);



            __global__
            void fma_c_dvec2(float4* A, double __x, float4* B, float4* dst, const size_t len);


            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void fma_c_dvec2_2D(float4* A, double __x, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds);
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
    namespace calc
    {
        static void dev_Kfma_m(float* DA, float* DB, float* DC, float* Ddst, const size_t len) {
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _DCptr = reinterpret_cast<float4*>(DC);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::fma_m_fvec4 << <decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
                _DAptr, _DBptr, _DCptr, _Ddstptr, len / 4);
        }

        static void dev_Kfma_m_2D(float* DA, float* DB, float* DC, float* Ddst, const size_t eq_pitch, const uint2 bounds) {
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _DCptr = reinterpret_cast<float4*>(DC);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 4, 16));

            decx::calc::GPUK::fma_m_fvec4_2D << <grid, block >> > (_DAptr, _DBptr, _DCptr, _Ddstptr, eq_pitch / 4, bounds);
        }

        static void dev_Kfma_m(int* DA, int* DB, int* DC, int* Ddst, const size_t len) {
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _DCptr = reinterpret_cast<float4*>(DC);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::fma_m_ivec4 << <decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
                _DAptr, _DBptr, _DCptr, _Ddstptr, len / 4);
        }

        static void dev_Kfma_m_2D(int* DA, int* DB, int* DC, int* Ddst, const size_t eq_pitch, const uint2 bounds) {
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _DCptr = reinterpret_cast<float4*>(DC);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 4, 16));

            decx::calc::GPUK::fma_m_ivec4_2D << <grid, block >> > (_DAptr, _DBptr, _DCptr, _Ddstptr, eq_pitch / 4, bounds);
        }

        static void dev_Kfma_m(de::Half* DA, de::Half* DB, de::Half* DC, de::Half* Ddst, const size_t len) {
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _DCptr = reinterpret_cast<float4*>(DC);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::fma_m_hvec8 << <decx::utils::ceil<size_t>(len / 8, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
                _DAptr, _DBptr, _DCptr, _Ddstptr, len / 8);
        }

        static void dev_Kfma_m_2D(de::Half* DA, de::Half* DB, de::Half* DC, de::Half* Ddst, const size_t eq_pitch, const uint2 bounds) {
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _DCptr = reinterpret_cast<float4*>(DC);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 8, 16));

            decx::calc::GPUK::fma_m_hvec8_2D << <grid, block >> > (_DAptr, _DBptr, _DCptr, _Ddstptr, eq_pitch / 8, bounds);
        }

        static void dev_Kfma_m(double* DA, double* DB, double* DC, double* Ddst, const size_t len) {
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _DCptr = reinterpret_cast<float4*>(DC);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::fma_m_dvec2 << <decx::utils::ceil<size_t>(len / 2, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
                _DAptr, _DBptr, _DCptr, _Ddstptr, len / 2);
        }

        static void dev_Kfma_m_2D(double* DA, double* DB, double* DC, double* Ddst, const size_t eq_pitch, const uint2 bounds) {
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _DCptr = reinterpret_cast<float4*>(DC);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 2, 16));

            decx::calc::GPUK::fma_m_dvec2_2D << <grid, block >> > (_DAptr, _DBptr, _DCptr, _Ddstptr, eq_pitch / 2, bounds);
        }

        // ----------------------------------------------- C -------------------------------------------------------------

        static void dev_Kfma_c(float* DA, float __x, float* DB, float* Ddst, const size_t len) {
            float4* Dptr_A = reinterpret_cast<float4*>(DA);
            float4* Dptr_B = reinterpret_cast<float4*>(DB);
            float4* Dptr_dst = reinterpret_cast<float4*>(Ddst);
            decx::calc::GPUK::fma_c_fvec4 << < decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
                Dptr_A, __x, Dptr_B, Dptr_dst, len / 4);
        }

        static void dev_Kfma_c_2D(float* DA, float __x, float* DB, float* Ddst, const size_t eq_pitch, const uint2 bounds) {
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 4, 16));

            decx::calc::GPUK::fma_c_fvec4_2D << <grid, block >> > (_DAptr, __x, _DBptr, _Ddstptr, eq_pitch / 4, bounds);
        }


        static void dev_Kfma_c(int* DA, int __x, int* DB, int* Ddst, const size_t len) {
            float4* Dptr_A = reinterpret_cast<float4*>(DA);
            float4* Dptr_B = reinterpret_cast<float4*>(DB);
            float4* Dptr_dst = reinterpret_cast<float4*>(Ddst);
            decx::calc::GPUK::fma_c_ivec4 << < decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
                Dptr_A, __x, Dptr_B, Dptr_dst, len / 4);
        }

        static void dev_Kfma_c_2D(int* DA, int __x, int* DB, int* Ddst, const size_t eq_pitch, const uint2 bounds) {
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 4, 16));

            decx::calc::GPUK::fma_c_ivec4_2D << <grid, block >> > (_DAptr, __x, _DBptr, _Ddstptr, eq_pitch / 4, bounds);
        }

        static void dev_Kfma_c(de::Half* DA, de::Half __x, de::Half* DB, de::Half* Ddst, const size_t len) {
            half2 _x;
            _x.x = *((__half*)&__x);                _x.y = *((__half*)&__x);
            float4* Dptr_A = reinterpret_cast<float4*>(DA);
            float4* Dptr_B = reinterpret_cast<float4*>(DB);
            float4* Dptr_dst = reinterpret_cast<float4*>(Ddst);
            decx::calc::GPUK::fma_c_hvec8 << < decx::utils::ceil<size_t>(len / 8, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
                Dptr_A, _x, Dptr_B, Dptr_dst, len / 8);
        }

        static void dev_Kfma_c_2D(de::Half* DA, de::Half __x, de::Half* DB, de::Half* Ddst, const size_t eq_pitch, const uint2 bounds) {
            half2 _x;
            _x.x = *((__half*)&__x);                _x.y = *((__half*)&__x);
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 8, 16));

            decx::calc::GPUK::fma_c_hvec8_2D << <grid, block >> > (_DAptr, _x, _DBptr, _Ddstptr, eq_pitch / 8, bounds);
        }


        static void dev_Kfma_c(double* DA, double __x, double* DB, double* Ddst, const size_t len) {
            float4* Dptr_A = reinterpret_cast<float4*>(DA);
            float4* Dptr_B = reinterpret_cast<float4*>(DB);
            float4* Dptr_dst = reinterpret_cast<float4*>(Ddst);
            decx::calc::GPUK::fma_c_dvec2 << < decx::utils::ceil<size_t>(len / 2, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
                Dptr_A, __x, Dptr_B, Dptr_dst, len / 2);
        }

        static void dev_Kfma_c_2D(double* DA, double __x, double* DB, double* Ddst, const size_t eq_pitch, const uint2 bounds) {
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 2, 16));

            decx::calc::GPUK::fma_c_dvec2_2D << <grid, block >> > (_DAptr, __x, _DBptr, _Ddstptr, eq_pitch / 2, bounds);
        }
    }
}


#endif