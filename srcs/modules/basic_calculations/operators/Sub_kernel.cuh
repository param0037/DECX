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
* The _2D kernel function is designed for Tensor and TensorArray, treat Tensor
* and TensorArray as a 2D matrix, with eq_pitch = ~.dpitch * ~.width, height of
* ~.height (Tensor) or ~.tensor_num * ~.height (TensorArray)
*/

#ifndef _SUB_KERNEL_CUH_
#define _SUB_KERNEL_CUH_

#include "../../core/basic.h"
#include "../../core/configs/config.h"
#include "../../core/utils/decx_utils_functions.h"
#include "../../classes/classes_util.h"



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
            void sub_m_ivec4(float4* A, float4* B, float4* dst, const size_t len);



            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void sub_m_ivec4_2D(float4* A, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds);


            __global__
            /**
            * int* x2, add together
            * @param len : have considered vec4
            */
            void sub_m_fvec4(float4* A, float4* B, float4* dst, const size_t len);



            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void sub_m_fvec4_2D(float4* A, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds);



            __global__
            void sub_m_hvec8(float4* A, float4* B, float4* dst, const size_t len);



            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void sub_m_hvec8_2D(float4* A, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds);



            __global__
            /**
            * int* x2, add together
            * @param len : have considered vec4
            */
            void sub_m_dvec2(float4* A, float4* B, float4* dst, const size_t len);



            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void sub_m_dvec2_2D(float4* A, float4* B, float4* dst, const size_t eq_pitch, const uint2 bounds);



            // ----------------------------- C --------------------------------------


            __global__
            void sub_c_ivec4(float4* src, int __x, float4* dst, const size_t len);


            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void sub_c_ivec4_2D(float4* src, int __x, float4* dst, const size_t eq_pitch, const uint2 bounds);



            __global__
            void sub_cinv_ivec4(int __x, float4* src, float4* dst, const size_t len);



            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void sub_cinv_ivec4_2D(float4* src, int __x, float4* dst, const size_t eq_pitch, const uint2 bounds);



            __global__
            void sub_c_fvec4(float4* src, float __x, float4* dst, const size_t len);



            __global__
            void sub_cinv_fvec4(float __x, float4* src, float4* dst, const size_t len);


            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void sub_c_fvec4_2D(float4* src, float __x, float4* dst, const size_t eq_pitch, const uint2 bounds);



            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void sub_cinv_fvec4_2D(float4* src, float __x, float4* dst, const size_t eq_pitch, const uint2 bounds);



            __global__
            void sub_c_hvec8(float4* src, half2 __x, float4* dst, const size_t len);



            __global__
            void sub_cinv_hvec8(half2 __x, float4* src, float4* dst, const size_t len);



            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void sub_c_hvec8_2D(float4* src, half2 __x, float4* dst, const size_t eq_pitch, const uint2 bounds);



            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void sub_cinv_hvec8_2D(float4* src, half2 __x, float4* dst, const size_t eq_pitch, const uint2 bounds);



            __global__
            void sub_c_dvec2(float4* src, double __x, float4* dst, const size_t len);



            __global__
            void sub_cinv_dvec2(double __x, float4* src, float4* dst, const size_t len);



            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void sub_c_dvec2_2D(float4* src, double __x, float4* dst, const size_t eq_pitch, const uint2 bounds);



            __global__
            /**
            * int* x2, add together
            * @param eq_pitch : have considered vec4
            * @param bounds.x : The width, in float4
            * @param bounds.y : The height, in float
            */
            void sub_cinv_dvec2_2D(float4* src, double __x, float4* dst, const size_t eq_pitch, const uint2 bounds);
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
        // float
        static void dev_Ksub_m(float* DA, float* DB, float* Ddst, const size_t len) {
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::sub_m_fvec4 << < decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
                _DAptr, _DBptr, _Ddstptr, len / 4);
        }

        static void dev_Ksub_m_2D(float* DA, float* DB, float* Ddst, const size_t eq_pitch, const uint2 bounds) {
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 4, 16));

            decx::calc::GPUK::sub_m_fvec4_2D << <grid, block >> > (_DAptr, _DBptr, _Ddstptr, eq_pitch / 4, bounds);
        }
        // end float

        // int
        static void dev_Ksub_m(int* DA, int* DB, int* Ddst, const size_t len) {
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::sub_m_ivec4 << < decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
                _DAptr, _DBptr, _Ddstptr, len / 4);
        }

        static void dev_Ksub_m_2D(int* DA, int* DB, int* Ddst, const size_t eq_pitch, const uint2 bounds) {
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 4, 16));

            decx::calc::GPUK::sub_m_ivec4_2D << <grid, block >> > (_DAptr, _DBptr, _Ddstptr, eq_pitch / 4, bounds);
        }
        // end int

        // de::Half
        static void dev_Ksub_m(de::Half* DA, de::Half* DB, de::Half* Ddst, const size_t len) {
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::sub_m_hvec8 << < decx::utils::ceil<size_t>(len / 8, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
                _DAptr, _DBptr, _Ddstptr, len / 8);
        }

        static void dev_Ksub_m_2D(de::Half* DA, de::Half* DB, de::Half* Ddst, const size_t eq_pitch, const uint2 bounds) {
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 8, 16));

            decx::calc::GPUK::sub_m_hvec8_2D << <grid, block >> > (_DAptr, _DBptr, _Ddstptr, eq_pitch / 8, bounds);
        }
        // end de::Half

        // double
        static void dev_Ksub_m(double* DA, double* DB, double* Ddst, const size_t len) {
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::sub_m_dvec2 << < decx::utils::ceil<size_t>(len / 2, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
                _DAptr, _DBptr, _Ddstptr, len / 2);
        }

        static void dev_Ksub_m_2D(double* DA, double* DB, double* Ddst, const size_t eq_pitch, const uint2 bounds) {
            float4* _DAptr = reinterpret_cast<float4*>(DA);
            float4* _DBptr = reinterpret_cast<float4*>(DB);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 2, 16));

            decx::calc::GPUK::sub_m_dvec2_2D << <grid, block >> > (_DAptr, _DBptr, _Ddstptr, eq_pitch / 2, bounds);
        }
        // end double

        // ----------------------------------------------- C -------------------------------------------------------------


        static void dev_Ksub_c(float* Dsrc, float __x, float* Ddst, const size_t len) {
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::sub_c_fvec4 << <decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
                _Dsrcptr, __x, _Ddstptr, len / 4);
        }

        static void dev_Ksub_cinv(float __x, float* Dsrc, float* Ddst, const size_t len) {
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::sub_cinv_fvec4 << <decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
                __x, _Dsrcptr, _Ddstptr, len / 4);
        }

        static void dev_Ksub_c_2D(float* Dsrc, float __x, float* Ddst, const size_t eq_pitch, const uint2 bounds) {
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 4, 16));

            decx::calc::GPUK::sub_c_fvec4_2D << <grid, block >> > (_Dsrcptr, __x, _Ddstptr, eq_pitch / 4, bounds);
        }

        static void dev_Ksub_cinv_2D(float* Dsrc, float __x, float* Ddst, const size_t eq_pitch, const uint2 bounds) {
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 4, 16));

            decx::calc::GPUK::sub_cinv_fvec4_2D << <grid, block >> > (_Dsrcptr, __x, _Ddstptr, eq_pitch / 4, bounds);
        }
        // end float


        static void dev_Ksub_c(int* Dsrc, int __x, int* Ddst, const size_t len) {
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::sub_c_ivec4 << <decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
                _Dsrcptr, __x, _Ddstptr, len / 4);
        }

        static void dev_Ksub_cinv(int __x, int* Dsrc, int* Ddst, const size_t len) {
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::sub_cinv_ivec4 << <decx::utils::ceil<size_t>(len / 4, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
                __x, _Dsrcptr, _Ddstptr, len / 4);
        }

        static void dev_Ksub_c_2D(int* Dsrc, int __x, int* Ddst, const size_t eq_pitch, const uint2 bounds) {
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 4, 16));

            decx::calc::GPUK::sub_c_ivec4_2D << <grid, block >> > (_Dsrcptr, __x, _Ddstptr, eq_pitch / 4, bounds);
        }

        static void dev_Ksub_cinv_2D(int* Dsrc, int __x, int* Ddst, const size_t eq_pitch, const uint2 bounds) {
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 4, 16));

            decx::calc::GPUK::sub_cinv_ivec4_2D << <grid, block >> > (_Dsrcptr, __x, _Ddstptr, eq_pitch / 4, bounds);
        }
        // end int 

        // half
        static void dev_Ksub_c(de::Half* Dsrc, de::Half __x, de::Half* Ddst, const size_t len) {
            half2 _x;
            _x.x = *((__half*)&__x);                _x.y = *((__half*)&__x);
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::sub_c_hvec8 << <decx::utils::ceil<size_t>(len / 8, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
                _Dsrcptr, _x, _Ddstptr, len / 8);
        }

        static void dev_Ksub_cinv(de::Half __x, de::Half* Dsrc, de::Half* Ddst, const size_t len) {
            half2 _x;
            _x.x = *((__half*)&__x);                _x.y = *((__half*)&__x);
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::sub_cinv_hvec8 << <decx::utils::ceil<size_t>(len / 8, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
                _x, _Dsrcptr, _Ddstptr, len / 8);
        }

        static void dev_Ksub_c_2D(de::Half* Dsrc, de::Half __x, de::Half* Ddst, const size_t eq_pitch, const uint2 bounds) {
            half2 _x;
            _x.x = *((__half*)&__x);                _x.y = *((__half*)&__x);
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 8, 16));

            decx::calc::GPUK::sub_c_hvec8_2D << <grid, block >> > (_Dsrcptr, _x, _Ddstptr, eq_pitch / 8, bounds);
        }

        static void dev_Ksub_cinv_2D(de::Half* Dsrc, de::Half __x, de::Half* Ddst, const size_t eq_pitch, const uint2 bounds) {
            half2 _x;
            _x.x = *((__half*)&__x);                _x.y = *((__half*)&__x);
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 8, 16));

            decx::calc::GPUK::sub_cinv_hvec8_2D << <grid, block >> > (_Dsrcptr, _x, _Ddstptr, eq_pitch / 8, bounds);
        }
        // end half

        // double
        static void dev_Ksub_c(double* Dsrc, double __x, double* Ddst, const size_t len) {
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::sub_c_dvec2 << <decx::utils::ceil<size_t>(len / 2, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
                _Dsrcptr, __x, _Ddstptr, len / 2);
        }

        static void dev_Ksub_cinv(double __x, double* Dsrc, double* Ddst, const size_t len) {
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            decx::calc::GPUK::sub_cinv_dvec2 << <decx::utils::ceil<size_t>(len / 8, decx::cuP.prop.maxThreadsPerBlock), decx::cuP.prop.maxThreadsPerBlock >> > (
                __x, _Dsrcptr, _Ddstptr, len / 8);
        }

        static void dev_Ksub_c_2D(double* Dsrc, double __x, double* Ddst, const size_t eq_pitch, const uint2 bounds) {
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 2, 16));

            decx::calc::GPUK::sub_c_dvec2_2D << <grid, block >> > (_Dsrcptr, __x, _Ddstptr, eq_pitch / 2, bounds);
        }

        static void dev_Ksub_cinv_2D(double* Dsrc, double __x, double* Ddst, const size_t eq_pitch, const uint2 bounds) {
            float4* _Dsrcptr = reinterpret_cast<float4*>(Dsrc);
            float4* _Ddstptr = reinterpret_cast<float4*>(Ddst);

            dim3 block(16, 16);
            dim3 grid(decx::utils::ceil<uint>(bounds.y, 16), decx::utils::ceil<uint>(bounds.x / 2, 16));

            decx::calc::GPUK::sub_cinv_dvec2_2D << <grid, block >> > (_Dsrcptr, __x, _Ddstptr, eq_pitch / 2, bounds);
        }
        // end double
    }
}


#endif