/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _CALSS_UTILS_H_
#define _CALSS_UTILS_H_

#include "../core/basic.h"
#include "../core/configs/config.h"
#include "../core/utils/double_buffer.h"


namespace de
{
#ifdef _DECX_CUDA_PARTS_
    __align__(8) struct Point2D
    {
        int x, y;
        __device__ __host__ Point2D(const int _x, const int _y) { x = _x; y = _y; }
        __device__ __host__ Point2D() {}
    };
#endif
#ifdef _DECX_CPU_PARTS_
    __align__(8) struct Point2D
    {
        int x, y;
        Point2D(const int _x, const int _y) { x = _x; y = _y; }
        Point2D() {}
    };
#endif

#ifdef _DECX_CUDA_PARTS_
    __align__(16) struct Point3D
    {
        int x, y, z;
        __device__ __host__ Point3D(const int _x, const int _y, const int _z) { x = _x; y = _y; z = _z; }
        __device__ __host__ Point3D() {}
    };
#endif
#ifdef _DECX_CPU_PARTS_
    __align__(16) struct Point3D
    {
        int x, y, z;
        Point3D(const int _x, const int _y, const int _z) { x = _x; y = _y; z = _z; }
        Point3D() {}
    };
#endif

#ifdef _DECX_CUDA_PARTS_
    __align__(8) struct Point2D_f
    {
        float x, y;
        __device__ __host__ Point2D_f(const float _x, const float _y) { x = _x; y = _y; }
        __device__ __host__ Point2D_f() {}
    };
#endif
#ifdef _DECX_CPU_PARTS_
    __align__(8) struct Point2D_f
    {
        float x, y;
        Point2D_f(const float _x, const float _y) { x = _x; y = _y; }
        Point2D_f() {}
    };
#endif


#ifdef _DECX_CUDA_PARTS_
    __align__(16) struct Point2D_d
    {
        double x, y;
        __device__ __host__ Point2D_d(const double _x, const double _y) { x = _x; y = _y; }
        __device__ __host__ Point2D_d() {}
    };
#else
    __align__(16) struct Point2D_d
    {
        double x, y;
        Point2D_d(double _x, double _y) { x = _x; y = _y; }
        Point2D_d() {}
    };
#endif


    __align__(2) struct Half
    {
        unsigned short val;
    };


#ifdef _DECX_CUDA_PARTS_
    typedef struct __align__(4) complex_h
    {
        ushort real, image;

        __host__ __device__
        complex_h(const ushort Freal, const ushort Fimage);


        __device__ void dev_construct_with_phase(const __half angle);


        __host__ __device__ complex_h();
    }CPh;
#endif


    typedef struct __align__(8) _DECX_API_ complex_f
    {
        float real, image;

#ifdef _DECX_CUDA_PARTS_
        __host__ __device__
#endif
        complex_f(const float Freal, const float Fimage)
        {
            this->real = Freal;
            this->image = Fimage;
        }


#ifdef _DECX_CUDA_PARTS_
        __device__ void dev_construct_with_phase(const float angle)
        {
            this->real = __cosf(angle);
            this->image = __sinf(angle);
        }
#endif


        void construct_with_phase(const float angle)
        {
            this->real = cosf(angle);
            this->image = sinf(angle);
        }


#ifdef _DECX_CUDA_PARTS_
        __host__ __device__ __inline__
        de::complex_f& operator=(const de::complex_f & __src) {
            *((float2*)this) = *((float2*)&__src);
            return *this;
        }

        __host__ __device__ 
#endif
        complex_f() 
        {
            this->real = 0.f;
            this->image = 0.f;
        }
    }CPf;
}


#ifdef _DECX_CUDA_PARTS_
__align__(8) struct half4
{
    __half x, y, z, w;
};



__align__(8) struct half2_4
{
    half2 x, y;
};


__align__(16) struct half2_8
{
    half2 x, y, z, w;
};
#endif




#ifdef _DECX_CUDA_PARTS_
__device__
static bool operator>(de::Half& __a, de::Half& __b)
{
#if __ABOVE_SM_53
    __half res = __hsub(*((__half*)&__b.val), *((__half*)&__a.val));

    return (bool)((short)32768 & *((short*)&res));
#else
    return false;
#endif
}



__device__
static bool operator<(de::Half& __a, de::Half& __b)
{
#if __ABOVE_SM_53
    __half res = __hsub(*((__half*)&__a.val), *((__half*)&__b.val));

    return (bool)((short)32768 & *((short*)&res));
#else
    return false;
#endif
}


__device__
static de::Half& operator+(de::Half& __a, de::Half& __b)
{
#if __ABOVE_SM_53
    de::Half res;
    res.val = __hadd(*((__half*)&__a.val), *((__half*)&__b.val));
    return res;
#else
    short res = 0;
    return *((de::Half*)&res);
#endif
}

#endif



typedef __align__(16) union _16b
{
    float4          _f4vec;
    int4            _i4vec;
#ifdef _DECX_CUDA_PARTS_
    half2_8        _h28vec;
#endif
};

// REMEMBER! chage it to de::, aligned with the include headers for users
namespace decx
{
    enum Fp16_Accuracy_Levels
    {
        /**
        * Usually does the loads in fp16 but all calculations in fp32.
        * Usually the slowest method, this method cares about overflow (from 16-bit to 32-bit).
        */
        Fp16_Accurate_L1 = 0,

        /**
        * Usually does the loads in fp16 but all calculations in fp32.
        * Usually not that fast than the method Dot_Fp16_Accurate_L1, this method doesn't care about overflow.
        */
        Fp16_Accurate_L2 = 1,

        /**
        * Usually no accurate dispatching, do the loads and calculations all in fp16.
        * Usually the fastest method.
        */
        Fp16_Accurate_L3 = 2,
    };
}


#if _C_EXPORT_ENABLED_
#ifdef __cplusplus
extern "C"
{
#endif
    typedef struct __align__(8) DECX_Point2D_t
    {
        int32_t x, y;
    }DECX_Point2D;


    typedef struct __align__(8) DECX_Point2D_f_t
    {
        float x, y;
    }DECX_Point2D_f;


    typedef struct __align__(16) DECX_Point2D_d_t
    {
        double x, y;
    }DECX_Point2D_d;


    typedef struct __align__(8) DECX_Complex_f_t
    {
        float _real, _image;
    }DECX_CPf;
#ifdef __cplusplus
}
#endif
#endif


#endif