/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _CALSS_UTILS_H_
#define _CALSS_UTILS_H_

#include "../core/basic.h"
#if defined(_DECX_CUDA_CODES_) || defined(_DECX_CPU_CODES_)
#include "../core/configs/config.h"

#ifdef _DECX_CPU_CODES_
#include "../core/thread_management/thread_arrange.h"
#endif

#endif


namespace de
{
#ifdef _DECX_CUDA_CODES_
    __align__(8) struct Point2D
    {
        int x, y;
        __device__ __host__ Point2D(const int _x, const int _y) { x = _x; y = _y; }
        __device__ __host__ Point2D() {}
    };
#endif
#ifdef _DECX_CPU_CODES_
    __align__(8) struct Point2D
    {
        int x, y;
        Point2D(int _x, int _y) { x = _x; y = _y; }
        Point2D() {}
    };
#endif

#ifdef _DECX_CUDA_CODES_
    __align__(8) struct Point2D_f
    {
        float x, y;
        __device__ __host__ Point2D_f(const float _x, const float _y) { x = _x; y = _y; }
        __device__ __host__ Point2D_f() {}
    };
#endif
#ifdef _DECX_CPU_CODES_
    __align__(8) struct Point2D_f
    {
        float x, y;
        Point2D_f(const float _x, const float _y) { x = _x; y = _y; }
        Point2D_f() {}
    };
#endif


#ifdef _DECX_CUDA_CODES_
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


#ifndef GNU_CPUcodes
    __align__(2) struct Half
    {
        unsigned short val;
    };
#endif


#ifdef _DECX_CUDA_CODES_
    typedef struct __align__(4) complex_h
    {
        ushort real, image;

        __host__ __device__
        complex_h(const ushort Freal, const ushort Fimage);


        __device__ void dev_construct_with_phase(const __half angle);


        __host__ __device__ complex_h();
    }CPh;
#endif


    typedef struct __align__(8) complex_f
    {
        float real, image;

#ifdef _DECX_CUDA_CODES_
        __host__ __device__
#endif
        complex_f(const float Freal, const float Fimage);


#ifdef _DECX_CUDA_CODES_
        __device__ void dev_construct_with_phase(const float angle);
#endif


        void construct_with_phase(const float angle);


#ifdef _DECX_CUDA_CODES_
        __host__ __device__ 
#endif
        complex_f();
    }CPf;
}


#ifdef _DECX_CUDA_CODES_
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




namespace decx
{
    namespace alloc {
        /*
        * This is a struct, which represents a buffer, and the two state words
        * indicates the states.
        */
        template<class T>
        struct MIF
        {
            /* This is the number of the size of the buffer that this->mem is pointing to,
            in bytes. */
            T* mem;
            
            bool 
            /* If true, the this buffer is loaded with data most recently. Otherwise, the
            data in it is relatively old. This state can be set by a function called
            decx::utils::set_mutex_memory_state<_Ty1, _Ty2>(MIF*, MIF*) */
            leading, 

            /* If true, this buffer is currently being used by calculation units (e.g. CUDA kernels) 
            This function is commonly used where device concurrency is needed. Otherwise, this buffer
            is idle. */
            _using;

            MIF() {
                leading = false;
                _using = false;
                mem = NULL;
            }

            MIF(T* _ptr) {
                leading = false;
                _using = false;
                mem = _ptr;
            }


            MIF(T* _ptr, const bool _leading) {
                leading = _leading;
                _using = false;
                mem = _ptr;
            }
        };
    }


    namespace utils
    {
        template <typename _Ty1, typename _Ty2>
        static inline void set_mutex_memory_state(decx::alloc::MIF<_Ty1>* _set_leading, decx::alloc::MIF<_Ty2>* _set_lagging);


        template <typename _Ty1, typename _Ty2, typename _Ty3>
        static inline void set_mutex_memory3_using(decx::alloc::MIF<_Ty1>* _proc_A, decx::alloc::MIF<_Ty2>* _proc_B,
            decx::alloc::MIF<_Ty3>* _proc_C);


        template <typename _Ty1, typename _Ty2, typename _Ty3>
        static inline void set_mutex_memory3_idle(decx::alloc::MIF<_Ty1>* _proc_A, decx::alloc::MIF<_Ty2>* _proc_B,
            decx::alloc::MIF<_Ty3>* _proc_C);


        template <typename _Ty1, typename _Ty2, typename _Ty3>
        static inline void set_mutex_memory2_using(decx::alloc::MIF<_Ty1>* _proc_A, decx::alloc::MIF<_Ty2>* _proc_B);


        template <typename _Ty1, typename _Ty2, typename _Ty3>
        static inline void set_mutex_memory2_idle(decx::alloc::MIF<_Ty1>* _proc_A, decx::alloc::MIF<_Ty2>* _proc_B);
    }
}



template <typename _Ty1, typename _Ty2>
inline void decx::utils::set_mutex_memory_state(decx::alloc::MIF<_Ty1>* _set_leading,
    decx::alloc::MIF<_Ty2>* _set_lagging)
{
    _set_leading->leading = true;
    _set_lagging->leading = false;
}



template <typename _Ty1, typename _Ty2, typename _Ty3>
inline void decx::utils::set_mutex_memory3_using(decx::alloc::MIF<_Ty1>* _proc_A, decx::alloc::MIF<_Ty2>* _proc_B,
    decx::alloc::MIF<_Ty3>* _proc_C)
{
    _proc_A->_using = true;
    _proc_B->_using = true;
    _proc_C->_using = true;
}



template <typename _Ty1, typename _Ty2, typename _Ty3>
inline void decx::utils::set_mutex_memory3_idle(decx::alloc::MIF<_Ty1>* _proc_A, decx::alloc::MIF<_Ty2>* _proc_B,
    decx::alloc::MIF<_Ty3>* _proc_C)
{
    _proc_A->_using = false;
    _proc_B->_using = false;
    _proc_C->_using = false;
}



template <typename _Ty1, typename _Ty2, typename _Ty3>
inline void decx::utils::set_mutex_memory2_using(decx::alloc::MIF<_Ty1>* _proc_A, decx::alloc::MIF<_Ty2>* _proc_B)
{
    _proc_A->_using = true;
    _proc_B->_using = true;
}



template <typename _Ty1, typename _Ty2, typename _Ty3>
inline void decx::utils::set_mutex_memory2_idle(decx::alloc::MIF<_Ty1>* _proc_A, decx::alloc::MIF<_Ty2>* _proc_B)
{
    _proc_A->_using = false;
    _proc_B->_using = false;
}



#ifdef _DECX_CUDA_CODES_
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


/*
* calculate and store the results of a integer is divided by another integer
* Both in type of unsigned int
*/
struct Num_uint
{
    Num_uint() {}

    Num_uint(uint _src, uint _denomi) {
        unsat_quo = _src / _denomi;
        _mod = _src % _denomi;
        unsatur = _src - _mod;
        over_quo = _mod == 0 ? unsat_quo : unsat_quo + 1;
    }

    uint unsat_quo;
    uint _mod;
    uint unsatur;
    uint over_quo;
};


/*
* calculate and store the results of a integer is divided by another integer
* Both in type of size_t
*/
struct Num_size_t
{
    Num_size_t() {}

    Num_size_t(size_t _src, size_t _denomi) {
        unsat_quo = _src / _denomi;
        _mod = _src % _denomi;
        unsatur = _src - _mod;
        over_quo = _mod == 0 ? unsat_quo : unsat_quo + 1;
    }

    size_t unsat_quo;
    size_t _mod;
    size_t unsatur;
    size_t over_quo;
};




typedef __align__(16) union _16b
{
    float4        _f4vec;
    int4        _i4vec;
#ifdef _DECX_CUDA_CODES_
    half2_8        _h28vec;
#endif
};


#endif