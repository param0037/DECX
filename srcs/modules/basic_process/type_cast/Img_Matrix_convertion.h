/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _IMG_MATRIX_CONVERTION_H_
#define _IMG_MATRIX_CONVERTION_H_

#include "../../classes/Matrix.h"
#include "../../cv/cv_classes/cv_classes.h"
#include "../../core/thread_management/thread_pool.h"


namespace decx
{
    /**
    * @param Wsrc : The width of input matrix, in uchar4
    * @param Wdst : The width of output matrix, in float4
    * @param height : The height of processed fragment
    */
    _THREAD_FUNCTION_ void
    _load_int8_2_fp32(float *src, float *dst, const uint Wsrc, const uint Wdst, const uint height);


    /**
    * @param Wsrc : The width of input matrix, in uchar4
    * @param Wdst : The width of output matrix, in float4
    * @param height : The height of processed fragment
    */
    static void 
    _cvt_int8_fp32_caller(float* src, float* dst, const uint Wsrc, const uint Wdst, const uint height);
}


_THREAD_FUNCTION_ void
decx::_load_int8_2_fp32(float* src, float* dst, const uint Wsrc, const uint Wdst, const uint height)
{
    uchar4 load_src;
    __m128 _cvted;

    size_t dex_src = 0, dex_dst = 0;
    /*
    * Wsrc -> aligned in 1x uchar4 (4); Wdst -> aligned in 2x float4 (8)
    * obviously, Wdst >= Wsrc;
    */
    size_t _gapW = (size_t)(Wdst - Wsrc) * 4;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < Wdst; ++j) {
            *((float*)&load_src) = src[dex_src];
            ++dex_src;

            _cvted = _mm_set_ps((float)((uchar*)&load_src)[3], (float)((uchar*)&load_src)[2],
                (float)((uchar*)&load_src)[1], (float)((uchar*)&load_src)[0]);

            _mm_store_ps(dst + dex_dst, _cvted);
            dex_dst += 4;
        }
        dex_dst += _gapW;
    }
}



static void
decx::_cvt_int8_fp32_caller(float* src, float* dst, const uint Wsrc, const uint Wdst, const uint height)
{
    const uint _available_thr = (uint)decx::cpI.cpu_concurrency;

    decx::utils::frag_manager f_mgrH;
    decx::utils::frag_manager_gen(&f_mgrH, height, _available_thr);

    std::future<void>* fut = new std::future<void>[_available_thr];

    float* src_ptr = src, *dst_ptr = dst;
    size_t frag_src = (size_t)Wsrc * (size_t)f_mgrH.frag_len;
    size_t frag_dst = ((size_t)Wdst * (size_t)f_mgrH.frag_len) << 2;

    if (f_mgrH.is_left) {
        for (int i = 0; i < _available_thr - 1; ++i) {
            fut[i] = decx::thread_pool.register_task(
                decx::_load_int8_2_fp32, src_ptr, dst_ptr, Wsrc, Wdst, f_mgrH.frag_len);
            src_ptr += frag_src;
            dst_ptr += frag_dst;
        }

        fut[_available_thr - 1] = decx::thread_pool.register_task(
            decx::_load_int8_2_fp32, src, dst, Wsrc, Wdst, f_mgrH.frag_left_over);
    }
    else {
        for (int i = 0; i < _available_thr; ++i) {
            fut[i] = decx::thread_pool.register_task(
                decx::_load_int8_2_fp32, src_ptr, dst_ptr, Wsrc, Wdst, f_mgrH.frag_len);
            src_ptr += frag_src;
            dst_ptr += frag_dst;
        }
    }

    for (int i = 0; i < _available_thr; ++i) {
        fut[i].get();
    }

    delete[] fut;
}




namespace de
{
    namespace cpu
    {
        _DECX_API_ de::DH Img2Matrix(de::vis::Img& src, de::Matrix<float>& dst);
    }
}


de::DH de::cpu::Img2Matrix(de::vis::Img& src, de::Matrix<float>& dst)
{
    de::DH handle;

    decx::_Img* _src = dynamic_cast<decx::_Img*>(&src);
    decx::_Matrix<float>* _dst = dynamic_cast<decx::_Matrix<float>*>(&dst);

    if (!decx::cpI.is_init) {
        decx::err::CPU_Not_init(&handle);
        Print_Error_Message(4, CPU_NOT_INIT);
        return handle;
    }

    if (_src->channel != 1) {
        decx::err::Channel_Error(&handle);
        Print_Error_Message(4, CHANNEL_ERROR);
        return handle;
    }

    decx::_cvt_int8_fp32_caller((float*)_src->Mat.ptr, _dst->Mat.ptr, 
        _src->pitch / 4, _dst->pitch / 4, 
        _dst->height);

    decx::err::Success(&handle);
}



#endif