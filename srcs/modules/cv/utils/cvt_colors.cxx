/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#include "cvt_colors.h"
#include "../../core/utils/fragment_arrangment.h"



_THREAD_FUNCTION_ void decx::vis::_BGR2Gray_ST_UC2UC(float* src, float* dst, const int2 dims, 
    const uint pitchsrc, const uint pitchdst)
{
    size_t glo_dex_src = 0, glo_dex_dst = 0;
    __m128i __recv;
    uint i_res;
    uchar4 reg_dst;

    for (int i = 0; i < dims.y; ++i) {
        glo_dex_src = i * pitchsrc;
        glo_dex_dst = i * pitchdst / 4;
        for (int j = 0; j < dims.x; ++j) {
            *((__m128*) & __recv) = _mm_load_ps(src + glo_dex_src);
            glo_dex_src += 4;

            i_res = 0;
            i_res += (uint)((uchar*)&__recv)[0] * 19595;
            i_res += (uint)((uchar*)&__recv)[1] * 38469;
            i_res += (uint)((uchar*)&__recv)[2] * 7472;
            reg_dst.x = (uchar)(i_res >> 16);

            i_res = 0;
            i_res += (uint)((uchar*)&__recv)[4] * 19595;
            i_res += (uint)((uchar*)&__recv)[5] * 38469;
            i_res += (uint)((uchar*)&__recv)[6] * 7472;
            reg_dst.y = (uchar)(i_res >> 16);

            i_res = 0;
            i_res += (uint)((uchar*)&__recv)[8] * 19595;
            i_res += (uint)((uchar*)&__recv)[9] * 38469;
            i_res += (uint)((uchar*)&__recv)[10] * 7472;
            reg_dst.z = (uchar)(i_res >> 16);

            i_res = 0;
            i_res += (uint)((uchar*)&__recv)[12] * 19595;
            i_res += (uint)((uchar*)&__recv)[13] * 38469;
            i_res += (uint)((uchar*)&__recv)[14] * 7472;
            reg_dst.w = (uchar)(i_res >> 16);

            dst[glo_dex_dst] = *((float*)&reg_dst);

            ++glo_dex_dst;
        }
    }
}



_THREAD_FUNCTION_ void decx::vis::_BGR2Mean_ST_UC2UC(float* src, float* dst, const int2 dims, 
    const uint pitchsrc, const uint pitchdst)
{
    size_t glo_dex_src = 0, glo_dex_dst = 0;
    __m128i __recv;
    uint i_res;

    uchar4 reg_dst;

    for (int i = 0; i < dims.y; ++i) {
        glo_dex_src = i * pitchsrc;
        glo_dex_dst = i * pitchdst / 4;
        for (int j = 0; j < dims.x; ++j) {
            *((__m128*) & __recv) = _mm_load_ps(src + glo_dex_src);
            glo_dex_src += 4;

            i_res = 0;
            i_res += (uint)((uchar*)&__recv)[0];
            i_res += (uint)((uchar*)&__recv)[1];
            i_res += (uint)((uchar*)&__recv)[2];
            reg_dst.x = (uchar)(i_res / 3);

            i_res = 0;
            i_res += (uint)((uchar*)&__recv)[4];
            i_res += (uint)((uchar*)&__recv)[5];
            i_res += (uint)((uchar*)&__recv)[6];
            reg_dst.y = (uchar)(i_res / 3);

            i_res = 0;
            i_res += (uint)((uchar*)&__recv)[8];
            i_res += (uint)((uchar*)&__recv)[9];
            i_res += (uint)((uchar*)&__recv)[10];
            reg_dst.z = (uchar)(i_res / 3);

            i_res = 0;
            i_res += (uint)((uchar*)&__recv)[12];
            i_res += (uint)((uchar*)&__recv)[13];
            i_res += (uint)((uchar*)&__recv)[14];
            reg_dst.w = (uchar)(i_res / 3);

            dst[glo_dex_dst] = *((float*)&reg_dst);

            ++glo_dex_dst;
        }
    }
}



_THREAD_FUNCTION_ void decx::vis::_Preserve_B_ST_UC2UC(float* src, float* dst, const int2 dims, 
    const uint pitchsrc, const uint pitchdst)
{
    size_t glo_dex_src = 0, glo_dex_dst = 0;
    __m128i __recv;

    uchar4 reg_dst;

    for (int i = 0; i < dims.y; ++i) {
        glo_dex_src = i * pitchsrc;
        glo_dex_dst = i * pitchdst / 4;
        for (int j = 0; j < dims.x; ++j) {
            *((__m128*) & __recv) = _mm_load_ps(src + glo_dex_src);
            glo_dex_src += 4;

            reg_dst.x = ((uchar4*)&__recv)[0].z;
            reg_dst.y = ((uchar4*)&__recv)[1].z;
            reg_dst.z = ((uchar4*)&__recv)[2].z;
            reg_dst.w = ((uchar4*)&__recv)[3].z;

            dst[glo_dex_dst] = *((float*)&reg_dst);

            ++glo_dex_dst;
        }
    }
}



_THREAD_FUNCTION_ void decx::vis::_Preserve_G_ST_UC2UC(float* src, float* dst, const int2 dims,
    const uint pitchsrc, const uint pitchdst)
{
    size_t glo_dex_src = 0, glo_dex_dst = 0;
    __m128i __recv;

    uchar4 reg_dst;

    for (int i = 0; i < dims.y; ++i) {
        glo_dex_src = i * pitchsrc;
        glo_dex_dst = i * pitchdst / 4;
        for (int j = 0; j < dims.x; ++j) {
            *((__m128*) & __recv) = _mm_load_ps(src + glo_dex_src);
            glo_dex_src += 4;

            reg_dst.x = ((uchar4*)&__recv)[0].y;
            reg_dst.y = ((uchar4*)&__recv)[1].y;
            reg_dst.z = ((uchar4*)&__recv)[2].y;
            reg_dst.w = ((uchar4*)&__recv)[3].y;

            dst[glo_dex_dst] = *((float*)&reg_dst);

            ++glo_dex_dst;
        }
    }
}



_THREAD_FUNCTION_ void decx::vis::_Preserve_R_ST_UC2UC(float* src, float* dst, const int2 dims,
    const uint pitchsrc, const uint pitchdst)
{
    size_t glo_dex_src = 0, glo_dex_dst = 0;
    __m128i __recv;

    uchar4 reg_dst;

    for (int i = 0; i < dims.y; ++i) {
        glo_dex_src = i * pitchsrc;
        glo_dex_dst = i * pitchdst / 4;
        for (int j = 0; j < dims.x; ++j) {
            *((__m128*) & __recv) = _mm_load_ps(src + glo_dex_src);
            glo_dex_src += 4;

            reg_dst.x = ((uchar4*)&__recv)[0].x;
            reg_dst.y = ((uchar4*)&__recv)[1].x;
            reg_dst.z = ((uchar4*)&__recv)[2].x;
            reg_dst.w = ((uchar4*)&__recv)[3].x;

            dst[glo_dex_dst] = *((float*)&reg_dst);

            ++glo_dex_dst;
        }
    }
}



_THREAD_FUNCTION_ void decx::vis::_Preserve_A_ST_UC2UC(float* src, float* dst, const int2 dims,
    const uint pitchsrc, const uint pitchdst)
{
    size_t glo_dex_src = 0, glo_dex_dst = 0;
    __m128i __recv;

    uchar4 reg_dst;

    for (int i = 0; i < dims.y; ++i) {
        glo_dex_src = i * pitchsrc;
        glo_dex_dst = i * pitchdst / 4;
        for (int j = 0; j < dims.x; ++j) {
            *((__m128*) & __recv) = _mm_load_ps(src + glo_dex_src);
            glo_dex_src += 4;

            reg_dst.x = ((uchar4*)&__recv)[0].w;
            reg_dst.y = ((uchar4*)&__recv)[1].w;
            reg_dst.z = ((uchar4*)&__recv)[2].w;
            reg_dst.w = ((uchar4*)&__recv)[3].w;

            dst[glo_dex_dst] = *((float*)&reg_dst);

            ++glo_dex_dst;
        }
    }
}


// --------------------------------------- CALLERS --------------------------------------------------------


void decx::vis::_channel_ops_general_caller(decx::vis::channel_ops_kernel kernel, float* src, float* dst, const int2 dims, 
    const uint pitchsrc, const uint pitchdst)
{
    int _concurrent = (int)decx::thread_pool._hardware_concurrent;
    int2 sub_dims = make_int2(dims.x / 4, dims.y / _concurrent);
    size_t fragment_src = pitchsrc * (size_t)sub_dims.y, 
        fragment_dst = pitchdst * (size_t)sub_dims.y / 4,
        offset_src = 0,
        offset_dst = 0;

    std::future<void>* _thread_handle = new std::future<void>[_concurrent];

    for (int i = 0; i < _concurrent - 1; ++i) {
        _thread_handle[i] = decx::cpu::register_task(&decx::thread_pool, kernel, src + offset_src, dst + offset_dst, sub_dims,
            pitchsrc, pitchdst);
        offset_src += fragment_src;
        offset_dst += fragment_dst;
    }

    sub_dims.y = dims.y - (_concurrent - 1) * sub_dims.y;
    _thread_handle[decx::thread_pool._hardware_concurrent - 1] =
        decx::cpu::register_task(&decx::thread_pool, kernel, src + offset_src, dst + offset_dst, sub_dims,
            pitchsrc, pitchdst);

    for (int i = 0; i < _concurrent; ++i) {
        _thread_handle[i].get();
    }

    delete[] _thread_handle;
}
