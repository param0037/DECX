/**
*   ----------------------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ----------------------------------------------------------------------------------
* 
* This is a part of the open source project named "DECX", a high-performance scientific
* computational library. This project follows the MIT License. For more information 
* please visit https://github.com/param0037/DECX.
* 
* Copyright (c) 2021 Wayne Anderson
* 
* Permission is hereby granted, free of charge, to any person obtaining a copy of this 
* software and associated documentation files (the "Software"), to deal in the Software 
* without restriction, including without limitation the rights to use, copy, modify, 
* merge, publish, distribute, sublicense, and/or sell copies of the Software, and to 
* permit persons to whom the Software is furnished to do so, subject to the following 
* conditions:
* 
* The above copyright notice and this permission notice shall be included in all copies 
* or substantial portions of the Software.
* 
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, 
* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR 
* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE 
* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR 
* OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*/


#ifndef _CVT_COLORS_H_
#define _CVT_COLORS_H_


#include "../../core/thread_management/thread_pool.h"




namespace decx
{

    namespace vis 
    {
        typedef void channel_ops_kernel(const float*, float*, const int2, const uint32_t, const uint32_t);

        /*
        * @param dims.x : The width of process area, in float
        * @param dims.y : The height of process area, in float
        * @param pitchsrc : The pitch of source matrix, in uchar4 (float) (int)
        * @param pitchsrc : The pitch of destinated matrix, in uchar
        */
        void _channel_ops_UC42UC_caller(decx::vis::channel_ops_kernel kernel, const float* src, float* dst, const int2 dims,
            const uint32_t pitchsrc, const uint32_t pitchdst);


        void _channel_ops_UC42UC4_caller(decx::vis::channel_ops_kernel kernel, const float* src, float* dst, const int2 dims,
            const uint32_t pitchsrc, const uint32_t pitchdst);
    }
}



namespace decx
{
namespace vis 
{
    namespace CPUK
    {
        /*
        * src and dst have two different scale of pitch, take dst's as scale, which is 16x
        * @param pitchsrc : pitch of source matrix, in uchar4
        * @param pitchdst : pitch of destination matrix, in uint8_t
        * @param procW : width of process area, in uchar4
        * @param procH : height of process area
        */
        _THREAD_FUNCTION_ void _BGR2Gray_UC42UC(const float* src, float* dst, const int2 dims, const uint32_t pitchsrc, const uint32_t pitchdst);



        _THREAD_FUNCTION_ void _BGR2Mean_UC42UC(const float* src, float* dst, const int2 dims, const uint32_t pitchsrc, const uint32_t pitchdst);


        /*
        * src and dst have two different scale of pitch, take dst's as scale, which is 16x
        * dims : The dims info of processed area, dims.x : pitch(16x); dims.y : height
        */
        _THREAD_FUNCTION_ void _Preserve_B_UC42UC(const float* src, float* dst, const int2 dims, const uint32_t pitchsrc, const uint32_t pitchdst);

        /*
        * src and dst have two different scale of pitch, take dst's as scale, which is 16x
        * dims : The dims info of processed area, dims.x : pitch(16x); dims.y : height
        */
        _THREAD_FUNCTION_ void _Preserve_G_UC42UC(const float* src, float* dst, const int2 dims, const uint32_t pitchsrc, const uint32_t pitchdst);

        /*
        * src and dst have two different scale of pitch, take dst's as scale, which is 16x
        * dims : The dims info of processed area, dims.x : pitch(16x); dims.y : height
        */
        _THREAD_FUNCTION_ void _Preserve_R_UC42UC(const float* src, float* dst, const int2 dims, const uint32_t pitchsrc, const uint32_t pitchdst);

        /*
        * src and dst have two different scale of pitch, take dst's as scale, which is 16x
        * dims : The dims info of processed area, dims.x : pitch(16x); dims.y : height
        */
        _THREAD_FUNCTION_ void _Preserve_A_UC42UC(const float* src, float* dst, const int2 dims, const uint32_t pitchsrc, const uint32_t pitchdst);


        _THREAD_FUNCTION_ void _RGB2YUV_UC42UC4(const float* src, float* dst, const int2 dims, const uint32_t pitchsrc, const uint32_t pitchdst);


        _THREAD_FUNCTION_ void _YUV2RGB_UC42UC4(const float* src, float* dst, const int2 dims, const uint32_t pitchsrc, const uint32_t pitchdst);


        _THREAD_FUNCTION_ void _RGB2BGR_UC42UC4(const float* src, float* dst, const int2 dims, const uint32_t pitchsrc, const uint32_t pitchdst);
    }
}
}

/*
* uchar4 _IO, _res;
            int16_t Y, U, V;
            int16_t R, G, B;
            *((float*)&_IO) = src[glo_dex_src];

            R = _IO.x;
            G = _IO.y;
            B = _IO.z;

            Y = 77 * R + 150 * G + 29 * B;
            U = 32768 - 43 * R - 85 * G + 128 * B;
            V = 32768 + 128 * R - 107 * G - 21 * B;

            _res.x = (Y >> 8);
            _res.y = (U >> 8);
            _res.z = (V >> 8);
            _res.w = 255;

            dst[glo_dex_dst] = *((float*)&_res);

            glo_dex_src++;
            glo_dex_dst++;
*/

/*
* uchar4 _IO, _res;
            int16_t Y, U, V;
            int16_t R, G, B;
            *((float*)&_IO) = src[glo_dex_src];
            
            Y = _IO.x;
            U = _IO.y - 128;
            V = _IO.z - 128;

            R = Y + V + ((103 * V) >> 8);
            G = Y - ((183 * V) >> 8) - ((88 * U) >> 8);
            B = Y + U + ((198 * U) >> 8);

            _res.x = R < 0 ? 0 : (R > 255 ? 255 : R);
            _res.y = G < 0 ? 0 : (G > 255 ? 255 : G);
            _res.z = B < 0 ? 0 : (B > 255 ? 255 : B);
            _res.w = 255;

            dst[glo_dex_dst] = *((float*)&_res);

            glo_dex_src++;
            glo_dex_dst++;
*/

#endif