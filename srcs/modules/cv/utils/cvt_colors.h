/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _CVT_COLORS_H_
#define _CVT_COLORS_H_


#include "../../core/thread_management/thread_pool.h"
#include "../cv_classes/cv_classes.h"




namespace decx
{

    namespace vis 
    {
        typedef void channel_ops_kernel(float*, float*, const int2, const uint, const uint);

        /*
        * @param dims.x : The width of process area, in float
        * @param dims.y : The height of process area, in float
        * @param pitchsrc : The pitch of source matrix, in uchar4 (float) (int)
        * @param pitchsrc : The pitch of destinated matrix, in uchar
        */
        void _channel_ops_general_caller(decx::vis::channel_ops_kernel kernel, float* src, float* dst, const int2 dims,
            const uint pitchsrc, const uint pitchdst);
    }
}



namespace decx
{
    namespace vis {
        /*
        * src and dst have two different scale of pitch, take dst's as scale, which is 16x
        * @param pitchsrc : pitch of source matrix, in uchar4
        * @param pitchdst : pitch of destination matrix, in uint8_t
        * @param procW : width of process area, in uchar4
        * @param procH : height of process area
        */
        _THREAD_FUNCTION_ void _BGR2Gray_ST_UC2UC(float* src, float* dst, const int2 dims, const uint pitchsrc, const uint pitchdst);



        _THREAD_FUNCTION_ void _BGR2Mean_ST_UC2UC(float* src, float* dst, const int2 dims, const uint pitchsrc, const uint pitchdst);


        /*
        * src and dst have two different scale of pitch, take dst's as scale, which is 16x
        * dims : The dims info of processed area, dims.x : pitch(16x); dims.y : height
        */
        _THREAD_FUNCTION_ void _Preserve_B_ST_UC2UC(float* src, float* dst, const int2 dims, const uint pitchsrc, const uint pitchdst);

        /*
        * src and dst have two different scale of pitch, take dst's as scale, which is 16x
        * dims : The dims info of processed area, dims.x : pitch(16x); dims.y : height
        */
        _THREAD_FUNCTION_ void _Preserve_G_ST_UC2UC(float* src, float* dst, const int2 dims, const uint pitchsrc, const uint pitchdst);

        /*
        * src and dst have two different scale of pitch, take dst's as scale, which is 16x
        * dims : The dims info of processed area, dims.x : pitch(16x); dims.y : height
        */
        _THREAD_FUNCTION_ void _Preserve_R_ST_UC2UC(float* src, float* dst, const int2 dims, const uint pitchsrc, const uint pitchdst);

        /*
        * src and dst have two different scale of pitch, take dst's as scale, which is 16x
        * dims : The dims info of processed area, dims.x : pitch(16x); dims.y : height
        */
        _THREAD_FUNCTION_ void _Preserve_A_ST_UC2UC(float* src, float* dst, const int2 dims, const uint pitchsrc, const uint pitchdst);
    }
}

#endif