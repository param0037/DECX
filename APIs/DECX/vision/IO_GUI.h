/**
*	---------------------------------------------------------------------
*	Author : Wayne Anderson
*   Date   : 2021.04.16
*	---------------------------------------------------------------------
*	This is a part of the open source program named "DECX", copyright c Wayne,
*	2021.04.16
*/


#ifndef _IO_GUI_H_
#define _IO_GUI_H_


#include "../basic.h"
#include "../classes/Matrix.h"


namespace de
{
    namespace vis
    {
        enum ImgChannelMergeType
        {
            BGR_to_Gray = 0,
            Preserve_B = 1,
            Preserve_G = 2,
            Preserve_R = 3,
            Preserve_Alpha = 4,
            RGB_mean = 5,
        };
    }
}


namespace de {
	namespace vis {
		_DECX_API_ void ShowImg(de::Matrix& src, const char* window_name);


		_DECX_API_ de::DH ReadImage(const char* img_path, de::Matrix& src);


		_DECX_API_ void wait_untill_quit();


		_DECX_API_ de::DH merge_channel(de::Matrix& src, de::Matrix& dst, const int flag);
	}
}

#endif