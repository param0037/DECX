/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _IMGPROC_H_
#define _IMGPROC_H_


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


namespace de
{
    namespace vis 
    {
        _DECX_API_ de::DH merge_channel(de::Matrix& src, de::Matrix& dst, const int flag);


        namespace cpu 
        {
            
            /**
            * @brief : Execute bilateral filter on an image
            * @param border_type : Selected from de::extend_label
            *   IF de::extend_label::_EXTEND_NONE_ : ignore the pixels on the convolutional border
            *   IF de::extend_label::_EXTEND_CONSTANTE_ : fill zeros to all the pixels on the convolutional border
            *   IF de::extend_label::_EXTEND_REFLECT_ : fill the reflections of the pixels on the convolutional border
            * @return 
            */
            _DECX_API_ de::DH Bilateral_Filter(de::Matrix& src, de::Matrix& dst, const de::Point2D neighbor_dims,
                const float sigma_space, const float sigma_color, const int border_type);



            _DECX_API_ de::DH Gaussian_Filter(de::Matrix& src, de::Matrix& dst, const de::Point2D neighbor_dims,
                const de::Point2D_f sigmaXY,
                const int border_type,
                const bool _is_central = true,
                const de::Point2D centerXY = de::Point2D(0, 0));
        }
    }
}




#endif