/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _NLM_H_
#define _NLM_H_


#include "../classes/class_utils.h"
#include "../classes/Matrix.h"


namespace de
{
    namespace vis
    {
        namespace cuda
        {
            _DECX_API_ de::DH NLM_RGB(de::Matrix& src, de::Matrix& dst, uint search_window_size, uint template_window_size, float h);


            /**
            * @param search_window_size : radius
            * @param template_window_size : radius
            */
            _DECX_API_ de::DH NLM_RGB_keep_alpha(de::Matrix& src, de::Matrix& dst, uint search_window_size, uint template_window_size, float h);


            _DECX_API_ de::DH NLM_RGB_keep_alpha(de::Matrix& src, de::Matrix& dst, uint search_window_radius, uint template_window_radius, float h);


            _DECX_API_ de::DH NLM_Gray(de::Matrix& src, de::Matrix& dst, uint search_window_size, uint template_window_size, float h);
        }
    }
}

#endif