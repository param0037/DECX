/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _IO_GUI_H_
#define _IO_GUI_H_


#include "../basic.h"
#include "../classes/Matrix.h"




namespace de {
	namespace vis {
		_DECX_API_ void ShowImg(de::Matrix& src, const char* window_name);


		_DECX_API_ de::DH ReadImage(const char* img_path, de::Matrix& src);


		_DECX_API_ void wait_untill_quit();
	}
}

#endif