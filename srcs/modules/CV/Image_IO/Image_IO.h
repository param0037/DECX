/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _IMAGE_IO_H_
#define _IMAGE_IO_H_


#include "SDL_includes.h"
#include "../../classes/Matrix.h"
#include "../../core/error.h"


namespace decx
{
    namespace vis {
        void Blt_Matrix2SDL_UC4(decx::_Matrix* src, SDL_Surface* dst);


        void Blt_Matrix2SDL_UC3(decx::_Matrix* src, SDL_Surface* dst);


        void Blt_Matrix2SDL_UC1(decx::_Matrix* src, SDL_Surface* dst);


        void Blt_SDL2Matrix_UC4(SDL_Surface* src, decx::_Matrix* dst);


        void Blt_SDL2Matrix_UC3(SDL_Surface* src, decx::_Matrix* dst);
    }
}


namespace de
{
    namespace vis {
        _DECX_API_ de::DH ReadImage(const char* img_path, de::Matrix& src);
    }
}


#endif