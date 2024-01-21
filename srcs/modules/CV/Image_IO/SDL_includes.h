/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _SDL_INCLUDES_H_
#define _SDL_INCLUDES_H_


#include "../../core/compile_params.h"

#ifdef Windows
#include "../../../../3rdparty/Windows/SDL/SDL2-devel-2.6.2-VC/include/SDL.h"
#include "../../../../3rdparty/Windows/SDL/SDL2-devel-2.6.2-VC/include/SDL_shape.h"
#include "../../../../3rdparty/Windows/SDL/SDL2_image-devel-2.6.2-VC/include/SDL_image.h"

#endif

#ifdef Linux
#include "../../../../3rdparty/Linux/SDL2/include/SDL.h"
#include "../../../../3rdparty/Linux/SDL2/include/SDL_shape.h"
#include "../../../../3rdparty/Linux/SDL2_Image/include/SDL_image.h"
#endif

#endif