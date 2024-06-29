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


#ifndef _SDL_INCLUDES_H_
#define _SDL_INCLUDES_H_


#include "../../core/compile_params.h"

#ifdef Windows
//#include "../../../../3rdparty/Windows/SDL/SDL2-devel-2.6.2-VC/include/SDL.h"
//#include "../../../../3rdparty/Windows/SDL/SDL2-devel-2.6.2-VC/include/SDL_shape.h"
//#include "../../../../3rdparty/Windows/SDL/SDL2_image-devel-2.6.2-VC/include/SDL_image.h"

#include <SDL.h>
#include <SDL_shape.h>
#include <SDL_image.h>

#endif

#ifdef Linux
#include "../../../../3rdparty/Linux/SDL2/include/SDL.h"
#include "../../../../3rdparty/Linux/SDL2/include/SDL_shape.h"
#include "../../../../3rdparty/Linux/SDL2_Image/include/SDL_image.h"
#endif

#endif