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


#ifndef _IMAGE_GUI_H_
#define _IMAGE_GUI_H_


#include "SDL_includes.h"
#include "../../core/error.h"
#include "../../classes/Matrix.h"


namespace decx
{
    namespace vis {
        class ImgShow_workplace;
    }
}


class decx::vis::ImgShow_workplace
{
public:
    bool silence;
    SDL_Event event;
    Uint32 pixelFormat;
    int access;
    SDL_Rect texture_dimensions;

    SDL_Surface* image;
    SDL_Window* display_window;
    SDL_Renderer* renderer;
    SDL_Texture* texture;

    ImgShow_workplace();


    int Create_from_file_path(const char* img_path, const char* window_name);


    void Create_from_surface(SDL_Surface* surface, const char* window_name);


    void display();
};


namespace decx {
    namespace vis {
        void ImageRenderer(SDL_Renderer* renderer, SDL_Texture* texture, SDL_Rect texture_dimensions);


        extern std::vector<decx::vis::ImgShow_workplace> displayed_img_array;
    }
}


namespace de {
    namespace vis {
        _DECX_API_ void ShowImg(const char* img_path, const char* window_name);


        _DECX_API_ void ShowImg(de::Matrix& src, const char* window_name);


        _DECX_API_ void wait_untill_quit();
    }
}


#endif