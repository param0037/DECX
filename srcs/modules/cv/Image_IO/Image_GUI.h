/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
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