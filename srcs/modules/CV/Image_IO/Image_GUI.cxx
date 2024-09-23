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


#include "Image_GUI.h"
#include "Image_IO.h"


decx::vis::ImgShow_workplace::ImgShow_workplace()
{
    this->display_window = NULL;
    this->image = NULL;
    this->renderer = NULL;
    this->texture = NULL;
    this->silence = true;

    this->texture_dimensions.x = 0;
    this->texture_dimensions.y = 0;
    this->texture_dimensions.w = 0;
    this->texture_dimensions.h = 0;
}


int decx::vis::ImgShow_workplace::Create_from_file_path(const char* img_path, const char* window_name)
{
    this->image = IMG_Load(img_path);
    if (this->image == NULL) {
        Print_Error_Message(4, IMAGE_LOAD_FAIL);
        return -1;
    }

    this->silence = false;
    this->texture_dimensions.x = 0;
    this->texture_dimensions.y = 0;
    this->texture_dimensions.w = this->image->w;
    this->texture_dimensions.h = this->image->h;

    this->display_window = SDL_CreateWindow(window_name, 10, 10,
        this->image->w,
        this->image->h,
        SDL_WINDOW_RESIZABLE | SDL_WINDOW_SHOWN);

    this->renderer = SDL_CreateRenderer(this->display_window, -1, 0);

    this->texture = (SDL_CreateTextureFromSurface(this->renderer, this->image));
    SDL_QueryTexture(this->texture,
        (Uint32*)&this->pixelFormat,
        (int*)&this->access,
        &this->texture_dimensions.w,
        &this->texture_dimensions.h);

    SDL_SetWindowSize(this->display_window, texture_dimensions.w, texture_dimensions.h);

    return 0;
}


void decx::vis::ImgShow_workplace::Create_from_surface(SDL_Surface* surface, const char* window_name)
{
    this->image = surface;

    this->silence = false;
    this->texture_dimensions.x = 0;
    this->texture_dimensions.y = 0;
    this->texture_dimensions.w = this->image->w;
    this->texture_dimensions.h = this->image->h;

    this->display_window = SDL_CreateWindow(window_name, 10, 10,
        this->image->w,
        this->image->h,
        SDL_WINDOW_RESIZABLE | SDL_WINDOW_SHOWN);

    this->renderer = SDL_CreateRenderer(this->display_window, -1, 0);

    this->texture = (SDL_CreateTextureFromSurface(this->renderer, this->image));
    SDL_QueryTexture(this->texture,
        (Uint32*)&this->pixelFormat,
        (int*)&this->access,
        &this->texture_dimensions.w,
        &this->texture_dimensions.h);

    SDL_SetWindowSize(this->display_window, texture_dimensions.w, texture_dimensions.h);
}



void decx::vis::ImgShow_workplace::display()
{
    SDL_PixelFormat* pixformat = NULL;
    pixformat = this->image->format;
    SDL_WindowShapeMode shapeMode1;

    if (pixformat->Amask != 0) {
        shapeMode1.mode = ShapeModeBinarizeAlpha;
        shapeMode1.parameters.binarizationCutoff = 255;
    }
    else {
        shapeMode1.mode = ShapeModeColorKey;
        shapeMode1.parameters.colorKey = { 0,0,0,0xff };
    }
    SDL_SetWindowShape(this->display_window, this->image, &shapeMode1);

    ImageRenderer(this->renderer, this->texture, texture_dimensions);
}



void decx::vis::ImageRenderer(SDL_Renderer* renderer, SDL_Texture* texture, SDL_Rect texture_dimensions)
{
    //Clear render-target to blue.
    SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, 0xff);
    SDL_RenderClear(renderer);

    //Render the texture.
    SDL_RenderCopy(renderer, texture, &texture_dimensions, &texture_dimensions);

    SDL_RenderPresent(renderer);
}


std::vector<decx::vis::ImgShow_workplace> decx::vis::displayed_img_array;


void de::vis::ShowImg(const char* img_path, const char* window_name)
{
    decx::vis::displayed_img_array.emplace_back();
    decx::vis::displayed_img_array.back().Create_from_file_path(img_path, window_name);
}


void de::vis::wait_untill_quit()
{
    SDL_Rect texture_dimensions;
    texture_dimensions.x = 0;
    texture_dimensions.y = 0;
    SDL_Event event;

    bool run = true;
    bool crit = true;
    while (run)
    {
        SDL_WaitEvent(&event);      // wait untill the indicated event occurs

        for (int i = 0; i < decx::vis::displayed_img_array.size(); ++i) {
            //refreshing the window
            texture_dimensions.w = decx::vis::displayed_img_array[i].image->w;
            texture_dimensions.h = decx::vis::displayed_img_array[i].image->h;
            decx::vis::ImageRenderer(decx::vis::displayed_img_array[i].renderer, decx::vis::displayed_img_array[i].texture, texture_dimensions);
            SDL_UpdateWindowSurface(decx::vis::displayed_img_array[i].display_window);

            if (SDL_GetWindowID(decx::vis::displayed_img_array[i].display_window) == event.window.windowID &&
                event.window.event == SDL_WINDOWEVENT_CLOSE) {
                SDL_DestroyWindow(decx::vis::displayed_img_array[i].display_window);
                decx::vis::displayed_img_array[i].silence = true;
            }
        }
        crit = true;
        for (int i = 0; i < decx::vis::displayed_img_array.size(); ++i) {
            crit &= decx::vis::displayed_img_array[i].silence;
        }
        // If all the windows are silent, then quit SDL
        if (crit) {
            run = false;
        }
    }
    
    for (int i = 0; i < decx::vis::displayed_img_array.size(); ++i) {
#ifndef Linux
        if (decx::vis::displayed_img_array[i].texture != NULL){
            SDL_DestroyTexture(decx::vis::displayed_img_array[i].texture);
        }
        if (decx::vis::displayed_img_array[i].renderer != NULL){
            SDL_DestroyRenderer(decx::vis::displayed_img_array[i].renderer);
        }
#endif
        if (decx::vis::displayed_img_array[i].image != NULL){
            SDL_FreeSurface(decx::vis::displayed_img_array[i].image);
        }
    }
}


_DECX_API_
void de::vis::ShowImg(de::Matrix& src, const char* window_name)
{
    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);

    SDL_Surface* image_surface = NULL;

    decx::vis::displayed_img_array.emplace_back();
    switch (_src->Type())
    {
    case de::_DATA_TYPES_FLAGS_::_UCHAR4_:
        image_surface = SDL_CreateRGBSurface(0, _src->Width(), _src->Height(), 32, 
            0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000);
        // copy data from decx::_Matrix::Mat.ptr to SDL_Surface::pixels
        decx::vis::Blt_Matrix2SDL_UC4(_src, image_surface);

        decx::vis::displayed_img_array.back().Create_from_surface(image_surface, window_name);
        break;

    case de::_DATA_TYPES_FLAGS_::_UINT8_:
        image_surface = SDL_CreateRGBSurface(0, _src->Width(), _src->Height(), 32,
            0x000000ff, 0x0000ff00, 0x00ff0000, 0xff000000);
        
        // image_surface = SDL_CreateRGBSurfaceWithFormat(0, _src->Width(), _src->Height(), 8,
        //     SDL_PIXELFORMAT_INDEX8);

        // copy data from decx::_Matrix::Mat.ptr to SDL_Surface::pixels
        decx::vis::Blt_Matrix2SDL_UC1(_src, image_surface);

        decx::vis::displayed_img_array.back().Create_from_surface(image_surface, window_name);
        break;
    default:
        break;
    }
}