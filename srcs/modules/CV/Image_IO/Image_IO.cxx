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


#include "../../core/error.h"
#include "Image_IO.h"


void decx::vis::Blt_Matrix2SDL_UC4(decx::_Matrix* src, SDL_Surface* dst)
{
    int* dst_data_ptr = (int*)dst->pixels;
    int* src_data_ptr = (int*)src->Mat.ptr;
    
    uint64_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < src->Height(); ++i) {
        dex_src = (uint64_t)src->Pitch() * (uint64_t)i;
        dex_dst = (uint64_t)dst->pitch * (uint64_t)i / 4;
        for (int j = 0; j < src->Width(); ++j) {
            dst_data_ptr[dex_dst] = src_data_ptr[dex_src];
            ++dex_src;
            ++dex_dst;
        }
    }
}



void decx::vis::Blt_Matrix2SDL_UC3(decx::_Matrix* src, SDL_Surface* dst)
{
    uchar3* dst_data_ptr = (uchar3*)dst->pixels;
    int* src_data_ptr = (int*)src->Mat.ptr;

    int buffer;

    uint64_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < src->Height(); ++i) {
        dex_src = (uint64_t)src->Pitch() * (uint64_t)i;
        dex_dst = (uint64_t)dst->pitch * (uint64_t)i / 4;
        for (int j = 0; j < src->Width(); ++j) {
            buffer = src_data_ptr[dex_src];
            dst_data_ptr[dex_dst] = *((uchar3*)&buffer);
            ++dex_src;
            ++dex_dst;
        }
    }
}


void decx::vis::Blt_Matrix2SDL_UC1(decx::_Matrix* src, SDL_Surface* dst)
{
    uchar4* dst_data_ptr = (uchar4*)dst->pixels;
    uint8_t* src_data_ptr = (uint8_t*)src->Mat.ptr;

    uint64_t dex_src = 0, dex_dst = 0;
    uchar4 pixel;

    for (int i = 0; i < src->Height(); ++i) {
        dex_src = (uint64_t)src->Pitch() * (uint64_t)i;
        dex_dst = (uint64_t)dst->pitch * (uint64_t)i / 4;
        for (int j = 0; j < src->Width(); ++j) {
            pixel.x = src_data_ptr[dex_src];
            pixel.y = pixel.x;
            pixel.z = pixel.x;
            pixel.w = 255;
            dst_data_ptr[dex_dst] = pixel;
            ++dex_src;
            ++dex_dst;
        }
    }
}



void decx::vis::Blt_SDL2Matrix_UC4(SDL_Surface* src, decx::_Matrix* dst)
{
    int* src_data_ptr = (int*)src->pixels;
    int* dst_data_ptr = (int*)dst->Mat.ptr;

    uint64_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < dst->Height(); ++i) {
        dex_src = (uint64_t)src->pitch * (uint64_t)i / 4;
        dex_dst = (uint64_t)dst->Pitch() * (uint64_t)i;
        for (int j = 0; j < dst->Width(); ++j) {
            dst_data_ptr[dex_dst] = src_data_ptr[dex_src];
            ++dex_src;
            ++dex_dst;
        }
    }

    dst->set_data_format(de::_DATA_FORMATS_::_COLOR_RGBA_);
}


void decx::vis::Blt_SDL2Matrix_UC3(SDL_Surface* src, decx::_Matrix* dst)
{
    uchar* src_data_ptr = (uchar*)src->pixels;
    int* dst_data_ptr = (int*)dst->Mat.ptr;
    
    uint64_t dex_src = 0, dex_dst = 0;
    uchar _in_buffer;
    uchar4 _out_buffer;
    for (int i = 0; i < dst->Height(); ++i) {
        dex_src = (uint64_t)src->pitch * (uint64_t)i;
        dex_dst = (uint64_t)dst->Pitch() * (uint64_t)i;
        for (int j = 0; j < dst->Width(); ++j) {
            _in_buffer = src_data_ptr[dex_src];
            _out_buffer.x = _in_buffer;

            _in_buffer = src_data_ptr[dex_src + 1];
            _out_buffer.y = _in_buffer;

            _in_buffer = src_data_ptr[dex_src + 2];
            _out_buffer.z = _in_buffer;
            _out_buffer.w = 255;

            dst_data_ptr[dex_dst] = *((int*)&_out_buffer);
            dex_src += 3;
            ++dex_dst;
        }
    }

    dst->set_data_format(de::_DATA_FORMATS_::_COLOR_RGB_);
}



_DECX_API_ 
de::DH de::vis::ReadImage(const char* img_path, de::Matrix& src)
{
    de::DH handle;

    SDL_Surface* image = NULL;
    image = IMG_Load(img_path);
    if (image == NULL) {
        decx::err::handle_error_info_modify(&handle, decx::DECX_error_types::DECX_FAIL_IMAGE_LOAD_FAILED,
            IMAGE_LOAD_FAIL);
        return handle;
    }
    
    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    switch (image->format->BytesPerPixel)
    {
    case 3:
        _src->re_construct(de::_DATA_TYPES_FLAGS_::_UCHAR4_, image->w, image->h);
        decx::vis::Blt_SDL2Matrix_UC3(image, _src);
        break;

    case 4:
        _src->re_construct(de::_DATA_TYPES_FLAGS_::_UCHAR4_, image->w, image->h);
        decx::vis::Blt_SDL2Matrix_UC4(image, _src);
        break;
    default:
        break;
    }

    return handle;
}