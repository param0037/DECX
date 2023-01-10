/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "Image_IO.h"
#include "../../core/error.h"


void decx::vis::Blt_Matrix2SDL_UC4(decx::_Matrix* src, SDL_Surface* dst)
{
    int* dst_data_ptr = (int*)dst->pixels;
    int* src_data_ptr = (int*)src->Mat.ptr;
    
    size_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < src->height; ++i) {
        dex_src = (size_t)src->pitch * (size_t)i;
        dex_dst = (size_t)dst->pitch * (size_t)i / 4;
        for (int j = 0; j < src->width; ++j) {
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

    size_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < src->height; ++i) {
        dex_src = (size_t)src->pitch * (size_t)i;
        dex_dst = (size_t)dst->pitch * (size_t)i / 4;
        for (int j = 0; j < src->width; ++j) {
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

    size_t dex_src = 0, dex_dst = 0;
    uchar4 pixel;

    for (int i = 0; i < src->height; ++i) {
        dex_src = (size_t)src->pitch * (size_t)i;
        dex_dst = (size_t)dst->pitch * (size_t)i / 4;
        for (int j = 0; j < src->width; ++j) {
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

    size_t dex_src = 0, dex_dst = 0;

    for (int i = 0; i < dst->height; ++i) {
        dex_src = (size_t)src->pitch * (size_t)i / 4;
        dex_dst = (size_t)dst->pitch * (size_t)i;
        for (int j = 0; j < dst->width; ++j) {
            dst_data_ptr[dex_dst] = src_data_ptr[dex_src];
            ++dex_src;
            ++dex_dst;
        }
    }
}


void decx::vis::Blt_SDL2Matrix_UC3(SDL_Surface* src, decx::_Matrix* dst)
{
    uchar3* src_data_ptr = (uchar3*)src->pixels;
    int* dst_data_ptr = (int*)dst->Mat.ptr;

    size_t dex_src = 0, dex_dst = 0;
    uchar3 _in_buffer;
    uchar4 _out_buffer;
    for (int i = 0; i < dst->height; ++i) {
        dex_src = (size_t)src->pitch * (size_t)i / 3;
        dex_dst = (size_t)dst->pitch * (size_t)i;
        for (int j = 0; j < dst->width; ++j) {
            _in_buffer = src_data_ptr[dex_src];

            _out_buffer.x = _in_buffer.x;
            _out_buffer.y = _in_buffer.y;
            _out_buffer.z = _in_buffer.z;
            _out_buffer.w = 255;

            dst_data_ptr[dex_dst] = *((int*)&_out_buffer);
            ++dex_src;
            ++dex_dst;
        }
    }
}



_DECX_API_ 
de::DH de::vis::ReadImage(const char* img_path, de::Matrix& src)
{
    de::DH handle;

    SDL_Surface* image = NULL;
    image = IMG_Load(img_path);
    if (image == NULL) {
        Print_Error_Message(4, IMAGE_LOAD_FAIL);
        decx::err::ImageLoadFailed(&handle);
        return handle;
    }

    decx::_Matrix* _src = dynamic_cast<decx::_Matrix*>(&src);
    //printf("(int)image->format->BytesPerPixel : %d\n", (int)image->format->BytesPerPixel);
    switch ((int)image->format->BytesPerPixel)
    {
    case 3:
        _src->re_construct(decx::_DATA_TYPES_FLAGS_::_UCHAR4_, image->w, image->h, _src->Store_Type);
        decx::vis::Blt_SDL2Matrix_UC3(image, _src);
        break;

    case 4:
        _src->re_construct(decx::_DATA_TYPES_FLAGS_::_UCHAR4_, image->w, image->h, _src->Store_Type);
        decx::vis::Blt_SDL2Matrix_UC4(image, _src);
        break;
    default:
        break;
    }
}