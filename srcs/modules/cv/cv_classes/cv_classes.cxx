/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#include "cv_classes.h"


decx::_Img::_Img()
{
    this->Mat.block = NULL;
    this->Mat.ptr = NULL;
}

decx::_Img::_Img(const uint width, const uint height, const int flag)
{
    this->height = height;
    this->width = width;
    this->ImgPlane = static_cast<size_t>(this->width) * static_cast<size_t>(this->height);

    switch (flag)
    {
    case de::vis::ImgConstructType::DE_UC1:        // UC1
        this->channel = 1;
        this->pitch = decx::utils::ceil<uint>(width, _IMG_ALIGN_) * _IMG_ALIGN_;
        break;

    case de::vis::ImgConstructType::DE_UC3:        // UC3
        this->channel = 4;
        this->pitch = decx::utils::ceil<uint>(width, _IMG_ALIGN4_) * _IMG_ALIGN4_;
        break;

    case de::vis::ImgConstructType::DE_UC4:        // UC4
        this->channel = 4;
        this->pitch = decx::utils::ceil<uint>(width, _IMG_ALIGN4_) * _IMG_ALIGN4_;
        break;

    default:    // default situation : align up with 4 bytes(uchar4)
        this->pitch = decx::utils::ceil<uint>(width, _IMG_ALIGN_) * _IMG_ALIGN_;
        this->channel = 4;
        break;
    }

    this->element_num = static_cast<size_t>(this->channel) * this->ImgPlane;
    this->total_bytes = this->element_num * sizeof(uchar);
    this->_element_num = (size_t)this->pitch * (size_t)this->height * (size_t)this->channel;
    this->_total_bytes = this->_element_num * sizeof(uchar);

    if (decx::alloc::_host_virtual_page_malloc(&this->Mat, this->_total_bytes)) {
        Print_Error_Message(4, ALLOC_FAIL);
        return;
    }
}



uchar* decx::_Img::Ptr(const uint row, const uint col)
{
    return (this->Mat.ptr + ((size_t)row * (size_t)(this->pitch) + col) * (size_t)(this->channel));
}





void decx::_Img::release()
{
#ifndef GNU_CPUcodes
    decx::alloc::_host_virtual_page_dealloc(&this->Mat);
#else
    decx::alloc::_host_virtual_page_dealloc(&this->Mat);
#endif
}





_DECX_API_
de::vis::Img* de::vis::CreateImgPtr(const uint width, const uint height, const int flag)
{
    return new decx::_Img(width, height, flag);
}


_DECX_API_
de::vis::Img& de::vis::CreateImgRef(const uint width, const uint height, const int flag)
{
    return *(new decx::_Img(width, height, flag));
}


_DECX_API_
de::vis::Img* de::vis::CreateImgPtr()
{
    return new decx::_Img();
}



_DECX_API_
de::vis::Img& de::vis::CreateImgRef()
{
    return *(new decx::_Img());
}