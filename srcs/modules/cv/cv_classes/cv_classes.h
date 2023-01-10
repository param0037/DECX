/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _CV_CLASS_H_
#define _CV_CLASS_H_

#include "../../core/basic.h"
#include "../../classes/classes_util.h"
#include "../../core/memory_management/MemBlock.h"
#include "../../core/allocators.h"



//#define IMG_ALIGNMENT_GRAY 16
//#define IMG_ALIGNMENT_RGBA 4

namespace de
{
    namespace vis
    {
        enum ImgConstructType
        {
            DE_UC1 = 1,
            DE_UC3 = 3,
            DE_UC4 = 4,
            DE_IMG_DEFAULT = 5,
            DE_IMG_4_ALIGN = 6
        };

        class _DECX_API_ Img
        {
        public:
            Img() {}

            virtual uint Width() { return 0; }

            virtual uint Height() { return 0; }

            virtual uchar* Ptr(const uint row, const uint col) {
                uchar* ptr = NULL;
                return ptr;
            }

            virtual void release() {
                return;
            }
        };
    }
}




namespace decx
{
    class _Img : public de::vis::Img
    {
    public:
        size_t ImgPlane, element_num, total_bytes;
        uint channel;

        int Mem_Store_Type, Image_Type;
        uint pitch;
        size_t _element_num, _total_bytes;

        decx::PtrInfo<uchar> Mat;
        uint width, height;

        _Img();


        _Img(const uint width, const uint heght, const int flag);


        virtual uint Width() { return this->width; }


        virtual uint Height() { return this->height; }


        virtual uchar* Ptr(const uint row, const uint col);


        virtual void release();
    };

}


/*
* I choose __m128 to load the uchar image array instead of __m256. If I use __m256 as the stride,
* register will be overloaded
*/
#define _IMG_ALIGN_ 4
#define _IMG_ALIGN4_ 4


namespace de
{
    namespace vis
    {
        _DECX_API_ de::vis::Img* CreateImgPtr(const uint width, const uint height, const int flag);


        _DECX_API_ de::vis::Img* CreateImgPtr();


        _DECX_API_ de::vis::Img& CreateImgRef(const uint width, const uint height, const int flag);


        _DECX_API_ de::vis::Img& CreateImgRef();
    }
}



#endif