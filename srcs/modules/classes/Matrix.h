/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _MATRIX_H_
#define _MATRIX_H_

#include "../core/basic.h"
#include "classes_util.h"
#include "../core/allocators.h"
#include "../core/thread_management/thread_pool.h"
#include "store_types.h"
#include "type_info.h"


#define _MATRIX_ALIGN_4B_ 8
#define _MATRIX_ALIGN_2B_ 16
#define _MATRIX_ALIGN_8B_ 4
#define _MATRIX_ALIGN_1B_ 32



/**
* in host, allocate page-locaked memory in 8-times both on width and height
* ensure the utilization of __m128 and __m256, as well as multi threads
*/
namespace de {
    class _DECX_API_ Matrix
    {
    public:
        Matrix() {}


        virtual uint Width() = 0;


        virtual uint Height() = 0;



        /* return the reference of the element in the matrix, which locates on specific row and colume
        * \params row -> where the element locates on row
        * \params col -> where the element locates on colume
        */
        virtual float*      ptr_fp32(const int row, const int col)  = 0;
        virtual double*     ptr_fp64(const int row, const int col)  = 0;
        virtual int*        ptr_int32(const int row, const int col) = 0;
        virtual de::CPf*    ptr_cpl32(const int row, const int col) = 0;
        virtual de::Half*   ptr_fp16(const int row, const int col)  = 0;
        virtual uint8_t*    ptr_uint8(const int row, const int col) = 0;
        
        virtual void release() = 0;


        virtual de::Matrix& SoftCopy(de::Matrix& src) = 0;


        virtual int Type() = 0;


        ~Matrix() {}
    };
}




/**
* The data storage structure is shown below
*            <-------------- width ------------->
*            <---------------- pitch ------------------>
*             <-dpitch->
*            [[x x x x x x... ...    ... ...x x x....] T        
*            [[x x x x x x... ...    ... ...x x x....] |    
*            [[x x x x x x... ...    ... ...x x x....] |    
*            ...                                  ...  |    height    
*            ...                                  ...  |        
*            ...                                  ...  |        
*            [[x x x x x x... ...    ... ...x x x....] _
*/


namespace decx
{
    class _DECX_API_ _Matrix : public de::Matrix
    {
    private:
        // call AFTER attributes are assigned !
        void re_alloc_data_space();

        // call AFTER attributes are assigned !
        void alloc_data_space();


        void _attribute_assign(const int type, const uint _width, const uint _height, const int store_type);

    public:
        uint width, height;

        bool _init;

        unsigned short Store_Type;

        size_t element_num;
        uint pitch;            // the true width (NOT IN BYTES), it is aligned with 8

        decx::PtrInfo<void> Mat;

        int type, _single_element_size;

        size_t _element_num,    // true_width * true_height
            _total_bytes;       // true_width * true_height * sizeof(T)


        void construct(const int type, uint width, uint height, const int flag);


        void re_construct(const int type, uint width, uint height, const int flag);


        _Matrix();


        _Matrix(const int type, const uint _width, const uint _height, const int store_type);


        virtual uint Width();


        virtual uint Height();



        virtual float*      ptr_fp32(const int row, const int col);
        virtual double*     ptr_fp64(const int row, const int col);
        virtual int*        ptr_int32(const int row, const int col);
        virtual de::CPf*    ptr_cpl32(const int row, const int col);
        virtual de::Half*   ptr_fp16(const int row, const int col);
        virtual uint8_t*    ptr_uint8(const int row, const int col);


        virtual void release();


        virtual de::Matrix& SoftCopy(de::Matrix &src);


        virtual int Type();


        virtual ~_Matrix();
    };
}




#endif
