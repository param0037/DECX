/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _GPU_MATRIX_H_
#define _GPU_MATRIX_H_

#include "../core/basic.h"
#include "../core/allocators.h"
#include "type_info.h"
#include "Matrix.h"


namespace de
{
    class _DECX_API_ GPU_Matrix
    {
    public:
        GPU_Matrix() {}


        virtual uint Width() = 0;


        virtual uint Height() = 0;

        
        virtual void release() = 0;


        virtual int Type() = 0;


        virtual de::GPU_Matrix& SoftCopy(de::GPU_Matrix& src) = 0;


        ~GPU_Matrix() {}
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
    class _DECX_API_ _GPU_Matrix : public de::GPU_Matrix
    {
    private:
        // call AFTER attributes are assigned !
        void re_alloc_data_space();

        // call AFTER attributes are assigned !
        void alloc_data_space();


        void _attribute_assign(const int _type, uint width, uint height);

        decx::_matrix_layout _layout;

        //unsigned short Store_Type;
        int type;

        bool _init;

        uint64_t total_bytes;

    public:
        
        //uint pitch;            // the true width (NOT IN BYTES), it is aligned with 4

        decx::PtrInfo<void> Mat;

        //int type, _single_element_size;


        void construct(const int _type, uint width, uint height);


        void re_construct(const int _type, uint width, uint height);


        _GPU_Matrix();


        _GPU_Matrix(const int _type, const uint _width, const uint _height);


        virtual uint Width();


        virtual uint Height();



        virtual void release();


        virtual int Type();


        virtual de::GPU_Matrix& SoftCopy(de::GPU_Matrix& src);


        ~_GPU_Matrix() {}


        uint32_t Pitch();


        const decx::_matrix_layout& get_layout();


        bool is_init();


        uint64_t get_total_bytes();
    };
}




#endif