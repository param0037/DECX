/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _FILTER2_H_
#define _FILTER2_H_


#include "../../../../classes/Matrix.h"
#include "../../../../classes/MatrixArray.h"


namespace de
{
    namespace cpu
    {
        _DECX_API_ de::DH Filter2D(de::Matrix& src, de::Matrix& kernel, de::Matrix& dst, const int flag, const de::_DATA_TYPES_FLAGS_ _output_type);


        _DECX_API_ de::DH Filter2D_single_channel(de::MatrixArray& src, de::Matrix& kernel, de::MatrixArray& dst, const int flag);


        _DECX_API_ de::DH Filter2D_multi_channel(de::MatrixArray& src, de::MatrixArray& kernel, de::MatrixArray& dst, const int flag);
    }
}




namespace decx
{
    namespace cpu
    {
        _DECX_API_ void Filter2D_Raw_API(decx::_Matrix* _src, decx::_Matrix* _kernel, decx::_Matrix* _dst,
            const int flag, const de::_DATA_TYPES_FLAGS_ _output_type, de::DH* handle);


        _DECX_API_ void Filter2D_single_channel_Raw_API(decx::_MatrixArray* _src, decx::_Matrix* _kernel, decx::_MatrixArray* _dst,
            const int flag, de::DH* handle);


        _DECX_API_ void Filter2D_multi_channel_Raw_API(decx::_MatrixArray* _src, decx::_MatrixArray* _kernel, decx::_MatrixArray* _dst,
            const int flag, de::DH* handle);
    }
}



#endif