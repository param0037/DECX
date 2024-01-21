/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _ROTATE_FP32_H_
#define _ROTATE_FP32_H_


#include "../../basic_calculations/operators/cp_ops_exec.h"
#include "../../classes/classes_util.h"
#include "../../classes/Vector.h"
#include "../../classes/Matrix.h"


namespace decx {
    namespace dsp {
        /*
        * @param src : Input pointer
        * @param angle : The rotating angle
        * @param src : Output pointer
        * @param _proc_len : In vec4 (de::CPf x4)
        */
        void complex_rotate_fp32_caller(const double* src, const float angle, double* dst, const size_t _proc_len);


        namespace CPU {
            _DECX_API_ de::DH Complex_Rotate(de::Matrix& src, const float angle, de::Matrix& dst);


            _DECX_API_ de::DH Complex_Rotate(de::Vector& src, const float angle, de::Vector& dst);
        }
    }
}





#endif