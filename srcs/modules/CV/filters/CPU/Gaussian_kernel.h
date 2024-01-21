/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/


#ifndef _GAUSSIAN_KERNEL_H_
#define _GAUSSIAN_KERNEL_H_


#include "../../../core/basic.h"
#include "../../../classes/Matrix.h"


namespace decx
{
    namespace vis {
        class gaussian_kernel1D;
    }
}



class decx::vis::gaussian_kernel1D
{
public:
    decx::PtrInfo<float> _kernel_data;

    uint32_t _ker_length;
    float _mean, _sigma;


    gaussian_kernel1D(const uint32_t ker_length, const float sigma, bool central, const float mean = 0);


    void assign(decx::PtrInfo<float> kernel_data, const uint32_t ker_length, const float sigma, bool central, const float mean = 0);


    void generate();


    ~gaussian_kernel1D();
};



#endif