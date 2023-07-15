/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "Array_creating.h"



void CreateMatrix()
{
    de::InitCPUInfo();
    de::InitCuda();

    de::Matrix& A = de::CreateMatrixRef(de::_FP64_, 101, 101, de::Page_Default);

    de::DH handle = de::cpu::Constant_fp64(A, 37);
    if (handle.error_type != de::DECX_SUCCESS) {
        printf(handle.error_string);
    }

    for (int i = A.Height() - 10; i < A.Height(); ++i) {
        for (int j = A.Width() - 10; j < A.Width(); ++j) {
            std::cout << std::setw(5) << *A.ptr_fp64(i, j);
        }
        std::cout << std::endl;
    }

    A.release();
}


void CreateVector()
{
    de::InitCPUInfo();
    de::InitCuda();

    de::Vector& A = de::CreateVectorRef(de::_FP64_, 1000, de::Page_Default);

    de::cpu::Constant_fp64(A, 37);
    
    for (int i = A.Len() - 10; i < A.Len() + 2; ++i) {
        std::cout << std::setw(5) << *A.ptr_fp64(i);
    }

    A.release();
}



int main()
{
    CreateMatrix();
    //CreateVector();

    system("pause");
    return 0;
}