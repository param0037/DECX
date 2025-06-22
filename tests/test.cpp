// 
#include <iostream>
#include <DECX.h>
#include <iomanip>
#include <vector>

// #include <opencv4/opencv.hpp>

using namespace std;

//_DECX_API_ void test_CUDA();


// int main()
// {
//     de::InitCPUInfo();
//     de::InitCuda();

//     de::Matrix& src = de::CreateMatrixRef();
//     de::vis::ReadImage("./electric_1.jpg", src);
    
//     de::Matrix& img_recovered = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height());
//     de::Matrix& src_gray = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height());
//     de::Matrix& specturm = de::CreateMatrixRef(de::_COMPLEX_F32_, src.Width(), src.Height());
//     //de::Matrix& specturm_p = de::CreateMatrixRef(de::_COMPLEX_F32_, src.Width(), src.Height());
//     //de::GPU_Matrix& Dspecturm = de::CreateGPUMatrixRef(de::_COMPLEX_F32_, src.Width(), src.Height());

//     de::vis::ColorTransform(src, src_gray, de::vis::RGB_to_Gray);
//     de::dsp::cpu::FFT(src_gray, specturm, de::_COMPLEX_F32_);

//     //de::DH handle = de::dsp::cpu::Gaussian_Window2D(specturm, specturm_p, {0, 0}, {100, 100}, 0);
//     // // // cout << handle.error_string << endl;
//     de::dsp::cpu::IFFT(specturm, img_recovered, de::_UINT8_);

//     de::vis::ShowImg(src_gray, "1");
//     de::vis::ShowImg(img_recovered, "2");
//     de::vis::wait_untill_quit();

//     return 0;
// }



// #define W 1013
// #define H 1077
// #define L 1005

#define W 1024
#define H 1024
#define L 1013

#if 0
void decx_GEMM_cuda_fp32()
{
    de::InitCPUInfo();
    de::InitCuda();
    
    de::Matrix& A = de::CreateMatrixRef(de::_FP32_, L, H);
    de::GPU_Matrix& DA = de::CreateGPUMatrixRef(de::_FP32_, A.Width(), A.Height());

    de::Matrix& B = de::CreateMatrixRef(de::_FP32_, W, L);
    de::GPU_Matrix& DB = de::CreateGPUMatrixRef(de::_FP32_, B.Width(), B.Height());
    
    de::Matrix& C = de::CreateMatrixRef(de::_FP32_, W, H);
    de::GPU_Matrix& DC = de::CreateGPUMatrixRef(de::_FP32_, C.Width(), C.Height());

    de::Matrix& dst = de::CreateMatrixRef(de::_FP32_, W, H);
    de::GPU_Matrix& Ddst = de::CreateGPUMatrixRef(de::_FP32_, dst.Width(), dst.Height());

    for (int i = 0; i < A.Height(); ++i) {
        for (int j = 0; j < A.Width(); ++j) {
            *A.ptr<float>(i, j) = (float)j;
        }
    }

    for (int i = 0; i < B.Height(); ++i) {
        for (int j = 0; j < B.Width(); ++j) {
            *B.ptr<float>(i, j) = (float)i;
        }
    }

    for (int i = 0; i < C.Height(); ++i) {
        for (int j = 0; j < C.Width(); ++j) {
            *C.ptr<float>(i, j) = 1e3;
        }
    }

    de::Memcpy(A, DA, {0, 0}, {0, 0}, {A.Width(), A.Height()}, de::DECX_MEMCPY_H2D);
    de::Memcpy(B, DB, {0, 0}, {0, 0}, {B.Width(), B.Height()}, de::DECX_MEMCPY_H2D);
    de::Memcpy(C, DC, {0, 0}, {0, 0}, {C.Width(), C.Height()}, de::DECX_MEMCPY_H2D);

    de::Number alpha, beta;
    alpha = 1.1f;
    beta = 2.f;
    
    clock_t s, e;
    s = clock();
    for (int i = 0; i < 1000; ++i) {
        // de::blas::cuda::GEMM(DA, DB, Ddst);
        de::blas::cuda::GEMM(DA, DB, DC, Ddst, alpha, beta);
    }
    e = clock();
    de::Memcpy(dst, Ddst, {0, 0}, {0, 0}, {dst.Width(), dst.Height()}, de::DECX_MEMCPY_D2H);

    for (int i = dst.Height() - 10; i < dst.Height(); ++i) {
        for (int j = dst.Width() - 10; j < dst.Width(); ++j) {
    // for (int i = 0; i < 10; ++i) {
    //     for (int j = 0; j < 10; ++j) {
            std::cout << std::setw(15) << *dst.ptr<float>(i, j);
        }
        std::cout << std::endl;
    }

    float ref_res = 0;
    for (int i = 0; i < L; ++i){
        ref_res += i*i;
    }
    //cout << "The results should be :" << (int64_t)(ref_res) << endl;
    cout << "The results should be :" << (int64_t)(ref_res * 1.1 + 2 * 1e3) << endl;

    cout << "avg time spent : " << (e - s) / 1000.f / 1000.f << "msec" << endl;
}

void decx_GEMM_cpu()
{
    de::InitCPUInfo();
    de::InitCuda();

    de::cpu::DecxSetThreadingNum(12);

    de::Matrix& A = de::CreateMatrixRef(de::_FP32_, L, H);

    de::Matrix& B = de::CreateMatrixRef(de::_FP32_, W, L);

    de::Matrix& dst = de::CreateMatrixRef(de::_FP32_, W, H);

    for (int i = 0; i < A.Height(); ++i) {
        for (int j = 0; j < A.Width(); ++j) {
            *A.ptr<float>(i, j) = j;
        }
    }

    for (int i = 0; i < B.Height(); ++i) {
        for (int j = 0; j < B.Width(); ++j) {
            *B.ptr<float>(i, j) = i;
        }
    }

    
    for (int i = 0; i < 1; ++i) {
        de::blas::cpu::GEMM(A, B, dst);
    }
    
    for (int i = dst.Height() - 5; i < dst.Height(); ++i) {
        for (int j = dst.Width() - 5; j < dst.Width(); ++j) {
            std::cout << std::setw(15) << (int)*dst.ptr<float>(i, j);
        }
        std::cout << std::endl;
    }

    float ref = 0;
    for (int i = 0; i < L; ++i){
        ref += i * i;
    }

    cout << (int)ref << endl;
}
#endif


void decx_GEMM_cuda_fp64()
{
    de::InitCPUInfo();
    de::InitCuda();

    de::Matrix& A = de::CreateMatrixRef(de::_FP64_, L, H);
    de::GPU_Matrix& DA = de::CreateGPUMatrixRef(de::_FP64_, A.Width(), A.Height());

    de::Matrix& B = de::CreateMatrixRef(de::_FP64_, W, L);
    de::GPU_Matrix& DB = de::CreateGPUMatrixRef(de::_FP64_, B.Width(), B.Height());
    
    de::Matrix& C = de::CreateMatrixRef(de::_FP64_, W, H);
    de::GPU_Matrix& DC = de::CreateGPUMatrixRef(de::_FP64_, C.Width(), C.Height());

    de::Matrix& dst = de::CreateMatrixRef(de::_FP64_, W, H);
    de::GPU_Matrix& Ddst = de::CreateGPUMatrixRef(de::_FP64_, dst.Width(), dst.Height());

    for (int i = 0; i < A.Height(); ++i) {
        for (int j = 0; j < A.Width(); ++j) {
            *A.ptr<double>(i, j) = j;
        }
    }

    for (int i = 0; i < B.Height(); ++i) {
        for (int j = 0; j < B.Width(); ++j) {
            *B.ptr<double>(i, j) = i;
        }
    }

    for (int i = 0; i < C.Height(); ++i) {
        for (int j = 0; j < C.Width(); ++j) {
            *C.ptr<double>(i, j) = 1000;
        }
    }
    
    de::Memcpy(A, DA, {0, 0}, {0, 0}, {A.Width(), A.Height()}, de::DECX_MEMCPY_H2D);
    de::Memcpy(B, DB, {0, 0}, {0, 0}, {B.Width(), B.Height()}, de::DECX_MEMCPY_H2D);
    de::Memcpy(C, DC, {0, 0}, {0, 0}, {C.Width(), C.Height()}, de::DECX_MEMCPY_H2D);
    
    de::Number alpha, beta;
    alpha = 1.1;
    beta = 2.0;

    for (int i = 0; i < 1; ++i) {
         de::blas::cuda::GEMM(DA, DB, Ddst);
        //de::blas::cuda::GEMM(DA, DB, DC, Ddst, alpha, beta);
    }
    de::Memcpy(dst, Ddst, {0, 0}, {0, 0}, {dst.Width(), dst.Height()}, de::DECX_MEMCPY_D2H);

    for (int i = dst.Height() - 10; i < dst.Height(); ++i) {
        for (int j = dst.Width() - 10; j < dst.Width(); ++j) {
            std::cout << std::setw(15) << (int64_t)*dst.ptr<double>(i, j);
        }
        std::cout << std::endl;
    }

    double ref_res = 0;
    for (int i = 0; i < L; ++i){
        ref_res += i*i;
    }
    cout << "The results should be :" << (int64_t)ref_res << endl;
    //cout << "The results should be :" << (int64_t)(ref_res * 1.1 + 1000 * 2) << endl;
}



void decx_GEMM_cuda_cplxf()
{
    de::InitCPUInfo();
    de::InitCuda();

    de::Matrix& A = de::CreateMatrixRef(de::_COMPLEX_F32_, L, H);
    de::GPU_Matrix& DA = de::CreateGPUMatrixRef(de::_COMPLEX_F32_, A.Width(), A.Height());

    de::Matrix& B = de::CreateMatrixRef(de::_COMPLEX_F32_, W, L);
    de::GPU_Matrix& DB = de::CreateGPUMatrixRef(de::_COMPLEX_F32_, B.Width(), B.Height());
    

    de::Matrix& dst = de::CreateMatrixRef(de::_COMPLEX_F32_, W, H);
    de::GPU_Matrix& Ddst = de::CreateGPUMatrixRef(de::_COMPLEX_F32_, dst.Width(), dst.Height());

    for (int i = 0; i < A.Height(); ++i) {
        for (int j = 0; j < A.Width(); ++j) {
            *A.ptr<de::CPf>(i, j) = de::CPf(j, 0.f);
        }
    }

    for (int i = 0; i < B.Height(); ++i) {
        for (int j = 0; j < B.Width(); ++j) {
            *B.ptr<de::CPf>(i, j) = de::CPf(i, 0.f);
        }
    }

    de::Memcpy(A, DA, {0, 0}, {0, 0}, {A.Width(), A.Height()}, de::DECX_MEMCPY_H2D);
    de::Memcpy(B, DB, {0, 0}, {0, 0}, {B.Width(), B.Height()}, de::DECX_MEMCPY_H2D);
    
    for (int i = 0; i < 5000; ++i) {
        de::blas::cuda::GEMM(DA, DB, Ddst);
    }
    de::Memcpy(dst, Ddst, {0, 0}, {0, 0}, {dst.Width(), dst.Height()}, de::DECX_MEMCPY_D2H);

    for (int i = dst.Height() - 10; i < dst.Height(); ++i) {
        for (int j = dst.Width() - 10; j < dst.Width(); ++j) {
    // for (int i = 0; i < 10; ++i) {
    //     for (int j = 0; j < 10; ++j) {
            std::cout << std::setw(15) << (int64_t)dst.ptr<de::CPf>(i, j)->real;
        }
        std::cout << std::endl;
    }

    double ref_res = 0;
    for (int i = 0; i < L; ++i){
        ref_res += i*i;
    }
    cout << "The results should be :" << (int64_t)ref_res << endl;
}



void decx_GEMM_cuda_cplxd()
{
    de::InitCPUInfo();
    de::InitCuda();

    de::Matrix& A = de::CreateMatrixRef(de::_COMPLEX_F64_, L, H);
    de::GPU_Matrix& DA = de::CreateGPUMatrixRef(de::_COMPLEX_F64_, A.Width(), A.Height());

    de::Matrix& B = de::CreateMatrixRef(de::_COMPLEX_F64_, W, L);
    de::GPU_Matrix& DB = de::CreateGPUMatrixRef(de::_COMPLEX_F64_, B.Width(), B.Height());
    
    de::Matrix& C = de::CreateMatrixRef(de::_COMPLEX_F64_, W, H);
    de::GPU_Matrix& DC = de::CreateGPUMatrixRef(de::_COMPLEX_F64_, C.Width(), C.Height());

    de::Matrix& dst = de::CreateMatrixRef(de::_COMPLEX_F64_, W, H);
    de::GPU_Matrix& Ddst = de::CreateGPUMatrixRef(de::_COMPLEX_F64_, dst.Width(), dst.Height());

    for (int i = 0; i < A.Height(); ++i) {
        for (int j = 0; j < A.Width(); ++j) {
            *A.ptr<de::CPd>(i, j) = de::CPd(j, 0.f);
        }
    }

    for (int i = 0; i < B.Height(); ++i) {
        for (int j = 0; j < B.Width(); ++j) {
            *B.ptr<de::CPd>(i, j) = de::CPd(i, 0.f);
        }
    }

    for (int i = 0; i < C.Height(); ++i) {
        for (int j = 0; j < C.Width(); ++j) {
            *C.ptr<de::CPd>(i, j) = de::CPd(1000, 0.f);
        }
    }

    de::Memcpy(A, DA, {0, 0}, {0, 0}, {A.Width(), A.Height()}, de::DECX_MEMCPY_H2D);
    de::Memcpy(B, DB, {0, 0}, {0, 0}, {B.Width(), B.Height()}, de::DECX_MEMCPY_H2D);
    de::Memcpy(C, DC, {0, 0}, {0, 0}, {C.Width(), C.Height()}, de::DECX_MEMCPY_H2D);
    
    de::Number alpha, beta;
    alpha = de::CPd(1.1, 0);
    beta = de::CPd(2, 0);

    clock_t s, e;
    s = clock();
    for (int i = 0; i < 1000; ++i) {
        de::blas::cuda::GEMM(DA, DB, Ddst);
        //de::blas::cuda::GEMM(DA, DB, DC, Ddst, alpha, beta);
    }
    e = clock();
    de::Memcpy(dst, Ddst, {0, 0}, {0, 0}, {dst.Width(), dst.Height()}, de::DECX_MEMCPY_D2H);

    for (int i = dst.Height() - 10; i < dst.Height(); ++i) {
        for (int j = dst.Width() - 10; j < dst.Width(); ++j) {
    // for (int i = 0; i < 10; ++i) {
    //     for (int j = 0; j < 10; ++j) {
            std::cout << std::setw(15) << (int64_t)dst.ptr<de::CPd>(i, j)->real;
        }
        std::cout << std::endl;
    }

    double ref_res = 0;
    for (int i = 0; i < L; ++i){
        ref_res += i*i;
    }
    cout << "The results should be :" << (int64_t)(ref_res) << endl;
    //cout << "The results should be :" << (int64_t)(ref_res * 1.1 + 1000 * 2) << endl;
    cout << "avg time spent : " << (e - s) / 1000.f / 1000.f << "msec" << endl;
}



void decx_GEMM_cuda_fp16()
{
    de::InitCPUInfo();
    de::InitCuda();

    de::Matrix& A = de::CreateMatrixRef(de::_FP16_, L, H);
    de::GPU_Matrix& DA = de::CreateGPUMatrixRef(de::_FP16_, A.Width(), A.Height());

    de::Matrix& B = de::CreateMatrixRef(de::_FP16_, W, L);
    de::GPU_Matrix& DB = de::CreateGPUMatrixRef(de::_FP16_, B.Width(), B.Height());

    de::Matrix& dst = de::CreateMatrixRef(de::_FP32_, W, H);
    de::GPU_Matrix& Ddst = de::CreateGPUMatrixRef(de::_FP32_, dst.Width(), dst.Height());

    de::Matrix& C = de::CreateMatrixRef(de::_FP32_, W, H);
    de::GPU_Matrix& DC = de::CreateGPUMatrixRef(de::_FP32_, dst.Width(), dst.Height());

    for (int i = 0; i < A.Height(); ++i) {
        for (int j = 0; j < A.Width(); ++j) {
            *A.ptr<de::Half>(i, j) = de::Float2Half(j * 0.01);
        }
    }

    for (int i = 0; i < B.Height(); ++i) {
        for (int j = 0; j < B.Width(); ++j) {
            *B.ptr<de::Half>(i, j) = de::Float2Half(i * 0.01);
        }
    }

    for (int i = 0; i < C.Height(); ++i) {
        for (int j = 0; j < C.Width(); ++j) {
            *C.ptr<de::Half>(i, j) = de::Float2Half(100);
        }
    }
    
    de::Memcpy(A, DA, {0, 0}, {0, 0}, {A.Width(), A.Height()}, de::DECX_MEMCPY_H2D);
    de::Memcpy(B, DB, {0, 0}, {0, 0}, {B.Width(), B.Height()}, de::DECX_MEMCPY_H2D);
    de::Memcpy(C, DC, {0, 0}, {0, 0}, {C.Width(), C.Height()}, de::DECX_MEMCPY_H2D);
    
    de::Number alpha, beta;
    alpha = de::Float2Half(1.1);
    beta = de::Float2Half(2);

    for (int i = 0; i < 1; ++i) {
        // de::blas::cuda::GEMM(DA, DB, Ddst);
        de::blas::cuda::GEMM(DA, DB, DC, Ddst, alpha, beta);
    }
    de::Memcpy(dst, Ddst, {0, 0}, {0, 0}, {dst.Width(), dst.Height()}, de::DECX_MEMCPY_D2H);

    for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
    // for (int i = dst.Height() - 10; i < dst.Height(); ++i) {
    //     for (int j = dst.Width() - 10; j < dst.Width(); ++j) {
            std::cout << std::setw(15) << *dst.ptr<float>(i, j);
        }
        std::cout << std::endl;
    }

    float ref_res = 0;
    for (int i = 0; i < L; ++i){
        ref_res += i*i*0.01*0.01;
    }
    //cout << "The results should be :" << ref_res << endl;
    cout << "The results should be :" << 1.1 * ref_res + 2 * 100 << endl;
}



// #if 1
// void decx_transpose_cuda_fp16()
// {
//     de::InitCPUInfo();
//     de::InitCuda();

//     de::Matrix& src = de::CreateMatrixRef(de::_FP16_, W, H);
//     de::GPU_Matrix& Dsrc = de::CreateGPUMatrixRef(de::_FP16_, src.Width(), src.Height());

//     de::Matrix& dst = de::CreateMatrixRef(de::_FP16_, H, W);
//     de::GPU_Matrix& Ddst = de::CreateGPUMatrixRef(de::_FP16_, dst.Width(), dst.Height());

//     for (int i = 0; i < src.Height(); ++i) {
//         for (int j = 0; j < src.Width(); ++j) {
//             *src.ptr<de::Half>(i, j) = de::Float2Half(j);
//         }
//     }

//     de::Memcpy(src, Dsrc, {0, 0}, {0, 0}, {src.Width(), src.Height()}, de::DECX_MEMCPY_H2D);
    
//     for (int i = 0; i < 1; ++i) {
//         de::cuda::Transpose(Dsrc, Ddst);
//     }
//     de::Memcpy(dst, Ddst, {0, 0}, {0, 0}, {dst.Width(), dst.Height()}, de::DECX_MEMCPY_D2H);

//     // for (int i = 0; i < 10; ++i) {
//     //     for (int j = 0; j < 10; ++j) {
//     for (int i = dst.Height() - 10; i < dst.Height(); ++i) {
//         for (int j = dst.Width() - 10; j < dst.Width(); ++j) {
//             std::cout << std::setw(15) << de::Half2Float(*dst.ptr<de::Half>(i, j));
//         }
//         std::cout << std::endl;
//     }

// }



// void decx_transpose_cpu_fp16()
// {
//     de::InitCPUInfo();
//     de::InitCuda();

//     de::Matrix& src = de::CreateMatrixRef(de::_FP32_, W, H);

//     de::Matrix& dst = de::CreateMatrixRef(de::_FP32_, H, W);

//     for (int i = 0; i < src.Height(); ++i) {
//         for (int j = 0; j < src.Width(); ++j) {
//             *src.ptr<float>(i, j) = j;
//         }
//     }

//     for (int i = 0; i < 1; ++i) {
//         de::blas::cpu::Transpose(src, dst);
//     }

//     // for (int i = 0; i < 10; ++i) {
//     //     for (int j = 0; j < 10; ++j) {
//     for (int i = dst.Height() - 10; i < dst.Height(); ++i) {
//         for (int j = dst.Width() - 10; j < dst.Width(); ++j) {
//             std::cout << std::setw(15) << *dst.ptr<float>(i, j);
//         }
//         std::cout << std::endl;
//     }

// }
// #endif



#define N 1920*1080
void DECX_FFT1D()
{
    de::InitCPUInfo();
    de::InitCuda();

    de::Vector& src = de::CreateVectorRef(de::_FP64_, N);
    de::Vector& recovered = de::CreateVectorRef(de::_FP64_, N);
    de::Vector& dst = de::CreateVectorRef(de::_COMPLEX_F64_, N);

    for (int i = 0; i < src.Len(); ++i){
        *src.ptr<double>(i) = i;
    }

    for (int i = src.Len() - 10; i < src.Len(); ++i){
        cout << (int)*src.ptr<double>(i) << endl;
    }
    cout << endl;
    de::cpu::DecxSetThreadingNum(1);
    de::dsp::cpu::FFT(src, dst);
    // de::dsp::cuda::FFT(src, dst);

    for (int i = src.Len() - 10; i < src.Len(); ++i){
        cout << dst.ptr<de::CPd>(i)->real << ", " << dst.ptr<de::CPd>(i)->image << endl;
    }
    
    de::dsp::cpu::IFFT(dst, recovered, de::_FP64_);
    cout << endl;
    for (int i = src.Len() - 10; i < src.Len(); ++i){
        cout << (int)*recovered.ptr<double>(i) << endl;
    }
}


void DECX_FFT2D()
{
    de::InitCPUInfo();
    de::InitCuda();

    de::Matrix& src = de::CreateMatrixRef();
    de::vis::ReadImage("./electric_1.jpg", src);
    
    de::Matrix& img_recovered = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height());
    de::Matrix& spec = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height());
    de::GPU_Matrix& Dimg_recovered = de::CreateGPUMatrixRef(de::_UINT8_, src.Width(), src.Height());

    de::Matrix& src_gray = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height());
    de::GPU_Matrix& Dsrc_gray = de::CreateGPUMatrixRef(de::_UINT8_, src.Width(), src.Height());

    de::Matrix& specturm = de::CreateMatrixRef(de::_COMPLEX_F32_, src.Width(), src.Height());
    de::GPU_Matrix& Dspecturm = de::CreateGPUMatrixRef(de::_COMPLEX_F32_, src.Width(), src.Height());

    de::vis::ColorTransform(src, src_gray, de::vis::RGB_to_Gray);

    // de::Memcpy(src_gray, Dsrc_gray, {0, 0}, {0, 0}, {src.Width(), src.Height()}, de::DECX_MEMCPY_H2D);

    // de::dsp::cuda::FFT(Dsrc_gray, Dspecturm, de::_COMPLEX_F32_);
    de::dsp::cpu::FFT(src_gray, specturm, de::_COMPLEX_F32_);

    //de::dsp::cuda::LowPass2D_Ideal(Dspecturm, Dspecturm, de::Point2D(300, 300));
    // de::dsp::cuda::IFFT(Dspecturm, Dimg_recovered, de::_UINT8_);
    de::dsp::cpu::IFFT(specturm, img_recovered, de::_UINT8_);

    // de::Memcpy(img_recovered, Dimg_recovered, {0, 0}, {0, 0}, {src.Width(), src.Height()}, de::DECX_MEMCPY_D2H);
    // de::Memcpy(specturm, Dspecturm, {0, 0}, {0, 0}, {src.Width(), src.Height()}, de::DECX_MEMCPY_D2H);

    for (int i = 0; i < src.Height(); ++i){
        for (int j = 0; j < src.Width(); ++j){
            *spec.ptr<uint8_t>(i, j) = (abs(specturm.ptr<de::CPf>(i, j)->real) + abs(specturm.ptr<de::CPf>(i, j)->image)) / 200;
        }
    }

    de::vis::ShowImg(src_gray, "1");
    de::vis::ShowImg(spec, "spec");
    de::vis::ShowImg(img_recovered, "2");
    de::vis::wait_untill_quit();
}

#if 0
void DECX_filter2D()
{
    de::InitCPUInfo();
    de::InitCuda();

    de::Matrix& src = de::CreateMatrixRef();
    de::vis::ReadImage("./electric_1.jpg", src);

    de::Matrix& src_gray = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height());
    de::GPU_Matrix& Dsrc_gray = de::CreateGPUMatrixRef(src_gray.Type(), src.Width(), src.Height());

    de::GPU_Matrix& Ddst = de::CreateGPUMatrixRef(de::_FP32_, src.Width(), src.Height());

    de::GPU_Matrix& Dkernel = de::CreateGPUMatrixRef(de::_FP32_, 31, 31);
    de::Matrix& kernel = de::CreateMatrixRef(de::_FP32_, 31, 31);

    for (int i = 0; i < kernel.Height(); ++i){
        for (int j = 0; j < kernel.Width(); ++j){
            *kernel.ptr<float>(i, j) = 1.f / (31 * 31);
        }
    }

    de::vis::ColorTransform(src, src_gray, de::vis::RGB_to_Gray);

    de::Memcpy(src_gray, Dsrc_gray, {0, 0}, {0, 0}, {src_gray.Width(), src_gray.Height()}, de::DECX_MEMCPY_H2D);
    de::Memcpy(kernel, Dkernel, {0, 0}, {0, 0}, {kernel.Width(), kernel.Height()}, de::DECX_MEMCPY_H2D);

    de::dsp::cuda::Filter2D(Dsrc_gray, Dkernel, Ddst, de::_EXTEND_NONE_, de::_FP32_);

    de::Matrix& dst = de::CreateMatrixRef(de::_FP32_, Ddst.Width(), Ddst.Height());
    de::Memcpy(dst, Ddst, {0, 0}, {0, 0}, {dst.Width(), dst.Height()}, de::DECX_MEMCPY_D2H);


    de::Matrix& dst_show = de::CreateMatrixRef(de::_UINT8_, Ddst.Width(), Ddst.Height());
    

    for (int i = 0; i < dst.Height(); ++i){
        for (int j = 0; j < dst.Width(); ++j){
            *dst_show.ptr<uint8_t>(i, j) = *dst.ptr<float>(i, j);
        }
    }

    de::vis::ShowImg(src_gray, "1");
    de::vis::ShowImg(dst_show, "2");
    de::vis::wait_untill_quit();
}



// void DECX_filter2D_cpu()
// {
//     de::InitCPUInfo();
//     de::InitCuda();

//     de::Matrix& src = de::CreateMatrixRef();
//     de::vis::ReadImage("./electric_1.jpg", src);

//     de::Matrix& src_gray = de::CreateMatrixRef(de::_UINT8_, src.Width(), src.Height());
//     de::Matrix& src_gray_fp32 = de::CreateMatrixRef(de::_FP32_, src.Width(), src.Height());
//     de::Matrix& kernel = de::CreateMatrixRef(de::_FP32_, 31, 31);
//     de::Matrix& dst = de::CreateMatrixRef();

//     for (int i = 0; i < kernel.Height(); ++i){
//         for (int j = 0; j < kernel.Width(); ++j){
//             *kernel.ptr<float>(i, j) = 1.f / (31 * 31);
//         }
//     }

//     de::vis::ColorTransform(src, src_gray, de::vis::RGB_to_Gray);

//     de::cpu::TypeCast(src_gray, src_gray_fp32, de::CVT_UINT8_FP32);
//     clock_t s, e;
//     s = clock();
//     for (int i = 0; i < 100; ++i)
//         de::dsp::cpu::Filter2D(src_gray_fp32, kernel, dst, de::_EXTEND_NONE_, de::_FP32_);
//     e = clock();
//     de::Matrix& dst_show = de::CreateMatrixRef(de::_UINT8_, dst.Width(), dst.Height());
    

//     for (int i = 0; i < dst.Height(); ++i){
//         for (int j = 0; j < dst.Width(); ++j){
//             *dst_show.ptr<uint8_t>(i, j) = *dst.ptr<float>(i, j);
//         }
//     }
//     cout << "time spent (sec) : " << (float)(e - s)/CLOCKS_PER_SEC << endl;

//     de::vis::ShowImg(src_gray, "1");
//     de::vis::ShowImg(dst_show, "2");
//     de::vis::wait_untill_quit();
// }

void decx_type_cast()
{
    de::InitCPUInfo();

    de::Matrix& src = de::CreateMatrixRef(de::_FP64_, 103, 110);
    de::Matrix& dst = de::CreateMatrixRef(de::_FP32_, 103, 110);

    for (int i = 0; i < src.Height(); ++i){
        for (int j = 0; j < src.Width(); ++j){
            *src.ptr<double>(i, j) = (i + j);
            // *src.ptr<float>(i, j) = 37;
        }
    }
    
    // de::cpu::DecxSetThreadingNum(1);
    // de::cpu::TypeCast(src, dst, de::CVT_FP32_UINT8 | de::CVT_UINT8_SATURATED);
    de::cpu::TypeCast(src, dst, de::CVT_FP64_FP32);

    // for (int i = 0; i < 10; ++i){
    //     for (int j = 0; j < 10; ++j){
    for (int i = dst.Height() - 10; i < dst.Height(); ++i){
        for (int j = dst.Width() - 10; j < dst.Width(); ++j){
            cout << *dst.ptr<float>(i, j) << setw(5);
        }
        cout << endl;
    }
}

void decx_type_cast_vec()
{
    de::InitCPUInfo();

    de::Vector& src = de::CreateVectorRef(de::_FP64_, 1013);
    de::Vector& dst = de::CreateVectorRef(de::_FP32_, 1013);

    for (int i = 0; i < src.Len(); ++i){
        *src.ptr<double>(i) = (i);
        // *src.ptr<float>(i, j) = 37;
    }
    
    // de::cpu::DecxSetThreadingNum(1);
    // de::cpu::TypeCast(src, dst, de::CVT_FP32_UINT8 | de::CVT_UINT8_SATURATED);
    de::cpu::TypeCast(src, dst, de::CVT_FP64_FP32);

    // for (int i = 0; i < 10; ++i){
    for (int i = dst.Len() - 10; i < dst.Len() + 5; ++i){
        cout << *dst.ptr<float>(i) << endl;
    }
}


#include <immintrin.h>
extern "C" __m256d _my_avx_cos_fp64x4(__m256d);

#include <cmath>


extern "C" _DECX_API_ __m128 _avx_cos_fp32x4(__m128);
extern "C" _DECX_API_ __m128 _avx_sin_fp32x4(__m128);
extern "C" _DECX_API_ __m128d _avx_cos_fp64x2(__m128d);
extern "C" _DECX_API_ __m128d _avx_sin_fp64x2(__m128d);
extern "C" _DECX_API_ __m256 _avx_cos_fp32x8(__m256);
extern "C" _DECX_API_ __m256 _avx_sin_fp32x8(__m256);
extern "C" _DECX_API_ __m256d _avx_cos_fp64x4(__m256d);
extern "C" _DECX_API_ __m256d _avx_sin_fp64x4(__m256d);


void decx_mat_arithmetic()
{
    de::InitCPUInfo();

    de::Matrix& A = de::CreateMatrixRef(de::_FP64_, 1000, 1000);
    de::Matrix& B = de::CreateMatrixRef(de::_FP64_, 1000, 1000);
    de::Matrix& dst = de::CreateMatrixRef(de::_FP64_, 1000, 1000);

    for (int i = 0; i < A.Height(); ++i){
        for (int j = 0; j < A.Width(); ++j){
            *A.ptr<double>(i, j) = (i + j);
            *B.ptr<double>(i, j) = sinf(i + j);
        }
    }

    for (int i = dst.Height() - 10; i < dst.Height(); ++i){
        for (int j = dst.Width() - 10; j < dst.Width(); ++j){
            cout << *A.ptr<double>(i, j) << setw(5);
        }
        cout << endl;
    }
    
    cout << endl;

    de::blas::cpu::Arithmetic(A, B, dst, de::MAX);
    // de::blas::cpu::Arithmetic(A, dst, de::SIN);

    // for (int i = 0; i < 10; ++i){
    //     for (int j = 0; j < 10; ++j){
    for (int i = dst.Height() - 10; i < dst.Height(); ++i){
        for (int j = dst.Width() - 10; j < dst.Width(); ++j){
            cout << *dst.ptr<double>(i, j) << setw(10);
        }
        cout << endl;
    }
    cout << "refs:" << endl;
    // for (int i = 0; i < 10; ++i){
    //     for (int j = 0; j < 10; ++j){
    for (int i = dst.Height() - 10; i < dst.Height(); ++i){
        for (int j = dst.Width() - 10; j < dst.Width(); ++j){
            cout << *B.ptr<double>(i, j) << setw(10);
        }
        cout << endl;
    }
}


void decx_mat_arithmetic_cuda()
{
    de::InitCPUInfo();
    de::InitCuda();

    de::Matrix& A = de::CreateMatrixRef(de::_FP32_, 1000, 1000);
    de::GPU_Matrix& DA = de::CreateGPUMatrixRef(de::_FP32_, 1000, 1000);
    de::Matrix& B = de::CreateMatrixRef(de::_FP32_, 1000, 1000);
    de::Matrix& dst = de::CreateMatrixRef(de::_FP32_, 1000, 1000);
    de::GPU_Matrix& Ddst = de::CreateGPUMatrixRef(de::_FP32_, 1000, 1000);

    for (int i = 0; i < A.Height(); ++i){
        for (int j = 0; j < A.Width(); ++j){
            *A.ptr<float>(i, j) = (i + j);
            *B.ptr<float>(i, j) = sinf(i + j);
        }
    }

    for (int i = dst.Height() - 10; i < dst.Height(); ++i){
        for (int j = dst.Width() - 10; j < dst.Width(); ++j){
            cout << *A.ptr<float>(i, j) << setw(5);
        }
        cout << endl;
    }
    
    cout << endl;

    de::Memcpy(A, DA, {0, 0}, {0, 0}, {A.Width(), A.Height()}, de::DECX_MEMCPY_H2D);

    // de::blas::cpu::Arithmetic(A, B, dst, de::MAX);
    de::blas::cuda::Arithmetic(DA, Ddst, de::SIN);

    de::Memcpy(dst, Ddst, {0, 0}, {0, 0}, {dst.Width(), dst.Height()}, de::DECX_MEMCPY_D2H);

    for (int i = 0; i < 10; ++i){
        for (int j = 0; j < 10; ++j){
    // for (int i = dst.Height() - 10; i < dst.Height(); ++i){
    //     for (int j = dst.Width() - 10; j < dst.Width(); ++j){
            cout << *dst.ptr<float>(i, j) << setw(10);
        }
        cout << endl;
    }
    cout << "refs:" << endl;
    // for (int i = 0; i < 10; ++i){
    //     for (int j = 0; j < 10; ++j){
    for (int i = dst.Height() - 10; i < dst.Height(); ++i){
        for (int j = dst.Width() - 10; j < dst.Width(); ++j){
            cout << *B.ptr<float>(i, j) << setw(10);
        }
        cout << endl;
    }
}

void decx_vec_arithmetic()
{
    de::InitCPUInfo();

    de::Vector& A = de::CreateVectorRef(de::_FP32_, 1000);
    de::Vector& B = de::CreateVectorRef(de::_FP32_, 1000);
    de::Vector& dst = de::CreateVectorRef(de::_FP32_, 1000);

    for (int i = 0; i < A.Len(); ++i){
        *A.ptr<float>(i) = (i);
        *B.ptr<float>(i) = 1000 - i;
    }
    
    de::blas::cpu::Arithmetic(A, B, dst, de::ADD);
    // de::blas::cpu::Arithmetic(A, dst, de::COS);
	de::DH* handle = de::GetLastError();
    printf("%s\n", handle->error_string);
    
    for (int i = 0; i < 10; ++i){
    // for (int i = dst.Len() - 10; i < dst.Len(); ++i){
        cout << *dst.ptr<float>(i) << endl;
    }
}



// void decx_type_cast()
// {
//     de::InitCPUInfo();

//     de::Vector& src = de::CreateVectorRef(de::_FP32_, 1000);
//     de::Vector& dst = de::CreateVectorRef(de::_UINT8_, 1000);

//     for (int i = 0; i < src.Len(); ++i){
//         *src.ptr<float>(i) = i - 5;
//     }
    
//     de::cpu::TypeCast(src, dst, de::CVT_FP32_UINT8 | de::CVT_UINT8_SATURATED);

//     for (int i = 0; i < 10; ++i){
//         cout << (int)*dst.ptr<uint8_t>(i) << endl;
//     }

//     cout << endl;

//     for (int i = dst.Len() - 10; i < dst.Len(); ++i){
//         cout << (int)*dst.ptr<uint8_t>(i) << endl;
//     }
// }


// void decx_dewarp()
// {
//     const uint32_t dst_W = 1024;
//     const uint32_t dst_H = 1024;
//     de::InitCPUInfo();

//     de::Matrix& img = de::CreateMatrixRef();
//     // de::vis::ReadImage("./1080p_linabell_cute.jpg", img);
//     de::vis::ReadImage("./test_BGRA.png", img);

//     de::Matrix& img_u8 = de::CreateMatrixRef(de::_UINT8_, img.Width(), img.Height());

//     de::vis::ColorTransform(img, img_u8, de::vis::RGB_to_Gray);

//     de::Matrix& src = de::CreateMatrixRef(de::_FP32_, img.Width(), img.Height());
//     de::Matrix& map = de::CreateMatrixRef(de::_COMPLEX_F32_, dst_W, dst_H);

//     de::Matrix& dst = de::CreateMatrixRef(de::_FP32_, dst_W, dst_H);

//     for (int i = 0; i < img.Height(); ++i){
//         for (int j = 0; j < img.Width(); ++j){
//             *src.ptr<float>(i, j) = *img_u8.ptr<uint8_t>(i, j);
//         }
//     }

//     const float scale_fac_X = (float)img.Width() / (float)dst_W;
//     const float scale_fac_Y = (float)img.Height() / (float)dst_H;

//     cout << scale_fac_X << ", " << scale_fac_Y << endl;

//     const int32_t HH = src.Height() / 2;
//     const int32_t HW = src.Width() / 2;
//     const float radius = sqrt(HH*HH + HW * HW);

//     for (int i = 0; i < dst.Height(); ++i){
//         for (int j = 0; j < dst.Width(); ++j){
//             // map.ptr<de::CPf>(i, j)->real = j * scale_fac_X;
//             // map.ptr<de::CPf>(i, j)->image = i * scale_fac_Y;

//             // float r = sqrt((i - HH)*(i - HH) + (j - HW)*(j - HW));
//             // float fac = expf((r - radius) / 1000);
//             // // float fac1 = expf((r - radius) / 100);
//             // float new_r = fac * r;
//             // float cos_theta = cos((float)(j - HW) / r * 2);
//             // float sin_theta = sin((float)(i - HH) / r * 2);
            
//             // map.ptr<de::CPf>(i, j)->real = cos_theta * new_r + HW;
//             // map.ptr<de::CPf>(i, j)->image = sin_theta * new_r + HH;

//             // if (map.ptr<de::CPf>(i, j)->real > 400 || map.ptr<de::CPf>(i, j)->image > 400){
//             //     cout << map.ptr<de::CPf>(i, j)->real << ", " << map.ptr<de::CPf>(i, j)->image << endl;
//             // }

//             // if (map.ptr<de::CPf>(i, j)->real < 0 || map.ptr<de::CPf>(i, j)->image < 0){
//             //     cout << map.ptr<de::CPf>(i, j)->real << ", " << map.ptr<de::CPf>(i, j)->image << endl;
//             // }

//             int x = j - HW;
//             int y = i - HH;
//             float r = sqrt(x*x + y*y);
//             // float cos_theta = (float)(j) / r;
//             // float sin_theta = (float)(i) / r;
//             //cout << cos_theta << ", " << sin_theta << endl;
//             float fac = tan((r - radius) / 200);
//             // float fac = expf((radius - r) / 1000);
//             map.ptr<de::CPf>(i, j)->real = x * fac + HW;
//             map.ptr<de::CPf>(i, j)->image = y * fac + HH;

//             // if (map.ptr<de::CPf>(i, j)->real > 400 || map.ptr<de::CPf>(i, j)->image > 400){
//             //     cout << map.ptr<de::CPf>(i, j)->real << ", " << map.ptr<de::CPf>(i, j)->image << endl;
//             // }

//             // if (map.ptr<de::CPf>(i, j)->real < 0 || map.ptr<de::CPf>(i, j)->image < 0){
//             //     cout << map.ptr<de::CPf>(i, j)->real << ", " << map.ptr<de::CPf>(i, j)->image << endl;
//             // }

//             // map.ptr<de::CPf>(i, j)->real = rand() % 400;
//             // map.ptr<de::CPf>(i, j)->image = rand() % 400;
//         }
//     }
//     // de::cpu::DecxSetThreadingNum(1);
    
//     clock_t s, e;
//     s = clock();
//     for (int i = 0; i < 1; ++i)
//         de::dsp::cpu::Resample(src, map, dst, de::INTERPOLATE_BILINEAR);

//     e = clock();

//     cout << "time spent (msec) : " << (double)(e - s) / (double)CLOCKS_PER_SEC * 1 << endl;

//     de::Matrix& dst_show = de::CreateMatrixRef(de::_UINT8_, dst_W, dst_H);

//     for (int i = 0; i < dst.Height(); ++i){
//         for (int j = 0; j < dst.Width(); ++j){
//             *dst_show.ptr<uint8_t>(i, j) = *dst.ptr<float>(i, j);
//         }
//     }

//     de::vis::ShowImg(img_u8, "1");
//     de::vis::ShowImg(dst_show, "2");
//     de::vis::wait_untill_quit();

// }


void decx_color_cvt()
{
    de::InitCPUInfo();

    de::Matrix& img = de::CreateMatrixRef();
    // de::vis::ReadImage("./1080p_linabell_cute.jpg", img);
    de::vis::ReadImage("./electric_1.jpg", img);

    de::Matrix& img_u8 = de::CreateMatrixRef(de::_UCHAR4_, img.Width(), img.Height());
    de::Matrix& img_RGB = de::CreateMatrixRef(de::_UCHAR4_, img.Width(), img.Height());

    de::vis::ColorTransform(img, img_u8, de::vis::RGB_to_YUV);          // AVUY
    de::vis::ColorTransform(img_u8, img_RGB, de::vis::YUV_to_RGB);      // 0
    
    de::vis::ShowImg(img, "1");
    de::vis::ShowImg(img_u8, "2");
    de::vis::ShowImg(img_RGB, "3");
    de::vis::wait_untill_quit();
}



// void opencv_color_cvt()
// {
//     using namespace cv;
    
    
//     Mat img = imread("./electric_1.jpg");

//     Mat dst;

//     cvtColor(img, dst, COLOR_BGR2YUV);

//     // imshow("1", img);
//     // imshow("2", dst);

//     // waitKey(0);

//     // destroyAllWindows();

//     imwrite("dst.jpg", dst);
// }



void decx_dewarp_u8()
{
    const uint32_t dst_W = 1024;
    const uint32_t dst_H = 1024;
    de::InitCPUInfo();

    de::Matrix& img = de::CreateMatrixRef();
    de::vis::ReadImage("./1080p_linabell_cute.jpg", img);
    // de::vis::ReadImage("./star_cat_cute.png", img);

    de::Matrix& img_u8 = de::CreateMatrixRef(de::_UINT8_, img.Width(), img.Height());

    de::vis::ColorTransform(img, img_u8, de::vis::RGB_to_Gray);

    de::Matrix& map = de::CreateMatrixRef(de::_COMPLEX_F32_, dst_W, dst_H);

    de::Matrix& dst = de::CreateMatrixRef(de::_UINT8_, dst_W, dst_H);

    const float scale_fac_X = (float)img.Width() / (float)dst_W;
    const float scale_fac_Y = (float)img.Height() / (float)dst_H;

    cout << scale_fac_X << ", " << scale_fac_Y << endl;

    const int32_t HH = img.Height() / 2;
    const int32_t HW = img.Width() / 2;
    const float radius = sqrt(HH*HH + HW * HW);

    for (int i = 0; i < dst.Height(); ++i){
        for (int j = 0; j < dst.Width(); ++j){
            // map.ptr<de::CPf>(i, j)->real = j * scale_fac_X;
            // map.ptr<de::CPf>(i, j)->image = i * scale_fac_Y;

            // float r = sqrt((i - HH)*(i - HH) + (j - HW)*(j - HW));
            // float fac = expf((r - radius) / 1000);
            // // float fac1 = expf((r - radius) / 100);
            // float new_r = fac * r;
            // float cos_theta = cos((float)(j - HW) / r * 2);
            // float sin_theta = sin((float)(i - HH) / r * 2);
            
            // map.ptr<de::CPf>(i, j)->real = cos_theta * new_r + HW;
            // map.ptr<de::CPf>(i, j)->image = sin_theta * new_r + HH;

            // if (map.ptr<de::CPf>(i, j)->real > 400 || map.ptr<de::CPf>(i, j)->image > 400){
            //     cout << map.ptr<de::CPf>(i, j)->real << ", " << map.ptr<de::CPf>(i, j)->image << endl;
            // }

            // if (map.ptr<de::CPf>(i, j)->real < 0 || map.ptr<de::CPf>(i, j)->image < 0){
            //     cout << map.ptr<de::CPf>(i, j)->real << ", " << map.ptr<de::CPf>(i, j)->image << endl;
            // }

            int x = j - HW;
            int y = i - HH;
            float r = sqrt(x*x + y*y);
            // float cos_theta = (float)(j) / r;
            // float sin_theta = (float)(i) / r;
            //cout << cos_theta << ", " << sin_theta << endl;
            float fac = tan((r - radius) / 200);
            // float fac = expf((radius - r) / 1000);
            map.ptr<de::CPf>(i, j)->real = x * fac + HW;
            map.ptr<de::CPf>(i, j)->image = y * fac + HH;

            // if (map.ptr<de::CPf>(i, j)->real > 400 || map.ptr<de::CPf>(i, j)->image > 400){
            //     cout << map.ptr<de::CPf>(i, j)->real << ", " << map.ptr<de::CPf>(i, j)->image << endl;
            // }

            // if (map.ptr<de::CPf>(i, j)->real < 0 || map.ptr<de::CPf>(i, j)->image < 0){
            //     cout << map.ptr<de::CPf>(i, j)->real << ", " << map.ptr<de::CPf>(i, j)->image << endl;
            // }

            // map.ptr<de::CPf>(i, j)->real = rand() % 400;
            // map.ptr<de::CPf>(i, j)->image = rand() % 400;
        }
    }
    // de::cpu::DecxSetThreadingNum(1);
    
    clock_t s, e;
    s = clock();
    for (int i = 0; i < 2000; ++i)
        de::dsp::cpu::Resample(img_u8, map, dst, de::INTERPOLATE_BILINEAR);

    e = clock();

    cout << "time spent (msec) : " << (double)(e - s) / (double)CLOCKS_PER_SEC * 1 / 2.0 << endl;

    de::vis::ShowImg(img_u8, "1");
    de::vis::ShowImg(dst, "2");
    de::vis::wait_untill_quit();

}



void decx_cuda_dewarp()
{
    const uint32_t dst_W = 1024;
    const uint32_t dst_H = 1024;
    de::InitCPUInfo();
    de::InitCuda();

    de::Matrix& img = de::CreateMatrixRef();
    // de::vis::ReadImage("./1080p_linabell_cute.jpg", img);
    de::vis::ReadImage("./star_cat_cute.png", img);

    de::Matrix& img_u8 = de::CreateMatrixRef(de::_UINT8_, img.Width(), img.Height());

    de::vis::ColorTransform(img, img_u8, de::vis::RGB_to_Gray);

    de::Matrix& src = de::CreateMatrixRef(de::_FP32_, img.Width(), img.Height());
    de::GPU_Matrix& Dsrc = de::CreateGPUMatrixRef(de::_FP32_, img.Width(), img.Height());
    de::Matrix& map = de::CreateMatrixRef(de::_COMPLEX_F32_, dst_W, dst_H);
    de::GPU_Matrix& Dmap = de::CreateGPUMatrixRef(de::_COMPLEX_F32_, dst_W, dst_H);

    de::Matrix& dst = de::CreateMatrixRef(de::_FP32_, dst_W, dst_H);
    de::GPU_Matrix& Ddst = de::CreateGPUMatrixRef(de::_FP32_, dst_W, dst_H);

    for (int i = 0; i < img.Height(); ++i){
        for (int j = 0; j < img.Width(); ++j){
            *src.ptr<float>(i, j) = *img_u8.ptr<uint8_t>(i, j);
        }
    }

    const float scale_fac_X = (float)img.Width() / (float)dst_W;
    const float scale_fac_Y = (float)img.Height() / (float)dst_H;

    cout << scale_fac_X << ", " << scale_fac_Y << endl;

    const int32_t HH = src.Height() / 2;
    const int32_t HW = src.Width() / 2;
    const float radius = sqrt(HH*HH + HW * HW);

    for (int i = 0; i < dst.Height(); ++i){
        for (int j = 0; j < dst.Width(); ++j){
            // map.ptr<de::CPf>(i, j)->real = j * scale_fac_X;
            // map.ptr<de::CPf>(i, j)->image = i * scale_fac_Y;

            int x = j - HW;
            int y = i - HH;
            float r = sqrt(x*x + y*y);
            // float cos_theta = (float)(j) / r;
            // float sin_theta = (float)(i) / r;
            //cout << cos_theta << ", " << sin_theta << endl;
            // float fac = atan((r - radius) / 300);
            float fac = expf((r - radius) / 400);
            map.ptr<de::CPf>(i, j)->real = x * fac + HW;
            map.ptr<de::CPf>(i, j)->image = y * fac + HH;
        }
    }

    de::Memcpy(src, Dsrc, {0, 0}, {0, 0}, {src.Width(), src.Height()}, de::DECX_MEMCPY_H2D);
    de::Memcpy(map, Dmap, {0, 0}, {0, 0}, {map.Width(), map.Height()}, de::DECX_MEMCPY_H2D);

    // de::cpu::DecxSetThreadingNum(1);
    clock_t s, e;
    s = clock();
    for (int i = 0; i < 1; ++i)
        de::dsp::cuda::Resample(Dsrc, Dmap, Ddst, de::INTERPOLATE_BILINEAR);

    e = clock();

    cout << "time spent (msec) : " << (double)(e - s) / (double)CLOCKS_PER_SEC * 1000 << endl;
    
    de::DH handle = de::Memcpy(dst, Ddst, {0, 0}, {0, 0}, {dst.Width(), dst.Height()}, de::DECX_MEMCPY_D2H);
    
    de::Matrix& dst_show = de::CreateMatrixRef(de::_UINT8_, dst_W, dst_H);

    for (int i = 0; i < dst.Height(); ++i){
        for (int j = 0; j < dst.Width(); ++j){
            *dst_show.ptr<uint8_t>(i, j) = *dst.ptr<float>(i, j);
        }
    }

    de::vis::ShowImg(img_u8, "1");
    de::vis::ShowImg(dst_show, "2");
    de::vis::wait_untill_quit();

}



void decx_cuda_dewarp_u8()
{
    const uint32_t dst_W = 625;
    const uint32_t dst_H = 625;
    de::InitCPUInfo();
    de::InitCuda();

    de::Matrix& img = de::CreateMatrixRef();
    de::vis::ReadImage("./1080p_linabell_cute.jpg", img);
    // de::vis::ReadImage("./electric_1.jpg", img);
    // de::vis::ReadImage("./XM.jpg", img);

    // de::Matrix& img_u8 = de::CreateMatrixRef(de::_UINT8_, img.Width(), img.Height());

    // de::vis::ColorTransform(img, img_u8, de::vis::RGB_to_Gray);

    de::GPU_Matrix& Dsrc = de::CreateGPUMatrixRef(de::_UCHAR4_, img.Width(), img.Height());
    de::Matrix& map = de::CreateMatrixRef(de::_COMPLEX_F32_, dst_W, dst_H);
    de::GPU_Matrix& Dmap = de::CreateGPUMatrixRef(de::_COMPLEX_F32_, dst_W, dst_H);

    de::Matrix& dst = de::CreateMatrixRef(de::_UCHAR4_, dst_W, dst_H);
    de::GPU_Matrix& Ddst = de::CreateGPUMatrixRef(de::_UCHAR4_, dst_W, dst_H);

    const float scale_fac_X = (float)img.Width() / (float)dst_W;
    const float scale_fac_Y = (float)img.Height() / (float)dst_H;

    cout << scale_fac_X << ", " << scale_fac_Y << endl;

    const int32_t HH = dst.Height() / 2;
    const int32_t HW = dst.Width() / 2;
    const float radius = sqrt(HH * HH + HW * HW) * sqrt(scale_fac_X * scale_fac_X + scale_fac_Y * scale_fac_Y);

    for (int i = 0; i < dst.Height(); ++i){
        for (int j = 0; j < dst.Width(); ++j){
            // map.ptr<de::CPf>(i, j)->real = j * scale_fac_X;
            // map.ptr<de::CPf>(i, j)->image = i * scale_fac_Y;

            int x = (j - HW) * scale_fac_Y;
            int y = (i - HH) * scale_fac_Y;
            float r = sqrt(x*x + y*y);
            // float cos_theta = (float)(j) / r;
            // float sin_theta = (float)(i) / r;
            //cout << cos_theta << ", " << sin_theta << endl;
            float fac = expf((r - radius) / 700);
            // float fac = acos((r - radius) / 20);
            map.ptr<de::CPf>(i, j)->real = (x * fac + HW) * scale_fac_X;
            map.ptr<de::CPf>(i, j)->image = (y * fac + HH) * scale_fac_Y;
        }
    }

    de::Memcpy(img, Dsrc, {0, 0}, {0, 0}, {img.Width(), img.Height()}, de::DECX_MEMCPY_H2D);
    de::Memcpy(map, Dmap, {0, 0}, {0, 0}, {map.Width(), map.Height()}, de::DECX_MEMCPY_H2D);

    // de::cpu::DecxSetThreadingNum(1);
    clock_t s, e;
    s = clock();
    for (int i = 0; i < 2000; ++i)
        de::dsp::cuda::Resample(Dsrc, Dmap, Ddst, de::INTERPOLATE_BILINEAR);

    e = clock();

    cout << "time spent (msec) : " << (double)(e - s) / (double)CLOCKS_PER_SEC * 500 << endl;
    
    de::DH handle = de::Memcpy(dst, Ddst, {0, 0}, {0, 0}, {dst.Width(), dst.Height()}, de::DECX_MEMCPY_D2H);
    
    de::vis::ShowImg(img, "1");
    de::vis::ShowImg(dst, "2");
    de::vis::wait_untill_quit();

}


void decx_Eigenvalue()
{
    de::InitCPUInfo();

    de::Matrix& src = de::CreateMatrixRef(de::_FP32_, 1000, 1000);

    for (int i = 0; i < src.Width(); ++i){
        *src.ptr<float>(i, i) = i;
    }
    for (int i = 0; i < src.Width() - 1; ++i){
        *src.ptr<float>(i, i + 1) = src.Width() - 1 - i;
        *src.ptr<float>(i + 1, i) = src.Width() - 1 - i;
    }

    // for (int i = 0; i < src.Width(); ++i){
    //     *src.ptr<float>(i, i) = (rand() % 10) - 15;
    // }
    // for (int i = 0; i < src.Width() - 1; ++i){
    //     *src.ptr<float>(i, i + 1) = (rand() % 20) - 10;
    //     *src.ptr<float>(i + 1, i) = (rand() % 20) - 10;
    // }

    // for (int i = 0; i < src.Height(); ++i){
    //     for (int j = 0; j < src.Width(); ++j){
    //         cout << setw(2) << *src.ptr<float>(i, j);
    //     }
    //     cout << endl;
    // }

    de::cpu::DecxSetThreadingNum(12);
    float* a = NULL, *b = NULL;
    de::blas::cpu::Eigenvalue(src, &a, &b);

    // if (a){
    //     for (int i = 0; i < src.Width(); ++i){
    //         cout << a[i] << ", " << b[i] << endl;
    //     }
    // }
}


void decx_generate()
{
    de::InitCPUInfo();
    
    de::cpu::DecxSetThreadingNum(12);
    de::Matrix& src = de::CreateMatrixRef(de::_FP32_, 100, 100);

    de::Number val = 37.3f;
    // de::cpu::Generate(src, val);
    de::cpu::Random(src, de::_INT32_, 100, 100, 0, de::Point2D_d(0, 10));

    for (int i = 90; i < 100; ++i){
        for (int j = 90; j < 104; ++j){
            cout << setw(12) << *src.ptr<int32_t>(i, j);
        }
        cout << endl;
    }

    // cout << "rand() : " << rand() << endl;
}

// void decx_cuda_extend()
// {
//     constexpr uint32_t extL = 10, extR = 10, extT = 10, extB = 10;
//     de::InitCPUInfo();
//     de::InitCuda();

//     de::Matrix& src = de::CreateMatrixRef(de::_FP32_, 1000, 1000);
//     de::GPU_Matrix& Dsrc = de::CreateGPUMatrixRef(de::_FP32_, 1000, 1000);
//     de::GPU_Matrix& Ddst = de::CreateGPUMatrixRef(de::_FP32_, 1000 + extL + extR, 1000 + extT + extB);
//     de::Matrix& dst = de::CreateMatrixRef(de::_FP32_, 1000 + extL + extR, 1000 + extT + extB);

//     for (int i = 0; i < src.Height(); ++i){
//         for (int j = 0; j < src.Width(); ++j){
//             *src.ptr<float>(i, j) = j;
//         }
//     }

//     de::Memcpy(src, Dsrc, {0, 0}, {0, 0}, {src.Width(), src.Height()}, de::DECX_MEMCPY_H2D);
//     clock_t s, e;
//     s = clock();
//     de::blas::cuda::Extend(Dsrc, Ddst, extL, extR, extT, extB);
//     e = clock();
    
//     cout << "time spent : " << (e - s) / (float)CLOCKS_PER_SEC * 1000 << endl;

//     de::Memcpy(dst, Ddst, {0, 0}, {0, 0}, {dst.Width(), dst.Height()}, de::DECX_MEMCPY_D2H);

//     for (int i = 9; i < 11; ++i){
//         for (int j = 0; j < 30; ++j){
//             cout << setw(3) << *dst.ptr<float>(i, j);
//         }
//         cout << endl;
//     }
// }
#endif

#define _6x6_test_ 0
void decx_GQR()
{
    de::InitCPUInfo();

    de::Matrix& src = de::CreateMatrixRef(de::_FP32_, 1024, 1024);
    de::Matrix& Q = de::CreateMatrixRef(de::_FP32_, 1024, 1024);
    de::Matrix& R = de::CreateMatrixRef(de::_FP32_, 1024, 1024);

#if _6x6_test_
    float source[6 * 6] = { 10, 20, 30, 40, 50, 60,
                         32, 32, 44, 55, 66, 35,
                         23, 66, 74, 64, 45, 65,
                         67, 28, 46, 26, 46, 42,
                         95, 95, 52, 88, 65, 11,
                         75, 53, 96, 47, 32, 32 };

    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 6; ++j) {
            *src.ptr<float>(i, j) = source[i * 6 + j];
        }
    }
#else
    for (int i = 0; i < src.Height(); ++i) {
        for (int j = 0; j < src.Width(); ++j) {
            *src.ptr<float>(i, j) = i + j + 2;
        }
    }
#endif
    clock_t s, e;
    s = clock();
    for (int i = 0; i < 1; ++i)
        de::blas::cpu::GQRF(src, Q, R);
    e = clock();
    cout << "time spent : " << (e - s) / (float)CLOCKS_PER_SEC * 1000 << "msec" << endl;
}

void decx_DP1D()
{
    de::InitCPUInfo();
    de::InitCuda();

    // de::Matrix& A = de::CreateMatrixRef(de::_FP32_, 1024, 1024);
    de::Vector& A = de::CreateVectorRef(de::_FP32_, 1017);
    de::Vector& B = de::CreateVectorRef(de::_FP32_, 1017);

    float res_ref = 0;
    for (int i = 0; i < A.Len(); ++i){
        *A.ptr<float>(i) = 0.1 * (float)i;
        *B.ptr<float>(i) = 0.1 * (float)i;
        res_ref += (0.1 * (float)i * 0.1 * (float)i);
    }

    de::Number res;

    de::blas::cuda::Dot_product(A, B, res, 0);
    cout << res.get_data<float>() << endl;
    cout << "res_ref : " << res_ref << endl;
}

void decx_DP2D()
{
    de::InitCPUInfo();
    de::InitCuda();

    de::Matrix& A = de::CreateMatrixRef(de::_FP32_, 1113, 1017);
    de::Matrix& B = de::CreateMatrixRef(de::_FP32_, 1113, 1017);
    de::REDUCE_METHOD rd_method = de::_REDUCE2D_V_;
    de::Vector& dst = de::CreateVectorRef(de::_FP32_, A.Height());
    int dp_l = 0;
    if (rd_method == de::_REDUCE2D_H_) {
        dst.release();
        dst = de::CreateVectorRef(de::_FP32_, A.Height());
        dp_l = A.Width();
    }
    else {
        dst.release();
        dst = de::CreateVectorRef(de::_FP32_, A.Width());
        dp_l = A.Height();
    }

    float res_ref = 0;
    for (int i = 0; i < A.Height(); ++i){
        for (int j = 0; j < A.Width(); ++j){
            *A.ptr<float>(i, j) = i;
            *B.ptr<float>(i, j) = i;
        }
    }

    float ref_res = 0;
    for (int i = 0; i < dp_l; ++i){
        ref_res += i * i;
    }
    
    de::blas::cuda::Dot_product(A, B, dst, rd_method, 0);

    for (int i = dst.Len() - 10; i < dst.Len(); ++i){
        cout << (int)(*dst.ptr<float>(i)) << endl;
    }
    cout << "ref_res : " << (int)ref_res << endl;
}

int main()
{
    // decx_GEMM_cuda_fp32();
    // decx_transpose_cuda_fp16();
    // decx_transpose_cpu_fp16();
    // decx_GEMM_cuda_fp16();

    // decx_GEMM_cpu();
    // frag_manager fmgr;
    // frag_manager_gen_Nx(&fmgr, 1013, 4, 16);
    // printf("W : tot: %d, frag_len: %d, frag_num: %d, last: %d\n", 
    //     fmgr.total, fmgr.frag_len, fmgr.frag_num, fmgr.last_frag_len);

    // decx_GEMM_cuda_fp64();
    // decx_GEMM_cuda_cplxf();
    // decx_GEMM_cuda_cplxd();
    // DECX_FFT2D();
    // DECX_filter2D();
    //DECX_filter2D_cpu();
    // decx_type_cast();
    // decx_type_cast_vec();
    // decx_vec_arithmetic();
    // decx_mat_arithmetic();
    // decx_GQR();
    // decx_mat_arithmetic_cuda();
    // decx_dewarp();
    // decx_color_cvt();
    // opencv_color_cvt();
    // decx_dewarp_u8();
    // decx_cuda_dewarp();
    // decx_cuda_dewarp_u8();
    // decx_cuda_extend();
    // decx_generate();
    // decx_Eigenvalue();

    // decx_DP1D();
    // decx_DP2D();
    // DECX_FFT1D();
    DECX_FFT2D();
    
    // //__m256d src = _mm256_setr_pd(1.0, 2.0, 3.0, 4.0);
    // __m256d src;
    // for (int i = 0; i < 4; ++i){
    //     ((double*)&src)[i] = i * 3.1415926;
    // }
    // // // cout << ((double*)(&src))[0] << endl;
    // __m256d dst = _avx_cos_fp64x4(src);
    // // __m128d ref;
    
    // for (int i = 0; i < 4; ++i){
    //     cout << "my_res : " << ((double*)&dst)[i] << 
    //         "; ref_res : " << cos(((double*)&src)[i]) << endl;
    // }

    return 0;
}
