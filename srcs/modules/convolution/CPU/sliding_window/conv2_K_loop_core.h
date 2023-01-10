///**
//*   ---------------------------------------------------------------------
//*   Author : Wayne Anderson
//*   Date   : 2021.04.16
//*   ---------------------------------------------------------------------
//*   This is a part of the open source program named "DECX", copyright c Wayne,
//*   2021.04.16, all right reserved.
//*   More information please visit https://github.com/param0037/backup_1
//*/
//
//#ifndef _CONV2_K_LOOP_CORE_H_
//#define _CONV2_K_LOOP_CORE_H_
//
//
//#include "../../../core/thread_management/thread_pool.h"
//#include "../../../core/thread_management/thread_arrange.h"
//
//
//namespace decx
//{
//    namespace conv {
//        namespace CPUK {
//            // *** ATTENTION *** ! -> In this model, kernel should be stored linearly (pitch = width)
//            /*
//            * In this model, we only pay attention to the width of kernel, regardless its height
//            * Since only the kernel width affects the behaviours during loading data from src matrix
//            */
//
//
//            /*
//            * @param Wsrc : width of src matrix, in float
//            * @param Wdst : width of dst matrix, in float
//            */
//            static _THREAD_CALL_
//                __m256 _conv2_fp32_loop_in_kernel_Nregs(const float* src, const float* kernel, const uint2  ker_dims,
//                    const ushort reg_WL, const size_t Wsrc, const uint _loop);
//
//
//            /*
//            * @param Wsrc : width of src matrix, in float
//            * @param Wdst : width of dst matrix, in float
//            * @param reg_WL : ( = ker_dims.x - 1 + 8 - 8 )
//            */
//            static _THREAD_CALL_
//                __m256 _conv2_fp32_loop_in_kernel_2regs(const float* src, const float* kernel, const uint2  ker_dims, const size_t Wsrc);
//        }
//    }
//}
//
//
//
//#define _CONV2_REGS_FP32_SHIFT_LOAD_GEN_(dex){                                                                              \
//    k_value = kernel[ker_dex];                                                                                              \
//    _proc_reg = _mm256_permute_ps(_proc_reg, _MM_SHUFFLE(0, 3, 2, 1));                                                      \
//    tmp2 = _mm256_permute2f128_ps(_proc_reg, _mm256_permutevar8x32_ps(_store_reg, _mm256_set1_epi32(dex)), 0b00100001);     \
//    _proc_reg = _mm256_blend_ps(_proc_reg, tmp2, 0b10001000);                                                               \
//    _accumulator = _mm256_fmadd_ps(_proc_reg, _mm256_set1_ps(k_value), _accumulator);                                       \
//    ++ker_dex;                                                                                                              \
//}
//
//
//
//#define _CONV2_REGS_FP32_SHIFT_LOAD_0 {                                                                             \
//    k_value = kernel[ker_dex];                                                                                      \
//    _proc_reg = _mm256_permute_ps(_proc_reg, _MM_SHUFFLE(0, 3, 2, 1));                                              \
//    tmp2 = _mm256_permute2f128_ps(_proc_reg, _mm256_permute_ps(_store_reg, _MM_SHUFFLE(0, 3, 2, 1)), 0b00100001);   \
//    _proc_reg = _mm256_blend_ps(_proc_reg, tmp2, 0b10001000);                                                       \
//    _accumulator = _mm256_fmadd_ps(_proc_reg, _mm256_set1_ps(k_value), _accumulator);                               \
//    ++ker_dex;                                                                                                      \
//}
//
//
//#define _CONV2_REGS_FP32_SHIFT_LOAD_1 {                                                                             \
//    k_value = kernel[ker_dex];                                                                                      \
//    _proc_reg = _mm256_permute_ps(_proc_reg, _MM_SHUFFLE(0, 3, 2, 1));                                              \
//    tmp2 = _mm256_permute2f128_ps(_proc_reg, _mm256_permute_ps(_store_reg, _MM_SHUFFLE(1, 2, 3, 0)), 0b00100001);   \
//    _proc_reg = _mm256_blend_ps(_proc_reg, tmp2, 0b10001000);                                                       \
//    _accumulator = _mm256_fmadd_ps(_proc_reg, _mm256_set1_ps(k_value), _accumulator);                               \
//    ++ker_dex;                                                                                                      \
//}
//
//
//#define _CONV2_REGS_FP32_SHIFT_LOAD_2 {                                                                             \
//    k_value = kernel[ker_dex];                                                                                      \
//    _proc_reg = _mm256_permute_ps(_proc_reg, _MM_SHUFFLE(0, 3, 2, 1));                                              \
//    tmp2 = _mm256_permute2f128_ps(_proc_reg, _mm256_permute_ps(_store_reg, _MM_SHUFFLE(2, 3, 1, 0)), 0b00100001);   \
//    _proc_reg = _mm256_blend_ps(_proc_reg, tmp2, 0b10001000);                                                       \
//    _accumulator = _mm256_fmadd_ps(_proc_reg, _mm256_set1_ps(k_value), _accumulator);                               \
//    ++ker_dex;                                                                                                      \
//}
//
//
//
//#define _CONV2_REGS_FP32_SHIFT_LOAD_3 {                                                                  \
//    k_value = kernel[ker_dex];                                                                           \
//    _proc_reg = _mm256_permute_ps(_proc_reg, _MM_SHUFFLE(0, 3, 2, 1));                                   \
//    tmp2 = _mm256_permute2f128_ps(_proc_reg, _store_reg, 0b00100001);                                    \
//    _proc_reg = _mm256_blend_ps(_proc_reg, tmp2, 0b10001000);                                            \
//    _accumulator = _mm256_fmadd_ps(_proc_reg, _mm256_set1_ps(k_value), _accumulator);                    \
//    ++ker_dex;                                                                                           \
//}
//
//
//
//#define _CONV2_REGS_FP32_SHIFT_LOAD_4 {                                                                             \
//    k_value = kernel[ker_dex];                                                                                      \
//    _proc_reg = _mm256_permute_ps(_proc_reg, _MM_SHUFFLE(0, 3, 2, 1));                                              \
//    tmp2 = _mm256_permute2f128_ps(_proc_reg, _mm256_permute_ps(_store_reg, _MM_SHUFFLE(0, 3, 2, 1)), 0b00110001);   \
//    _proc_reg = _mm256_blend_ps(_proc_reg, tmp2, 0b10001000);                                                       \
//    _accumulator = _mm256_fmadd_ps(_proc_reg, _mm256_set1_ps(k_value), _accumulator);                               \
//    ++ker_dex;                                                                                                      \
//}
//
//
//
//#define _CONV2_REGS_FP32_SHIFT_LOAD_5 {                                                                             \
//    k_value = kernel[ker_dex];                                                                                      \
//    _proc_reg = _mm256_permute_ps(_proc_reg, _MM_SHUFFLE(0, 3, 2, 1));                                              \
//    tmp2 = _mm256_permute2f128_ps(_proc_reg, _mm256_permute_ps(_store_reg, _MM_SHUFFLE(1, 2, 3, 0)), 0b00110001);   \
//    _proc_reg = _mm256_blend_ps(_proc_reg, tmp2, 0b10001000);                                                       \
//    _accumulator = _mm256_fmadd_ps(_proc_reg, _mm256_set1_ps(k_value), _accumulator);                               \
//    ++ker_dex;                                                                                                      \
//}
//
//
//
//#define _CONV2_REGS_FP32_SHIFT_LOAD_6 {                                                                             \
//    k_value = kernel[ker_dex];                                                                                      \
//    _proc_reg = _mm256_permute_ps(_proc_reg, _MM_SHUFFLE(0, 3, 2, 1));                                              \
//    tmp2 = _mm256_permute2f128_ps(_proc_reg, _mm256_permute_ps(_store_reg, _MM_SHUFFLE(2, 3, 1, 0)), 0b00110001);   \
//    _proc_reg = _mm256_blend_ps(_proc_reg, tmp2, 0b10001000);                                                       \
//    _accumulator = _mm256_fmadd_ps(_proc_reg, _mm256_set1_ps(k_value), _accumulator);                               \
//    ++ker_dex;                                                                                                      \
//}
//
//
//
//#define _CONV2_REGS_FP32_SHIFT_LOAD_7 {                                                                  \
//    k_value = kernel[ker_dex];                                                                           \
//    _proc_reg = _mm256_permute_ps(_proc_reg, _MM_SHUFFLE(0, 3, 2, 1));                                   \
//    tmp2 = _mm256_permute2f128_ps(_proc_reg, _store_reg, 0b00110001);                                    \
//    _proc_reg = _mm256_blend_ps(_proc_reg, tmp2, 0b10001000);                                            \
//    _accumulator = _mm256_fmadd_ps(_proc_reg, _mm256_set1_ps(k_value), _accumulator);                    \
//    ++ker_dex;                                                                                           \
//}
//
//
//
//
//_THREAD_CALL_
//__m256 decx::conv::CPUK::_conv2_fp32_loop_in_kernel_Nregs(const float* __restrict     src, 
//                                              const float* __restrict     kernel, 
//                                              const uint2           ker_dims, 
//                                              const ushort          reg_WL, 
//                                              const size_t          Wsrc,
//                                              const uint            _loop)
//{
//    register __m256 _proc_reg, _store_reg, tmp2,
//        _accumulator = _mm256_set1_ps(0);
//
//    float k_value;      // kernel value
//    uint ker_dex = 0;
//
//    for (int i = 0; i < ker_dims.y; ++i) 
//    {
//        _proc_reg = _mm256_load_ps(src + i * (Wsrc << 3));
//        _store_reg = _mm256_load_ps(src + i * (Wsrc << 3) + 8);
//        k_value = kernel[ker_dex];
//        // first multiply-add with the first element of kernel on every row
//        _accumulator = _mm256_fmadd_ps(_proc_reg, _mm256_set1_ps(k_value), _accumulator);
//        ++ker_dex;
//
//        _CONV2_REGS_FP32_SHIFT_LOAD_0;         _CONV2_REGS_FP32_SHIFT_LOAD_1;
//        _CONV2_REGS_FP32_SHIFT_LOAD_2;         _CONV2_REGS_FP32_SHIFT_LOAD_3;
//        _CONV2_REGS_FP32_SHIFT_LOAD_4;           _CONV2_REGS_FP32_SHIFT_LOAD_5;
//        _CONV2_REGS_FP32_SHIFT_LOAD_6;           _CONV2_REGS_FP32_SHIFT_LOAD_7;
//
//        for (uint _L = 0; _L < _loop; ++_L) {
//            _store_reg = _mm256_load_ps(src + i * (Wsrc << 3) + 16 + (_L << 3));
//            _CONV2_REGS_FP32_SHIFT_LOAD_0;         _CONV2_REGS_FP32_SHIFT_LOAD_1;
//            _CONV2_REGS_FP32_SHIFT_LOAD_2;         _CONV2_REGS_FP32_SHIFT_LOAD_3;
//            _CONV2_REGS_FP32_SHIFT_LOAD_4;           _CONV2_REGS_FP32_SHIFT_LOAD_5;
//            _CONV2_REGS_FP32_SHIFT_LOAD_6;           _CONV2_REGS_FP32_SHIFT_LOAD_7;
//        }
//        _store_reg = _mm256_load_ps(src + i * (Wsrc << 3) + 16 + (_loop << 3));
//
//        for (int j = 0; j < reg_WL; ++j) {
//            _CONV2_REGS_FP32_SHIFT_LOAD_GEN_(j);
//        }
//    }
//    return _accumulator;
//}
//
//
//
//_THREAD_CALL_
//__m256 decx::conv::CPUK::_conv2_fp32_loop_in_kernel_2regs(const float* __restrict     src, 
//                                              const float* __restrict     kernel, 
//                                              const uint2           ker_dims, 
//                                              const size_t          Wsrc)
//{
//    register __m256 _proc_reg, _store_reg, tmp2,
//        _accumulator = _mm256_set1_ps(0);
//
//    float k_value;      // kernel value
//    uint ker_dex = 0;
//
//    for (int i = 0; i < ker_dims.y; ++i) 
//    {
//        _proc_reg = _mm256_load_ps(src + i * (Wsrc << 3));
//        _store_reg = _mm256_load_ps(src + i * (Wsrc << 3) + 8);
//        k_value = kernel[ker_dex];
//        // first multiply-add with the first element of kernel on every row
//        _accumulator = _mm256_fmadd_ps(_proc_reg, _mm256_set1_ps(k_value), _accumulator);
//        ++ker_dex;
//
//        for (int _internal_dex = 0; _internal_dex < ker_dims.x - 1; ++_internal_dex)
//        {
//            k_value = kernel[ker_dex];
//            switch (_internal_dex)
//            {
//            case 7:
//                _accumulator = _mm256_fmadd_ps(_store_reg, _mm256_set1_ps(k_value), _accumulator);
//                break;
//
//            default:
//                _proc_reg = _mm256_permute_ps(_proc_reg, _MM_SHUFFLE(0, 3, 2, 1));
//                tmp2 = _mm256_permute2f128_ps(_proc_reg, _mm256_permutevar8x32_ps(_store_reg, _mm256_set1_epi32(_internal_dex)), 0b00100001);
//                _proc_reg = _mm256_blend_ps(_proc_reg, tmp2, 0b10001000);
//                _accumulator = _mm256_fmadd_ps(_proc_reg, _mm256_set1_ps(k_value), _accumulator);
//                break;
//            }
//            
//            ++ker_dex;
//        }
//    }
//    return _accumulator;
//}
//
//
//
//
//namespace decx
//{
//    namespace conv {
//        namespace CPUK {
//            // *** ATTENTION *** ! -> In this model, kernel should be stored linearly (pitch = width)
//            /*
//            * In this model, we only pay attention to the width of kernel, regardless its height
//            * Since only the kernel width affects the behaviours during loading data from src matrix
//            */
//
//            /**
//            * radius = 3
//            * @param Wsrc : width of src matrix, in float
//            * @param Wdst : width of dst matrix, in float
//            */
//            static _THREAD_CALL_
//                __m256d _conv2_fp64_loop_in_kernel_2regs_rw2(const double* src, const double* kernel, const uint2 ker_dims, const size_t Wsrc);
//
//
//            /**
//            * radius = 4
//            * @param Wsrc : width of src matrix, in float
//            * @param Wdst : width of dst matrix, in float
//            */
//            static _THREAD_CALL_
//                __m256d _conv2_fp64_loop_in_kernel_2regs_rw4(const double* src, const double* kernel, const uint2 ker_dims, const size_t Wsrc);
//
//
//            /**
//            * radius > 6, reg_WL = 2
//            * @param Wsrc : width of src matrix, in float
//            * @param Wdst : width of dst matrix, in float
//            */
//            static _THREAD_CALL_
//                __m256d _conv2_fp64_loop_in_kernel_Nregs_rw2(const double* src, const double* kernel, const uint2 ker_dims,
//                    const size_t Wsrc, const uint _loop);
//
//
//            /**
//            * radius > 6, reg_WL = 2
//            * @param Wsrc : width of src matrix, in float
//            * @param Wdst : width of dst matrix, in float
//            */
//            static _THREAD_CALL_
//                __m256d _conv2_fp64_loop_in_kernel_Nregs_rw4(const double* src, const double* kernel, const uint2 ker_dims,
//                    const size_t Wsrc, const uint _loop);
//        }
//    }
//}
//
//
//
//#define _CONV2_REGS_FP64_SHIFT_LOAD_3 {       \
//    k_value = kernel[ker_dex];              \
//    _proc_reg = decx::utils::_mm256d_shift1_H2L(_proc_reg);     \
//    _proc_reg = _mm256_blend_pd(_proc_reg, _store_reg, 0b1000);        \
//    _accumulator = _mm256_fmadd_pd(_proc_reg, _mm256_set1_pd(k_value), _accumulator);                               \
//    ++ker_dex;      \
//}
//
//
//#define _CONV2_REGS_FP64_SHIFT_LOAD_2 {       \
//    k_value = kernel[ker_dex];              \
//    _proc_reg = decx::utils::_mm256d_shift1_H2L(_proc_reg);     \
//    _proc_reg = _mm256_blend_pd(_proc_reg, _mm256_permute_pd(_store_reg, 0b0101), 0b1000);        \
//    _accumulator = _mm256_fmadd_pd(_proc_reg, _mm256_set1_pd(k_value), _accumulator);                               \
//    ++ker_dex;      \
//}
//
//
//#define _CONV2_REGS_FP64_SHIFT_LOAD_0 {       \
//    k_value = kernel[ker_dex];              \
//    _proc_reg = decx::utils::_mm256d_shift1_H2L(_proc_reg);     \
//    _proc_reg = _mm256_blend_pd(_proc_reg, _mm256_permute_pd(_mm256_permute2f128_pd(_store_reg, _store_reg, 1), 0b0101), 0b1000);        \
//    _accumulator = _mm256_fmadd_pd(_proc_reg, _mm256_set1_pd(k_value), _accumulator);                               \
//    ++ker_dex;      \
//}
//
//
//#define _CONV2_REGS_FP64_SHIFT_LOAD_1 {       \
//    k_value = kernel[ker_dex];              \
//    _proc_reg = decx::utils::_mm256d_shift1_H2L(_proc_reg);     \
//    _proc_reg = _mm256_blend_pd(_proc_reg, _mm256_permute2f128_pd(_store_reg, _store_reg, 1), 0b1000);        \
//    _accumulator = _mm256_fmadd_pd(_proc_reg, _mm256_set1_pd(k_value), _accumulator);                               \
//    ++ker_dex;      \
//}
//
//
//
//
//_THREAD_CALL_
//__m256d decx::conv::CPUK::_conv2_fp64_loop_in_kernel_2regs_rw2(const double* src, const double* kernel, const uint2 ker_dims, const size_t Wsrc)
//{
//    __m256d _proc_reg, _store_reg,
//        _accumulator = _mm256_set1_pd(0);
//
//    float k_value;      // kernel value
//    uint ker_dex = 0;
//
//    for (int i = 0; i < ker_dims.y; ++i)
//    {
//        _proc_reg = _mm256_load_pd(src + i * (Wsrc << 2));
//        _store_reg = _mm256_load_pd(src + i * (Wsrc << 2) + 4);
//        k_value = kernel[ker_dex];
//        // first multiply-add with the first element of kernel on every row
//        _accumulator = _mm256_fmadd_pd(_proc_reg, _mm256_set1_pd(k_value), _accumulator);
//        ++ker_dex;
//        _CONV2_REGS_FP64_SHIFT_LOAD_0;         _CONV2_REGS_FP64_SHIFT_LOAD_1;
//    }
//    return _accumulator;
//}
//
//
//
//_THREAD_CALL_
//__m256d decx::conv::CPUK::_conv2_fp64_loop_in_kernel_2regs_rw4(const double* src, const double* kernel, const uint2 ker_dims, const size_t Wsrc)
//{
//    __m256d _proc_reg, _store_reg,
//        _accumulator = _mm256_set1_pd(0);
//
//    float k_value;      // kernel value
//    uint ker_dex = 0;
//
//    for (int i = 0; i < ker_dims.y; ++i)
//    {
//        _proc_reg = _mm256_load_pd(src + i * (Wsrc << 2));
//        _store_reg = _mm256_load_pd(src + i * (Wsrc << 2) + 4);
//        k_value = kernel[ker_dex];
//        // first multiply-add with the first element of kernel on every row
//        _accumulator = _mm256_fmadd_pd(_proc_reg, _mm256_set1_pd(k_value), _accumulator);
//        ++ker_dex;
//        _CONV2_REGS_FP64_SHIFT_LOAD_0;         _CONV2_REGS_FP64_SHIFT_LOAD_1;
//        _CONV2_REGS_FP64_SHIFT_LOAD_2;         _CONV2_REGS_FP64_SHIFT_LOAD_3;
//    }
//    return _accumulator;
//}
//
//
//
//_THREAD_CALL_
//__m256d decx::conv::CPUK::_conv2_fp64_loop_in_kernel_Nregs_rw2(const double* src, const double* kernel, const uint2 ker_dims,
//    const size_t Wsrc, const uint _loop)
//{
//    __m256d _proc_reg, _store_reg,
//        _accumulator = _mm256_set1_pd(0);
//
//    float k_value;      // kernel value
//    uint ker_dex = 0;
//
//    for (int i = 0; i < ker_dims.y; ++i)
//    {
//        _proc_reg = _mm256_load_pd(src + i * (Wsrc << 2));
//        _store_reg = _mm256_load_pd(src + i * (Wsrc << 2) + 4);
//        k_value = kernel[ker_dex];
//        // first multiply-add with the first element of kernel on every row
//        _accumulator = _mm256_fmadd_pd(_proc_reg, _mm256_set1_pd(k_value), _accumulator);
//        ++ker_dex;
//        
//        _CONV2_REGS_FP64_SHIFT_LOAD_0;         _CONV2_REGS_FP64_SHIFT_LOAD_1;
//        _CONV2_REGS_FP64_SHIFT_LOAD_2;         _CONV2_REGS_FP64_SHIFT_LOAD_3;
//
//        for (int _L = 1; _L < _loop + 1; ++_L){
//            _store_reg = _mm256_load_pd(src + i * (Wsrc << 2) + 4 + (_L << 2));
//            _CONV2_REGS_FP64_SHIFT_LOAD_0;         _CONV2_REGS_FP64_SHIFT_LOAD_1;
//            _CONV2_REGS_FP64_SHIFT_LOAD_2;         _CONV2_REGS_FP64_SHIFT_LOAD_3;
//        }
//
//        _store_reg = _mm256_load_pd(src + i * (Wsrc << 2) + ((_loop + 2) << 2));
//        _CONV2_REGS_FP64_SHIFT_LOAD_0;         _CONV2_REGS_FP64_SHIFT_LOAD_1;
//    }
//    return _accumulator;
//}
//
//
//
//
//_THREAD_CALL_
//__m256d decx::conv::CPUK::_conv2_fp64_loop_in_kernel_Nregs_rw4(const double* src, const double* kernel, const uint2 ker_dims, 
//    const size_t Wsrc, const uint _loop)
//{
//    __m256d _proc_reg, _store_reg,
//        _accumulator = _mm256_set1_pd(0);
//
//    float k_value;      // kernel value
//    uint ker_dex = 0;
//
//    for (int i = 0; i < ker_dims.y; ++i)
//    {
//        _proc_reg = _mm256_load_pd(src + i * (Wsrc << 2));
//        _store_reg = _mm256_load_pd(src + i * (Wsrc << 2) + 4);
//        k_value = kernel[ker_dex];
//        // first multiply-add with the first element of kernel on every row
//        _accumulator = _mm256_fmadd_pd(_proc_reg, _mm256_set1_pd(k_value), _accumulator);
//        ++ker_dex;
//        _CONV2_REGS_FP64_SHIFT_LOAD_0;         _CONV2_REGS_FP64_SHIFT_LOAD_1;
//        _CONV2_REGS_FP64_SHIFT_LOAD_2;         _CONV2_REGS_FP64_SHIFT_LOAD_3;
//
//        for (int _L = 1; _L < _loop + 1; ++_L) {
//            _store_reg = _mm256_load_pd(src + i * (Wsrc << 2) + 4 + (_L << 2));
//            _CONV2_REGS_FP64_SHIFT_LOAD_0;         _CONV2_REGS_FP64_SHIFT_LOAD_1;
//            _CONV2_REGS_FP64_SHIFT_LOAD_2;         _CONV2_REGS_FP64_SHIFT_LOAD_3;
//        }
//
//        _store_reg = _mm256_load_pd(src + i * (Wsrc << 2) + ((_loop + 2) << 2));
//        _CONV2_REGS_FP64_SHIFT_LOAD_0;         _CONV2_REGS_FP64_SHIFT_LOAD_1;
//        _CONV2_REGS_FP64_SHIFT_LOAD_2;         _CONV2_REGS_FP64_SHIFT_LOAD_3;
//    }
//    return _accumulator;
//}
//
//
//
//#endif