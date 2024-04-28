/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CPU_FFT_DEFS_H_
#define _CPU_FFT_DEFS_H_

// The fragment allocate length is _MAX_TILING_CPU_FFT_
#define _MAX_TILING_CPU_FFT_FP32_ 1920
#define _MIN_TILING_CPU_FFT_FP32_ _MAX_TILING_CPU_FFT_FP32_ / 2

#define _MAX_TILING_CPU_FFT_FP64_ 960
#define _MIN_TILING_CPU_FFT_FP64_ _MAX_TILING_CPU_FFT_FP64_ / 2

#define _CPU_FFT_PROC_ALIGN_(__type) (256 / (sizeof(__type) * 2 * 8))

#endif