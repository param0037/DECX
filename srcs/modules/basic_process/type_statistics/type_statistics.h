/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/


#ifndef _TYPE_STATISTIC_H_
#define _TYPE_STATISTIC_H_

#ifdef _DECX_CUDA_CODES_
#include "cuda_summing.cuh"
#include "cuda_summing_fp16.cuh"
#include "cuda_maximum.cuh"
#include "cuda_minimum.cuh"
#endif


#ifdef _DECX_CPU_CODES_
#include "cpu_summing.h"
#include "cpu_maximum.h"
#include "cpu_minimum.h"
#endif


#endif