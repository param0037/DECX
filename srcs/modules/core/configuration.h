/**
*    ---------------------------------------------------------------------
*    Author : Wayne
*   Date   : 2021.04.16
*    ---------------------------------------------------------------------
*    This is a part of the open source program named "DECX", copyright c Wayne,
*    2021.04.16
*/

#pragma once

#ifdef _DECX_CUDA_CODES_
#define _BLOCK_DEFAULT_ 32

#define _SHARED_MEM_SIZE_ 49152

#define __DEVICE__ 0


#define _MTPB_ 1024
#define _double_MTPB_ 2048
#define _MTPB_offset 1025
#define _MTPB_offset_2 1026
#define _half_MTPB_offset 513
#define _half_MTPB_ 512

#define _SQRT_MTPB_ 32



#define _CUDA_WARP_SIZE_ 32
#endif