/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#include "basic.h"
#include "cudaStream_management/cudaStream_queue.h"
#include "cudaStream_management/cudaEvent_queue.h"
#include "configs/config.h"



decx::cudaStream_Queue decx::CStream;
decx::cudaEvent_Queue decx::CEvent;
decx::cudaProp decx::cuP;


//uchar decx::Const_Mem[CONSTANT_MEM_SIZE];