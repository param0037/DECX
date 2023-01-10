/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/backup_1
*/

#ifndef _CUDASTREAM_QUEUE_CUH_
#define _CUDASTREAM_QUEUE_CUH_


#include "../basic.h"
#include "cudaStream_package.h"
#include "../decx_alloc_interface.h"
#include "../memory_management/PtrInfo.h"


#define _CS_STREAM_Q_INIT_SIZE_ 10


namespace decx
{
    class cudaStream_Queue;
}


class decx::cudaStream_Queue
{
private:
    size_t true_capacity;

    decx::PtrInfo<decx::cuda_stream> _cuda_stream_arr;

    /*
    * This function will not label the found cuda_stream as occupied
    * @param uint *res_dex : This is the pointer of result index
    * @param const int flag : This is the flag of cudaStream
    */
    bool _find_idle_stream(uint *res_dex, const int flag);


    decx::cuda_stream* add_stream_physical(const int flag);

public:
    size_t _cuda_stream_num;


    cudaStream_Queue();


    decx::cuda_stream* stream_accessor_ptr(const int flag);


    decx::cuda_stream& stream_accessor_ref(const int flag);


    void release();


    ~cudaStream_Queue() {}
};




namespace decx
{
    extern decx::cudaStream_Queue CStream;
}



#endif