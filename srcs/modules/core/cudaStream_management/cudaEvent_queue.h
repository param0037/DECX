/**
*   ---------------------------------------------------------------------
*   Author : Wayne Anderson
*   Date   : 2021.04.16
*   ---------------------------------------------------------------------
*   This is a part of the open source program named "DECX", copyright c Wayne,
*   2021.04.16, all right reserved.
*   More information please visit https://github.com/param0037/DECX
*/

#ifndef _CUDAEVENT_QUEUE_CUH_
#define _CUDAEVENT_QUEUE_CUH_


#include "../basic.h"
#include "cudaEvent_package.h"
#include "../decx_alloc_interface.h"
#include "../memory_management/PtrInfo.h"


#define _CS_STREAM_Q_INIT_SIZE_ 10

#ifdef _DECX_CORE_CUDA_
namespace decx
{
    class cudaEvent_Queue;
}


class decx::cudaEvent_Queue
{
private:
    std::mutex _mtx;

    size_t true_capacity;

    decx::PtrInfo<decx::cuda_event> _cuda_event_arr;

    /*
    * This function will not label the found cuda_stream as occupied
    * @param uint *res_dex : This is the pointer of result index
    * @param const int flag : This is the flag of cudaStream
    */
    bool _find_idle_event(uint *res_dex, const int flag);


    decx::cuda_event* add_event_physical(const int flag);

public:
    size_t _cuda_event_num;


    cudaEvent_Queue();


    decx::cuda_event* event_accessor_ptr(const int flag);


    decx::cuda_event& event_accessor_ref(const int flag);


    void release();


    ~cudaEvent_Queue();
};




namespace decx
{
    extern decx::cudaEvent_Queue CEvent;
}
#endif


namespace decx
{
    namespace cuda
    {
        _DECX_API_ decx::cuda_event* get_cuda_event_ptr(const int flag);


        _DECX_API_ decx::cuda_event& get_cuda_event_ref(const int flag);
    }
}


#endif